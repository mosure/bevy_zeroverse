//! Controlled renderer ablations. Run explicitly with a native graphics device.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless::{create_app, setup_globals},
    io::channels,
    render::{color::linear_to_srgb, RenderMode},
    sample::{CaptureFailure, Sample, SamplerState},
    scene::{RegenerateSceneEvent, ZeroverseSceneType},
};
use std::time::{Duration, Instant};

fn capture(app: &mut App) -> Sample {
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color],
        frames: 12,
        warmup_frames: 24,
        timesteps: Vec::new(),
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "{:?}",
            app.world().resource::<CaptureFailure>()
        );
        if let Ok(sample) = channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            return sample;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "lighting capture stalled"
        );
    }
}

#[test]
#[ignore = "requires native GPU/display; writes reproducible lighting ablations"]
fn direct_lighting_and_shadow_maps_affect_rgb() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(6),
        indoor_human_density: 0.0,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: 2,
        width: 640.0,
        height: 480.0,
        playback_steps: 1,
        playback_mode: bevy_zeroverse::camera::PlaybackMode::Still,
        playback_speed: 0.0,
        render_modes: vec![RenderMode::Color],
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    app.world_mut().write_message(RegenerateSceneEvent);
    let baseline = capture(&mut app);
    for mut light in app
        .world_mut()
        .query::<&mut SpotLight>()
        .iter_mut(app.world_mut())
    {
        light.shadow_maps_enabled = false;
    }
    for mut light in app
        .world_mut()
        .query::<&mut PointLight>()
        .iter_mut(app.world_mut())
    {
        light.shadow_maps_enabled = false;
    }
    let no_local_shadows = capture(&mut app);
    for mut light in app
        .world_mut()
        .query::<&mut DirectionalLight>()
        .iter_mut(app.world_mut())
    {
        light.shadow_maps_enabled = false;
    }
    let no_shadows = capture(&mut app);
    for mut light in app
        .world_mut()
        .query::<&mut SpotLight>()
        .iter_mut(app.world_mut())
    {
        light.intensity = 0.0;
    }
    for mut light in app
        .world_mut()
        .query::<&mut PointLight>()
        .iter_mut(app.world_mut())
    {
        light.intensity = 0.0;
    }
    for mut light in app
        .world_mut()
        .query::<&mut DirectionalLight>()
        .iter_mut(app.world_mut())
    {
        light.illuminance = 0.0;
    }
    let no_direct = capture(&mut app);
    let output = std::path::Path::new("out/indoor_lighting");
    std::fs::create_dir_all(output).unwrap();
    let mut metrics = Vec::new();
    for (camera, base) in baseline.views.iter().enumerate() {
        let base_values: &[f32] = bytemuck::cast_slice(&base.color);
        for (label, sample) in [
            ("baseline", &baseline),
            ("no_local_shadows", &no_local_shadows),
            ("no_shadows", &no_shadows),
            ("no_direct", &no_direct),
        ] {
            assert_eq!(
                base.world_from_view, sample.views[camera].world_from_view,
                "lighting ablation cameras must remain fixed"
            );
            let values: &[f32] = bytemuck::cast_slice(&sample.views[camera].color);
            let mut total = 0.0_f64;
            let mut abs_difference = 0.0_f64;
            let mut changed = 0usize;
            let mut bytes = Vec::new();
            for (a, b) in base_values
                .as_chunks::<4>()
                .0
                .iter()
                .zip(values.as_chunks::<4>().0.iter())
            {
                let luma = |p: &[f32]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                let delta = (luma(a) - luma(b)).abs();
                total += luma(b) as f64;
                abs_difference += delta as f64;
                changed += usize::from(delta > 0.005);
                bytes.extend(
                    b[..3]
                        .iter()
                        .map(|v| (linear_to_srgb(*v) * 255.0).round().clamp(0.0, 255.0) as u8),
                );
            }
            let pixels = 640.0 * 480.0;
            image::save_buffer(
                output.join(format!("view_{camera}_{label}.png")),
                &bytes,
                640,
                480,
                image::ColorType::Rgb8,
            )
            .unwrap();
            let metric = serde_json::json!({"camera":camera,"condition":label,"mean_linear_luminance":total/pixels,"mean_absolute_luminance_delta":abs_difference/pixels,"changed_fraction_at_0_005":changed as f64/pixels});
            println!("{metric}");
            if label != "baseline" {
                assert!(
                    abs_difference / pixels > 0.0005,
                    "{label} had no meaningful effect"
                );
            }
            metrics.push(metric);
        }
    }
    std::fs::write(
        output.join("ablation.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
}
