#![recursion_limit = "256"]
//! Matched real-room raster/geometry qualification, without CPU filtering in the
//! production path. The independent area-average oracle runs only in this test.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{DefaultZeroverseCamera, PlaybackMode},
    headless::{create_app, setup_globals, update_capture},
    io::{channels, image_copy::ImageCopier},
    render::{color::linear_to_srgb, ssaa::SsaaCamera, RenderMode},
    sample::{CaptureFailure, Sample, SamplerState},
    scene::{procedural_indoor::reset_indoor_sequence, RegenerateSceneEvent, ZeroverseSceneType},
};
use std::time::{Duration, Instant};

fn take(app: &mut App, size: u32, factor: u32) -> Sample {
    let mut config = app.world().resource::<BevyZeroverseConfig>().clone();
    config.width = size as f32;
    config.height = size as f32;
    config.ssaa_factor = factor;
    app.insert_resource(config.clone());
    app.world_mut()
        .resource_mut::<DefaultZeroverseCamera>()
        .resolution = Some(UVec2::splat(size));
    reset_indoor_sequence(app.world_mut(), 4700);
    app.world_mut().write_message(RegenerateSceneEvent);
    app.update();
    let mut sampler = SamplerState::from_config(&config);
    sampler.regenerate_scene = false;
    app.insert_resource(sampler);
    let started = Instant::now();
    loop {
        update_capture(app);
        assert!(app.should_exit().is_none(), "renderer exited");
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "{:?}",
            app.world().resource::<CaptureFailure>().0
        );
        if let Ok(sample) = channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            eprintln!(
                "{size}px SSAA {factor}: {:.3}s",
                started.elapsed().as_secs_f64()
            );
            return sample;
        }
        assert!(
            started.elapsed() < Duration::from_secs(180),
            "capture stalled"
        );
    }
}

fn area_average(input: &[u8], size: usize) -> Vec<f32> {
    let input: &[f32] = bytemuck::cast_slice(input);
    let mut output = vec![0.; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            for c in 0..4 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        output[(y * size + x) * 4 + c] +=
                            input[((y * 2 + dy) * size * 2 + x * 2 + dx) * 4 + c] * 0.25;
                    }
                }
            }
        }
    }
    output
}

#[test]
#[ignore = "native GPU: three-view/two-time SSAA RGB oracle, exact geometry, target recycling and metadata"]
fn supersampling_preserves_calibration_labels_flow_and_resolves_rgb_on_gpu() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(4700),
        indoor_human_density: 0.,
        indoor_gi_rays: 256,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: 3,
        width: 256.,
        height: 256.,
        playback_steps: 2,
        playback_step: 0.4,
        playback_mode: PlaybackMode::Still,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
            RenderMode::OpticalFlow,
            RenderMode::CoVisibility,
        ],
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    let original = take(&mut app, 256, 1);
    let readback_bytes = |app: &mut App| {
        app.world_mut()
            .query::<&ImageCopier>()
            .iter(app.world())
            .map(ImageCopier::allocated_readback_bytes)
            .sum::<u64>()
    };
    let bytes = readback_bytes(&mut app);
    let filtered = take(&mut app, 256, 0);
    assert_eq!(
        readback_bytes(&mut app),
        bytes,
        "SSAA must not increase PCIe readback bytes"
    );
    let handles: Vec<_> = app
        .world_mut()
        .query::<&SsaaCamera>()
        .iter(app.world())
        .cloned()
        .collect();
    assert_eq!(handles.len(), 3);
    let images = app.world().resource::<Assets<Image>>();
    for camera in handles {
        assert_eq!(
            images.get(&camera.source).unwrap().size(),
            UVec2::splat(512)
        );
        assert_eq!(
            images.get(&camera.output).unwrap().size(),
            UVec2::splat(256)
        );
    }
    let fixed = take(&mut app, 256, 2);
    assert!(
        filtered.views == fixed.views,
        "Auto must resolve identically to explicit 2× below 720p"
    );
    let reference = take(&mut app, 512, 1);
    let replay = take(&mut app, 256, 1);
    assert_eq!(
        app.world_mut()
            .query::<&SsaaCamera>()
            .iter(app.world())
            .count(),
        0
    );
    assert!(original.sampling.is_none() && replay.sampling.is_none());
    assert_eq!(filtered.sampling.unwrap().ssaa_factor, 2);
    assert_eq!(original.views.len(), 6);
    assert_eq!(original.aabb, filtered.aabb);
    assert_eq!(original.object_obbs, filtered.object_obbs);
    let output = std::path::Path::new("out/ssaa");
    std::fs::create_dir_all(output).unwrap();
    let mut metrics = Vec::new();
    for (index, (((a, b), high), replay)) in original
        .views
        .iter()
        .zip(&filtered.views)
        .zip(&reference.views)
        .zip(&replay.views)
        .enumerate()
    {
        assert_eq!(a.calibration, b.calibration);
        assert_eq!(a.world_from_view, b.world_from_view);
        assert_eq!(a.world_from_view, high.world_from_view);
        assert_eq!(a.trajectory_progress, b.trajectory_progress);
        for (label, x, y, z) in [
            ("depth", &a.depth, &b.depth, &replay.depth),
            ("normal", &a.normal, &b.normal, &replay.normal),
            ("position", &a.position, &b.position, &replay.position),
            ("semantic", &a.semantic, &b.semantic, &replay.semantic),
            (
                "flow",
                &a.optical_flow,
                &b.optical_flow,
                &replay.optical_flow,
            ),
            (
                "co_visibility",
                &a.co_visibility,
                &b.co_visibility,
                &replay.co_visibility,
            ),
        ] {
            assert!(!x.is_empty(), "missing {label}");
            assert!(
                x == y && x == z,
                "view {index}: {label} changed under SSAA or recycling"
            );
        }
        let expected = area_average(&high.color, 256);
        let actual: &[f32] = bytemuck::cast_slice(&b.color);
        assert_eq!(expected.len(), actual.len());
        let mae = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / actual.len() as f32;
        let fraction = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| (*a - *b).abs() > 0.001)
            .count() as f32
            / actual.len() as f32;
        eprintln!("view {index}: GPU box vs independent 512px capture: MAE {mae}, changed fraction {fraction}");
        // Shadow/transparent boundary replay can differ at isolated pixels; a
        // box-filter, viewport, or stale-request defect changes broad regions.
        assert!(
            mae < 1e-5 && fraction < 0.001,
            "view {index}: RGB box oracle diverged"
        );
        metrics.push(serde_json::json!({"view":index,"linear_rgba_mae":mae,"fraction_changed_above_0_001":fraction,"geometry_byte_exact":true}));
        for (name, data) in [
            ("off", bytemuck::cast_slice::<u8, f32>(&a.color)),
            ("2x", actual),
        ] {
            let rgb: Vec<u8> = data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    p[..3]
                        .iter()
                        .map(|v| (linear_to_srgb(*v) * 255.).round().clamp(0., 255.) as u8)
                })
                .collect();
            image::save_buffer(
                output.join(format!("view_{index}_{name}.png")),
                &rgb,
                256,
                256,
                image::ColorType::Rgb8,
            )
            .unwrap();
        }
    }
    std::fs::write(output.join("oracle.json"), serde_json::to_vec_pretty(&serde_json::json!({"seed":4700,"cameras":3,"timesteps":2,"readback_bytes":bytes,"views":metrics})).unwrap()).unwrap();
}
