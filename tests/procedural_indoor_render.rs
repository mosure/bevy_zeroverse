//! Run explicitly on a machine with a native graphics adapter.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless::{create_app, setup_globals},
    io::channels,
    render::{color::ColorEncoding, depth::DepthFormat, RenderMode},
    sample::{Sample, SamplerState},
    scene::{
        procedural_indoor::validation::validate_annotations_with_precision, RegenerateSceneEvent,
        ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

fn capture(app: &mut App, modes: Vec<RenderMode>) -> Sample {
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: modes,
        frames: 8,
        warmup_frames: 16,
        timesteps: Vec::new(),
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
                .is_none(),
            "capture failed: {:?}",
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
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
            "capture stalled"
        );
    }
}

#[test]
#[ignore = "requires a native GPU and display server; exercises real shaders and readback"]
fn empty_assets_rotated_odd_size_capture_and_legacy_scene_switches() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let modes = vec![
        RenderMode::Color,
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::Position,
    ];
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(3),
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        keybinds: false,
        press_esc_close: false,
        initialize_scene: false,
        num_cameras: 2,
        width: 321.0,
        height: 239.0,
        render_modes: modes.clone(),
        playback_mode: PlaybackMode::Still,
        playback_steps: 1,
        rotation_augmentation: true,
        depth_format: DepthFormat::Linear,
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    for _ in 0..4 {
        app.update();
    }
    for seed in 3..7 {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        *app.world_mut().resource_mut::<RenderMode>() = RenderMode::Color;
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, modes.clone());
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        assert_eq!(sample.color_encoding, ColorEncoding::TonemappedLinear);
        assert_eq!(sample.views.len(), 2);
        assert!(sample.object_obbs.len() >= sample.indoor.as_ref().unwrap().objects.len());
        for view in &sample.views {
            assert_eq!(view.color.len(), 321 * 239 * 16);
            assert_eq!(view.semantic.len(), view.color.len());
            validate_annotations_with_precision(
                view,
                sample.aabb,
                321,
                239,
                sample.annotation_precision,
            )
            .unwrap();
        }
        assert!(app.world().resource::<Assets<StandardMaterial>>().len() <= 80);
        assert!(app.world().resource::<Assets<Image>>().len() < 80);
    }
    // Regenerate while an annotation material is active. This catches extraction
    // before Bevy's EntitySpecializationTicks bookkeeping (observed on WebGPU).
    for (index, mode) in [
        RenderMode::Semantic,
        RenderMode::Normal,
        RenderMode::Position,
        RenderMode::Depth,
    ]
    .into_iter()
    .enumerate()
    {
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .render_modes = vec![mode.clone()];
        *app.world_mut().resource_mut::<RenderMode>() = mode.clone();
        for _ in 0..12 {
            app.update();
        }
        let seed = 700 + index as u64;
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .indoor_seed = Some(seed);
        app.update();
        app.world_mut().write_message(RegenerateSceneEvent);
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, vec![mode.clone()]);
        assert_eq!(sample.indoor.as_ref().unwrap().seed, seed);
        for view in sample.views {
            let bytes = match mode {
                RenderMode::Semantic => view.semantic,
                RenderMode::Normal => view.normal,
                RenderMode::Position => view.position,
                RenderMode::Depth => view.depth,
                _ => unreachable!(),
            };
            let values: &[f32] = bytemuck::cast_slice(&bytes);
            assert_eq!(values.len(), 321 * 239 * 4);
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().copied().fold(0.0_f32, f32::max) > 0.0);
        }
        println!("annotation {mode:?} regeneration passed");
    }
    for scene in [ZeroverseSceneType::CornellCube, ZeroverseSceneType::Room] {
        {
            let mut args = app.world_mut().resource_mut::<BevyZeroverseConfig>();
            args.scene_type = scene.clone();
            args.render_mode = RenderMode::Color;
            args.render_modes = vec![RenderMode::Color];
        }
        // Allow deferred legacy catalog loading before requesting its scene.
        for _ in 0..4 {
            app.update();
        }
        app.world_mut().write_message(RegenerateSceneEvent);
        for _ in 0..16 {
            app.update();
        }
        let sample = capture(&mut app, vec![RenderMode::Color]);
        assert!(sample.indoor.is_none());
        assert_eq!(sample.color_encoding, ColorEncoding::Legacy);
        for view in sample.views {
            let values: &[f32] = bytemuck::cast_slice(&view.color);
            assert_eq!(values.len(), 321 * 239 * 4);
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(values.iter().copied().fold(0.0_f32, f32::max) > 0.0);
        }
        println!("legacy {scene:?} capture passed");
    }
}
