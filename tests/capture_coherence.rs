//! Captures must own trajectory time even when the viewer configuration is animated.
#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]

use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureFormat},
        renderer::RenderDevice,
    },
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{Playback, PlaybackMode, ZeroverseCamera},
    headless::{create_app, setup_globals},
    io::{channels, image_copy::ImageCopier},
    render::{depth::DepthFormat, ground_truth::GroundTruthCamera, RenderMode},
    sample::{AnnotationPrecision, CaptureFailure, CaptureProgress, Sample, SamplerState},
    scene::{
        procedural_indoor::{
            gi::IndoorGiSettings, layout::IndoorManifest,
            validation::validate_annotations_with_precision, IndoorQuality,
        },
        SceneAabbNode, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

fn capture(app: &mut App, config: &BevyZeroverseConfig) -> Sample {
    let previous = Playback {
        mode: PlaybackMode::Sin,
        progress: 0.37,
        speed: 9.0,
        direction: -1.0,
    };
    *app.world_mut().resource_mut::<Playback>() = previous;
    let root = app
        .world_mut()
        .query_filtered::<&GlobalTransform, With<SceneAabbNode>>()
        .single(app.world())
        .unwrap()
        .to_matrix();
    let mut state = SamplerState::from_config(config);
    state.regenerate_scene = false;
    app.insert_resource(state);
    let start = Instant::now();
    while app.world().resource::<SamplerState>().enabled {
        app.update();
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "{:?}",
            app.world().resource::<CaptureFailure>()
        );
        assert!(start.elapsed() < Duration::from_secs(90), "capture stalled");
    }
    assert_eq!(
        *app.world().resource::<Playback>(),
        previous,
        "viewer playback was not restored"
    );
    let sample = channels::sample_receiver()
        .unwrap()
        .lock()
        .unwrap()
        .try_recv()
        .unwrap();
    assert_eq!(sample.views.len(), config.playback_steps as usize);
    let manifest = sample.indoor.as_ref().unwrap();
    for (index, view) in sample.views.iter().enumerate() {
        let time = index as f32 * config.playback_step;
        assert_eq!(
            view.time, time,
            "wall time leaked into sampler trajectory time"
        );
        let expected = root * manifest.cameras[0].transform_at(time).to_matrix();
        let actual = Mat4::from_cols_array_2d(&view.world_from_view);
        let error = actual
            .to_cols_array()
            .into_iter()
            .zip(expected.to_cols_array())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            error < 1e-5,
            "camera did not use requested normalized progress: {error}"
        );
        let alignment = validate_annotations_with_precision(
            view,
            sample.aabb,
            config.width as u32,
            config.height as u32,
            sample.annotation_precision,
        )
        .unwrap();
        println!(
            "{:?} step={index} time={time}: {:?}",
            sample.annotation_precision, alignment
        );
    }
    sample
}

#[test]
#[ignore = "requires native GPU; checks async metadata and legacy sequential modalities"]
fn animated_viewer_settings_capture_fixed_steps_and_reject_midflight_changes() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(6),
        indoor_quality: IndoorQuality::Portable,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        image_copiers: true,
        num_cameras: 1,
        width: 161.0,
        height: 123.0,
        playback_mode: PlaybackMode::Sin,
        playback_speed: 9.0,
        yaw_speed: 0.8,
        playback_steps: 3,
        playback_step: 0.4,
        depth_format: DepthFormat::Linear,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
        ],
        ..default()
    };
    let mut app = create_app(None, Some(config.clone()), false);
    app.insert_resource(IndoorGiSettings {
        enabled: false,
        ..default()
    });
    app.finish();
    app.cleanup();
    for _ in 0..16 {
        app.update();
    }
    assert_eq!(app.world().resource::<CaptureProgress>().backoff_sleeps, 0);
    assert_eq!(
        app.world().resource::<IndoorManifest>().generator_version,
        bevy_zeroverse::scene::procedural_indoor::layout::GENERATOR_VERSION
    );
    let native = capture(&mut app, &config);
    assert_eq!(
        native.annotation_precision,
        AnnotationPrecision::Float32Geometry
    );

    // Exercise the preserved one-attachment renderer and sequential material modes.
    let (entity, target) = app
        .world_mut()
        .query_filtered::<(Entity, &RenderTarget), With<ZeroverseCamera>>()
        .single(app.world())
        .map(|(entity, target)| {
            let RenderTarget::Image(target) = target else {
                panic!()
            };
            (entity, target.handle.clone())
        })
        .unwrap();
    let copier = ImageCopier::for_targets(
        vec![target],
        Extent3d {
            width: config.width as u32,
            height: config.height as u32,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        app.world().resource::<RenderDevice>(),
    );
    app.world_mut()
        .entity_mut(entity)
        .remove::<GroundTruthCamera>()
        .insert(copier.clone());
    let legacy = capture(&mut app, &config);
    assert_eq!(legacy.annotation_precision, AnnotationPrecision::Float16Hdr);

    // A camera change after issuing a copy must fail, never relabel the old pixels.
    let previous = *app.world().resource::<Playback>();
    let mut state = SamplerState::from_config(&config);
    state.regenerate_scene = false;
    app.insert_resource(state);
    let old_id = copier.requested_id();
    let start = Instant::now();
    while copier.requested_id() == old_id {
        app.update();
        assert!(start.elapsed() < Duration::from_secs(30));
    }
    if let Projection::Perspective(p) = &mut *app.world_mut().get_mut::<Projection>(entity).unwrap()
    {
        p.fov += 0.1;
    }
    app.update();
    assert!(!app.world().resource::<SamplerState>().enabled);
    assert!(app
        .world()
        .resource::<CaptureFailure>()
        .0
        .as_deref()
        .unwrap()
        .contains("camera pose/projection/time changed"));
    assert_eq!(*app.world().resource::<Playback>(), previous);
    assert!(channels::sample_receiver()
        .unwrap()
        .lock()
        .unwrap()
        .try_recv()
        .is_err());

    // Regeneration replaces failed/inactive cameras and starts a fresh request epoch.
    app.world_mut().resource_mut::<CaptureFailure>().0 = None;
    app.world_mut()
        .write_message(bevy_zeroverse::scene::RegenerateSceneEvent);
    for _ in 0..4 {
        app.update();
    }
    let regenerated = capture(&mut app, &config);
    assert_eq!(regenerated.indoor.as_ref().unwrap().seed, 7);
    assert_eq!(
        regenerated.annotation_precision,
        AnnotationPrecision::Float32Geometry
    );

    // The interactive configuration keeps its normal application cadence even
    // with a sampler attached. This also covers the no-GI viewer path explicitly.
    let previous_sleeps = app.world().resource::<CaptureProgress>().backoff_sleeps;
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .headless = false;
    for mut camera in app
        .world_mut()
        .query_filtered::<&mut Camera, With<ZeroverseCamera>>()
        .iter_mut(app.world_mut())
    {
        camera.is_active = true;
    }
    let viewer = capture(&mut app, &config);
    assert_eq!(
        viewer.annotation_precision,
        AnnotationPrecision::Float32Geometry
    );
    assert_eq!(
        app.world().resource::<CaptureProgress>().backoff_sleeps,
        previous_sleeps,
        "interactive capture must not enter the headless polling window"
    );
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .headless = true;

    // Low-level callers may change the requested modes after native three-plane
    // cameras exist. Unsupported channels must fail, never disappear silently.
    for unsupported in [RenderMode::OpticalFlow, RenderMode::MotionVectors] {
        let previous = *app.world().resource::<Playback>();
        let previous_sleeps = app.world().resource::<CaptureProgress>().backoff_sleeps;
        app.world_mut().resource_mut::<CaptureFailure>().0 = None;
        let mut state = SamplerState::from_config(&config);
        state.regenerate_scene = false;
        state.render_modes.push(unsupported.clone());
        app.insert_resource(state);
        let start = Instant::now();
        while app.world().resource::<SamplerState>().enabled {
            app.update();
            assert!(start.elapsed() < Duration::from_secs(30));
        }
        let failure = app.world().resource::<CaptureFailure>();
        assert!(failure
            .0
            .as_deref()
            .unwrap()
            .contains(&format!("does not support {unsupported:?}")));
        assert_eq!(*app.world().resource::<Playback>(), previous);
        assert_eq!(
            app.world().resource::<CaptureProgress>().backoff_sleeps,
            previous_sleeps,
            "an unsubmitted invalid request must not enter the polling window"
        );
        assert!(channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
            .is_err());
    }
    println!(
        "total native headless polling sleeps: {}",
        app.world().resource::<CaptureProgress>().backoff_sleeps
    );
}
