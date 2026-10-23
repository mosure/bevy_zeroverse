//! Opt-in reproduction of the published contact-edge replay failure. Run alone:
//! `cargo test --lib render::raster_order::gpu_tests -- --ignored --test-threads=1`
use crate::{
    app::{BevyZeroverseConfig, OvoxelMode},
    camera::PlaybackMode,
    headless,
    io::channels,
    render::{depth::DepthFormat, glass::AnnotationGlass, RenderMode},
    sample::{Sample, SamplerState},
    scene::{
        procedural_indoor::{self, preparation::IndoorPrefetch},
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires an available native GPU; repeated eight-view SSAA/PBR capture"]
fn contact_edge_rgb_and_annotations_replay_with_shadow_batches() {
    qualify_replay(&[65158971, 65286812, 65158971, 65286812]);
}

#[test]
#[ignore = "requires an available native GPU; production mixed-mount rejection seeds"]
fn recovered_training_rigs_capture_all_times_and_annotations() {
    qualify_replay(&[65341346, 65333746, 65332074, 65341346, 65333746, 65332074]);
}

fn qualify_replay(seeds: &[u64]) {
    bevy::app::TaskPoolOptions::with_num_threads(6).create_default_pools();
    procedural_indoor::preparation::configure_preparation_threads(4).unwrap();
    headless::setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
    let config = BevyZeroverseConfig {
        headless: true,
        image_copiers: true,
        editor: false,
        keybinds: false,
        press_esc_close: false,
        playback_mode: PlaybackMode::Still,
        num_cameras: 8,
        width: 512.,
        height: 512.,
        playback_steps: 3,
        playback_step: 0.5,
        render_mode: RenderMode::Color,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Semantic,
            RenderMode::CoVisibility,
        ],
        depth_format: DepthFormat::Linear,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ovoxel_mode: OvoxelMode::Disabled,
        indoor_seed: Some(41920000),
        indoor_density: 0.65,
        indoor_human_density: 0.,
        indoor_camera: Some(
            include_str!("../../../tests/fixtures/raster_replay_camera.json").into(),
        ),
        indoor_appearance: Some(
            r#"{"material_detail":1.0,"illumination_scale":1.0,"exposure_ev100_offset":0.0}"#
                .into(),
        ),
        annotation_glass: AnnotationGlass::Through,
        ..Default::default()
    };
    let mut app = headless::create_app(None, Some(config.clone()), false);
    app.finish();
    app.cleanup();
    let mut references = BTreeMap::<u64, Sample>::new();
    let mut accepted = 0;
    let mut peak_batches = 0;
    for (sequence, &seed) in seeds.iter().enumerate() {
        app.world_mut().resource_mut::<SamplerState>().enabled = false;
        procedural_indoor::reset_indoor_sequence(app.world_mut(), seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
        {
            let mut prefetch = app.world_mut().resource_mut::<IndoorPrefetch>();
            prefetch.depth = 3;
            prefetch.explicit_seeds =
                Some(seeds.iter().skip(sequence + 1).take(2).copied().collect());
        }
        let mut sampler = SamplerState::from_config(&config);
        sampler.regenerate_scene = false;
        app.insert_resource(sampler);
        let start = Instant::now();
        let sample = loop {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "capture timed out for {seed}; readiness={:?}; preparing={}",
                app.world().resource::<crate::sample::CaptureReadiness>(),
                procedural_indoor::indoor_generation_pending(app.world())
            );
            headless::update_capture(&mut app);
            peak_batches = peak_batches.max(crate::render::preview_shadows::active_batches(
                app.world_mut(),
            ));
            if let Some(error) = crate::sample::failure::take_capture_failure(app.world_mut()) {
                panic!("unexpected capture failure for {seed}: {error}");
            }
            if let Ok(sample) = channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                break sample;
            }
        };
        accepted += 1;
        assert_eq!(sample.indoor.as_ref().map(|room| room.seed), Some(seed));
        assert_eq!(sample.views.len(), 24);
        procedural_indoor::validation::validate_layout(sample.indoor.as_ref().unwrap()).unwrap();
        for view in &sample.views {
            for plane in [&view.color, &view.depth, &view.normal] {
                assert!(plane
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .all(f32::is_finite));
                assert!(plane
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
                    .any(|v| v > 0.));
            }
        }
        if let Some(previous) = references.get(&seed) {
            assert_eq!(sample.indoor, previous.indoor);
            for (index, (a, b)) in previous.views.iter().zip(&sample.views).enumerate() {
                assert_eq!(a.calibration, b.calibration);
                assert_eq!(a.world_from_view, b.world_from_view);
                assert_eq!(a.time, b.time);
                for (name, a, b) in [
                    ("color", &a.color, &b.color),
                    ("depth", &a.depth, &b.depth),
                    ("normal", &a.normal, &b.normal),
                    ("semantic", &a.semantic, &b.semantic),
                    ("co_visibility", &a.co_visibility, &b.co_visibility),
                ] {
                    assert!(!a.is_empty());
                    assert!(
                        a == b,
                        "{name} replay differs: seed {seed}, sequence {sequence}, view {index}"
                    );
                }
            }
        } else {
            references.insert(seed, sample);
        }
    }
    assert_eq!(accepted, seeds.len());
    assert!(peak_batches > 0, "replay must exercise shadow batching");
}
