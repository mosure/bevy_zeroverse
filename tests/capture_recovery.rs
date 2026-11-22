//! Native transaction recovery must never reuse the preceding room's pixels.
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::PlaybackMode,
    headless,
    io::channels::{self, AppFrameRequest, IdleWork},
    render::{depth::DepthFormat, RenderMode},
    sample::{
        failure::{CaptureError, SceneRejectionKind},
        Sample,
    },
    scene::{
        procedural_indoor::validation::validate_annotations_with_precision, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

const CAMERA: &str = r#"{"primary_room":false,"path_length_min":1.6,"path_length_max":3.0,"long_path_fraction":0.0,"multiview":{"max_baseline":6.0,"min_baseline":0.4,"min_overlap":0.15,"min_reference_baseline":0.8,"min_spread":0.25,"trajectory_variation":0.3}}"#;

fn capture(seed: u64) -> Result<Sample, CaptureError> {
    channels::app_frame_sender()
        .send(AppFrameRequest {
            indoor_seed: Some(seed),
            rgb_artifact_eligible_cameras_by_timestep: None,
            prefetch_indoor: 2,
            idle_work: IdleWork::CpuOnly,
            recover_scene_rejections: true,
        })
        .unwrap();
    let receiver = channels::sample_receiver().unwrap().lock().unwrap();
    let started = Instant::now();
    loop {
        if let Some(error) = channels::take_capture_error() {
            return Err(error);
        }
        match receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(sample) => return Ok(sample),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                if started.elapsed() < Duration::from_secs(180) => {}
            Err(error) => panic!("capture failed or timed out: {error}"),
        }
    }
}

struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        headless::shutdown_app().unwrap();
    }
}

#[test]
#[ignore = "native GPU: repeated typed rejection retirement and exact multimodal successor replay"]
fn rejected_rooms_retire_and_valid_successors_replay_in_one_renderer() {
    headless::setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
    headless::setup_and_run_app(
        true,
        Some(BevyZeroverseConfig {
            headless: true,
            image_copiers: true,
            editor: false,
            keybinds: false,
            press_esc_close: false,
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(62_236_797),
            indoor_camera: Some(CAMERA.into()),
            indoor_human_density: 0.,
            indoor_gi_rays: 256,
            width: 256.,
            height: 256.,
            num_cameras: 8,
            ssaa_factor: 0,
            playback_mode: PlaybackMode::Still,
            playback_steps: 2,
            playback_step: 0.25,
            depth_format: DepthFormat::Linear,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::Position,
                RenderMode::Semantic,
                RenderMode::OpticalFlow,
                RenderMode::CoVisibility,
            ],
            ..Default::default()
        }),
    );
    let _stop = Stop;
    let reference = capture(62_236_797).unwrap();
    assert_eq!(reference.indoor.as_ref().unwrap().seed, 62_236_797);
    assert_eq!(reference.views.len(), 16);
    let mut rows = Vec::new();
    let cycles = std::env::var("ZEROVERSE_RECOVERY_CYCLES")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(4);
    assert!((1..=128).contains(&cycles));
    for cycle in 0..cycles {
        for (seed, kind) in [
            (62_762_988, SceneRejectionKind::MissingProgramAnchor),
            (62_236_796, SceneRejectionKind::CameraPlacement),
        ] {
            let started = Instant::now();
            let Err(error) = capture(seed) else {
                panic!("regression fixture must be rejected");
            };
            assert!(
                matches!(&error,CaptureError::Rejected{rejection,retired:true} if rejection.seed==seed && rejection.kind==kind),
                "{error:?}"
            );
            rows.push(serde_json::json!({"cycle":cycle,"seed":seed,"error":error,"seconds":started.elapsed().as_secs_f64()}));
        }
        let started = Instant::now();
        let sample = capture(62_236_797).unwrap();
        assert_eq!(sample.indoor, reference.indoor);
        assert_eq!(sample.aabb, reference.aabb);
        assert!(
            sample.views == reference.views,
            "accepted scene pixels, calibration, time or labels changed after recovery"
        );
        for view in &sample.views {
            validate_annotations_with_precision(
                view,
                sample.aabb,
                256,
                256,
                sample.annotation_precision,
            )
            .unwrap();
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let rss_kib = status
            .lines()
            .find(|s| s.starts_with("VmRSS:"))
            .and_then(|s| s.split_whitespace().nth(1))
            .and_then(|s| s.parse::<u64>().ok());
        rows.push(serde_json::json!({"cycle":cycle,"seed":62_236_797,"accepted":true,"seconds":started.elapsed().as_secs_f64(),"rss_kib":rss_kib}));
    }
    if let Ok(path) = std::env::var("ZEROVERSE_RECOVERY_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    }
}

/// Uses a valid explicit motion request whose actor does not exist in this
/// sampled room. This exercises the production generated-scene boundary,
/// without a special failing seed, fault flag, or relaxed geometry validation.
#[test]
#[cfg(feature = "human_motion")]
#[ignore = "native Vulkan (hardware or software): generated rejection retirement and retained RGB/metadata replay"]
fn generated_cpu_failures_retire_without_mutating_retained_samples() {
    use bevy::prelude::*;
    use bevy_zeroverse::{
        sample::{failure::take_capture_failure, CaptureFailure, CaptureProgress, SamplerState},
        scene::{procedural_indoor::reset_indoor_sequence, RegenerateSceneEvent},
    };
    headless::setup_globals(Some(format!("{}/assets", env!("CARGO_MANIFEST_DIR"))));
    bevy_zeroverse::scene::procedural_indoor::preparation::configure_preparation_threads(2)
        .unwrap();
    let config = BevyZeroverseConfig {
        headless: true,
        image_copiers: true,
        editor: false,
        keybinds: false,
        press_esc_close: false,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(66242909),
        indoor_human_density: 0.,
        indoor_camera: Some(r#"{"motion_fraction":0.0,"lenses":{"probability":0.0}}"#.into()),
        indoor_gi_rays: 256,
        width: 64.,
        height: 64.,
        num_cameras: 2,
        playback_mode: PlaybackMode::Still,
        playback_steps: 1,
        depth_format: DepthFormat::Linear,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
        ],
        ..default()
    };
    let mut app = headless::create_app(None, Some(config), false);
    app.finish();
    app.cleanup();

    fn run(app: &mut App, seed: u64, motion: bool) -> Result<Sample, CaptureError> {
        app.world_mut().resource_mut::<BevyZeroverseConfig>().human_motion = motion.then(||
            r#"{"trajectories":[{"actor_id":999,"prompt":"walking","waypoints":[{"frame":0,"position":[0,0,0]},{"frame":40,"position":[0,0,1]}]}]}"#.into());
        reset_indoor_sequence(app.world_mut(), seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
        let mut state = SamplerState::from_config(app.world().resource::<BevyZeroverseConfig>());
        state.regenerate_scene = false;
        app.insert_resource(state);
        let started = Instant::now();
        loop {
            headless::update_capture(app);
            if let Some(error) = take_capture_failure(app.world_mut()) {
                assert!(
                    channels::sample_receiver()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .try_recv()
                        .is_err(),
                    "rejected room published a sample"
                );
                return Err(error);
            }
            if let Ok(sample) = channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                assert!(app.world().resource::<CaptureFailure>().0.is_none());
                return Ok(sample);
            }
            assert!(app.should_exit().is_none(), "renderer exited");
            assert!(
                started.elapsed() < Duration::from_secs(180),
                "native recovery fixture timed out"
            );
        }
    }

    let retained = run(&mut app, 66242909, false).unwrap();
    assert_eq!(retained.indoor.as_ref().unwrap().seed, 66242909);
    assert_eq!(retained.views.len(), 2);
    let original = serde_json::to_vec(&retained).unwrap();
    let completed = app.world().resource::<CaptureProgress>().completed_requests;
    let mut rows = Vec::new();
    for seed in [66242910, 66242911] {
        let started = Instant::now();
        let Err(error) = run(&mut app, seed, true) else {
            panic!("missing actor accepted");
        };
        assert!(
            matches!(&error, CaptureError::Rejected { rejection, retired: true }
            if rejection.seed == seed && rejection.kind == SceneRejectionKind::PlacementClearance
                && rejection.reason.contains("motion planning")),
            "{error:?}"
        );
        assert_eq!(
            app.world().resource::<CaptureProgress>().completed_requests,
            completed
        );
        assert!(app.world().resource::<Sample>().views.is_empty());
        assert_eq!(original, serde_json::to_vec(&retained).unwrap());
        rows.push(serde_json::json!({"seed":seed,"error":error,"seconds":started.elapsed().as_secs_f64()}));
    }
    let successor = run(&mut app, 66242912, false).unwrap();
    assert_eq!(successor.indoor.as_ref().unwrap().seed, 66242912);
    let replay = run(&mut app, 66242909, false).unwrap();
    assert_eq!(replay.indoor, retained.indoor);
    assert_eq!(replay.aabb, retained.aabb);
    assert!(
        replay.views == retained.views,
        "RGB/geometry/calibration changed after retirement"
    );
    for sample in [&retained, &successor, &replay] {
        for view in &sample.views {
            assert!(!view.color.is_empty());
            validate_annotations_with_precision(
                view,
                sample.aabb,
                64,
                64,
                sample.annotation_precision,
            )
            .unwrap();
        }
    }
    assert_eq!(original, serde_json::to_vec(&retained).unwrap());
    assert_eq!(
        app.world()
            .resource::<bevy_zeroverse::human_motion::HumanMotionReport>()
            .model_loads,
        0
    );
    if let Ok(path) = std::env::var("ZEROVERSE_RECOVERY_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&serde_json::json!({
            "passed":true,"rejections":rows,"accepted_captures":3,"calibrated_views":6,
            "retained_consumer_preserved":true,"rgb_metadata_replay_bitwise":true,"motion_model_loads":0,
            "capture_engine":bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
            "adapter":app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|v|format!("{} ({:?}, {:?})", v.name, v.device_type, v.backend))
        })).unwrap()).unwrap();
    }
}
