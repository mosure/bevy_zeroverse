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
