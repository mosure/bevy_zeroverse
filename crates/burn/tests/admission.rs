//! Each case isolates the process-owned renderer channels/configuration.
use bevy_zeroverse::{
    io::channels,
    render::RenderMode,
    sample::{Sample, View},
    scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest},
};
use bevy_zeroverse_burn::{
    chunk::{discover_chunks, load_chunk},
    generator::{
        GenConfig,
        admission::{CaptureControl, CaptureEvent, CaptureOutcome, CaptureStage},
        run_chunk_generation_controlled,
    },
};
use std::{
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

fn sample(seed: u64) -> Sample {
    let scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.3, 2, 0.).unwrap();
    Sample {
        indoor: Some(scene),
        view_dim: 2,
        human_pose_steps: vec![vec![]; 2],
        views: (0..4)
            .map(|i| View {
                color: bytemuck::cast_slice(&[0.2f32; 4 * 4 * 4]).to_vec(),
                time: (i / 2) as f32 * 0.2,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn archive_admission_contract() {
    let Ok(case) = std::env::var("ZEROVERSE_ADMISSION_CASE") else {
        for case in [
            "cancel_start",
            "hold",
            "cancel_capture",
            "writer",
            "rejected",
            "timeout",
        ] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "archive_admission_contract", "--nocapture"])
                .env("ZEROVERSE_ADMISSION_CASE", case)
                .env("BEVY_ZEROVERSE_FAKE_APP", "1")
                .status()
                .unwrap();
            assert!(status.success(), "case {case}");
        }
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let output = tmp.path().join("archive");
    if case == "writer" {
        std::fs::write(&output, b"not a directory").unwrap();
    }
    let (events_tx, events) = mpsc::channel();
    let control = Arc::new(CaptureControl::with_observer(move |event| {
        events_tx.send(event).unwrap();
    }));
    let config = GenConfig {
        seed: Some(500),
        samples: 2,
        chunk_size: 1,
        width: 4,
        height: 4,
        cameras: 2,
        playback_steps: 2,
        playback_step: 0.2,
        render_modes: vec![RenderMode::Color],
        timeout_secs: 1,
        indoor_human_density: 0.,
        ..GenConfig::indoor(&output, 2)
    };
    if case == "cancel_start" {
        control.cancel();
    }
    channels::init_channels();
    let worker = {
        let c = control.clone();
        thread::spawn(move || run_chunk_generation_controlled(config, c))
    };
    assert!(matches!(
        events.recv_timeout(Duration::from_secs(5)).unwrap(),
        CaptureEvent::Stage {
            stage: CaptureStage::WaitingForDemand,
            ..
        }
    ));
    let requests = channels::app_frame_receiver().unwrap().lock().unwrap();
    if case == "cancel_start" {
        assert!(worker.join().unwrap().is_ok());
        assert!(requests.try_recv().is_err());
        assert_eq!(control.snapshot().admitted, 0);
        return;
    }
    // An intentional wait exceeds the actual capture timeout without expiring it.
    if case == "hold" {
        control.hold(true);
        control.grant_through(1);
        assert!(requests.recv_timeout(Duration::from_millis(1100)).is_err());
        control.hold(false);
    } else {
        control.grant_through(1);
    }
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(request.prefetch_indoor, 1);
    assert_eq!(request.idle_work, channels::IdleWork::CpuOnly);
    if case == "cancel_capture" {
        control.cancel();
    }
    let started = Instant::now();
    if case != "timeout" {
        channels::sample_sender()
            .send(sample(if case == "rejected" { 501 } else { 500 }))
            .unwrap();
    }
    if case == "hold" {
        loop {
            if matches!(
                events.recv_timeout(Duration::from_secs(5)).unwrap(),
                CaptureEvent::Completed {
                    outcome: CaptureOutcome::Published,
                    ..
                }
            ) {
                break;
            }
        }
        assert!(requests.try_recv().is_err());
        control.grant_through(2);
        let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(request.prefetch_indoor, 0, "never prepare beyond the shard");
        channels::sample_sender().send(sample(501)).unwrap();
    }
    let result = worker.join().unwrap();
    if matches!(case.as_str(), "writer" | "rejected" | "timeout") {
        assert!(result.is_err());
        assert_eq!(control.snapshot().admitted, 1);
        assert!(control.snapshot().completed >= 1);
        assert!(
            requests.try_recv().is_err(),
            "rejected seeds cannot be silently recycled"
        );
        if case == "timeout" {
            assert!(started.elapsed() < Duration::from_secs(4));
        }
    } else {
        result.unwrap();
        let expected = if case == "hold" { 2 } else { 1 };
        let chunks = discover_chunks(&output).unwrap();
        assert_eq!(chunks.len(), expected);
        for (i, chunk) in chunks.iter().enumerate() {
            let samples = load_chunk(chunk).unwrap();
            assert_eq!(samples.len(), 1);
            let s = &samples[0];
            assert_eq!(s.indoor.as_ref().unwrap().seed, 500 + i as u64);
            assert_eq!(s.views.len(), 4);
            assert!(
                s.views
                    .iter()
                    .all(|v| v.depth.is_empty() && v.normal.is_empty() && v.semantic.is_empty())
            );
        }
        assert_eq!(control.snapshot().admitted, expected as u64);
        let next = GenConfig {
            seed: Some(500),
            chunk_size: 1,
            ..GenConfig::indoor(&output, 1)
        };
        let started = Instant::now();
        let error =
            run_chunk_generation_controlled(next, Arc::new(CaptureControl::new())).unwrap_err();
        assert!(error.to_string().contains("shut down"));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
