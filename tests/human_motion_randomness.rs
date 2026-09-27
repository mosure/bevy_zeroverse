//! Bounded real-weight regression: compare decoded poses, not seed/provenance text.
#![cfg(all(feature = "human_motion", not(target_arch = "wasm32")))]

use anyhow::{ensure, Result};
use bevy::prelude::Vec3;
use bevy_zeroverse::human_motion::{planning::motion_seed, HumanMotionConfig};
use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup_async, WgpuDevice};
use burn_ardy::Ardy;
use burn_human_inference::gpu::WgpuBackend;
use burn_human_motion::{MotionClip, Waypoint};
use burn_llama::TextEncoder;
use serde::Serialize;
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Debug, Serialize)]
struct Difference {
    /// Remove root translation AND orientation before comparing articulated joints.
    articulation_rms_m: f64,
    articulation_max_m: f32,
    frames_with_articulation_difference_over_1mm: usize,
    root_rms_m: f64,
    identical_frames: bool,
}

fn compare(a: &MotionClip, b: &MotionClip) -> Result<Difference> {
    a.validate()?;
    b.validate()?;
    ensure!(a.frames.len() == b.frames.len() && a.fps == b.fps);
    ensure!(serde_json::to_value(&a.rig)? == serde_json::to_value(&b.rig)?);
    let mut sum = 0.0;
    let mut root_sum = 0.0;
    let mut max = 0.0_f32;
    let mut changed = 0;
    let joints = a.rig.joints.len() - 1;
    for (a_frame, b_frame) in a.frames.iter().zip(&b.frames) {
        let (a_pos, _) = a.rig.forward(a_frame)?;
        let (b_pos, _) = b.rig.forward(b_frame)?;
        let mut frame_sum = 0.0;
        for (a_pos, b_pos) in a_pos.iter().zip(&b_pos).skip(1) {
            let a_local =
                a_frame.local_rotations[0].inverse() * (*a_pos - a_frame.root_translation);
            let b_local =
                b_frame.local_rotations[0].inverse() * (*b_pos - b_frame.root_translation);
            let distance = a_local.distance(b_local);
            frame_sum += f64::from(distance).powi(2);
            max = max.max(distance);
        }
        changed += usize::from((frame_sum / joints as f64).sqrt() > 0.001);
        sum += frame_sum;
        root_sum += f64::from(a_frame.root_translation.distance(b_frame.root_translation)).powi(2);
    }
    Ok(Difference {
        articulation_rms_m: (sum / (a.frames.len() * joints) as f64).sqrt(),
        articulation_max_m: max,
        frames_with_articulation_difference_over_1mm: changed,
        root_rms_m: (root_sum / a.frames.len() as f64).sqrt(),
        // Provenance contains the seed, so it must NOT be used as evidence that
        // two generated motions differ. Compare actual frame content instead.
        identical_frames: serde_json::to_vec(&a.frames)? == serde_json::to_vec(&b.frames)?,
    })
}

#[test]
#[ignore = "requires native GPU and cached/downloaded ARDY + Llama weights"]
fn cached_prompts_preserve_seed_diversity_and_replay_across_batches() -> Result<()> {
    pollster::block_on(async {
        let started = Instant::now();
        let device = WgpuDevice::default();
        let setup = init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
        let adapter = setup.adapter.get_info();
        eprintln!(
            "Motion randomness qualification: {} {:?}",
            adapter.name, adapter.backend
        );
        let ardy = Ardy::<WgpuBackend>::load_pretrained(&device, |_, _| {}).await?;
        let mut text = TextEncoder::<WgpuBackend>::load_pretrained(&device, |_, _| {}).await?;
        let prompts = [
            "A person walks forward with a natural arm swing.",
            "A person stands and waves the right hand.",
        ];
        // Encode each prompt once. Every request for it reuses identical pooled
        // text features, exactly as the viewer's embedding cache does.
        let mut encoded = Vec::new();
        for prompt in prompts {
            encoded.push(text.encode(prompt, |_, _| {}).await?);
        }
        let config = HumanMotionConfig::default();
        let seed_a = motion_seed(0, 7);
        let seed_b = motion_seed(1, 7);
        let seed_c = motion_seed(0, 8);
        let cases = [
            (0, seed_a),
            (0, seed_b),
            (0, seed_c),
            (1, seed_a),
            (1, seed_b),
            (1, seed_a ^ (1 << 63)), // catch accidental u32 seed truncation
            (0, seed_a),             // identical requests in different lanes of one batch
            (1, seed_a),
        ];
        let requests: Vec<_> = cases
            .iter()
            .map(|&(prompt, seed)| {
                let end_z = if prompt == 0 { 3.0 } else { 0.0 };
                config.request(
                    prompts[prompt].into(),
                    seed,
                    vec![
                        Waypoint {
                            frame: 0,
                            position: Vec3::new(0.0, 0.94, 0.0),
                            heading: Some(0.0),
                            constrain_height: true,
                        },
                        Waypoint {
                            frame: config.frames - 1,
                            position: Vec3::new(0.0, 0.94, end_z),
                            heading: Some(0.0),
                            constrain_height: true,
                        },
                    ],
                )
            })
            .collect();
        let embeddings: Vec<_> = cases.iter().map(|&(i, _)| encoded[i].clone()).collect();
        eprintln!("Models ready; generating eight same-prompt/seed control clips.");
        let inference_started = Instant::now();
        let clips = ardy
            .generate_batch(&requests, &embeddings, |_| true)
            .await?;
        let first_batch_seconds = inference_started.elapsed().as_secs_f64();
        let mut differences = Vec::new();
        for (a, b) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)] {
            let diff = compare(&clips[a], &clips[b])?;
            ensure!(
                !diff.identical_frames
                    && diff.articulation_rms_m > 0.001
                    && diff.frames_with_articulation_difference_over_1mm > config.frames / 2,
                "same-prompt seed diversity collapsed for {a}/{b}: {diff:?}"
            );
            differences.push(json!({"cases":[a,b],"difference":diff}));
        }
        let mut replays = Vec::new();
        for (a, b) in [(0, 6), (3, 7)] {
            let diff = compare(&clips[a], &clips[b])?;
            ensure!(
                diff.identical_frames,
                "same-seed batch lanes differ: {diff:?}"
            );
            replays.push(json!({"kind":"duplicate_batch_lane","cases":[a,b],"difference":diff}));
        }
        eprintln!("Seed diversity passed; checking reordered replay and a serial request.");
        let reversed_requests: Vec<_> = requests.iter().rev().cloned().collect();
        let reversed_embeddings: Vec<_> = embeddings.iter().rev().cloned().collect();
        let reversed = ardy
            .generate_batch(&reversed_requests, &reversed_embeddings, |_| true)
            .await?;
        for (i, (a, b)) in clips.iter().zip(reversed.iter().rev()).enumerate() {
            let diff = compare(a, b)?;
            ensure!(
                diff.identical_frames,
                "reordered replay differs at {i}: {diff:?}"
            );
            replays.push(json!({"kind":"reordered_replay","case":i,"difference":diff}));
        }
        let serial = ardy
            .generate(&requests[0], &embeddings[0], |_| true)
            .await?;
        let diff = compare(&clips[0], &serial)?;
        ensure!(
            diff.identical_frames,
            "serial/batch replay differs: {diff:?}"
        );
        replays.push(json!({"kind":"serial_replay","case":0,"difference":diff}));

        let output = std::env::var_os("ZEROVERSE_MOTION_SEED_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("out/motion_seed_validation"));
        std::fs::create_dir_all(&output)?;
        let frame_hashes: Vec<_> = clips
            .iter()
            .map(|clip| {
                serde_json::to_vec(&clip.frames)
                    .map(|bytes| burn_human_motion::artifacts::sha256(&bytes))
            })
            .collect::<Result<_, _>>()?;
        std::fs::write(
            output.join("summary.json"),
            serde_json::to_vec_pretty(&json!({
                "capture_engine":bevy_zeroverse::CAPTURE_ENGINE_IDENTITY,
                "device":adapter.name,"backend":format!("{:?}",adapter.backend),
                "driver":adapter.driver_info,"model_loads":1,"text_encodes":2,
                "model_manifests":{"ardy":burn_ardy::pretrained::DEFAULT.sha256,
                    "llama":burn_llama::pretrained::DEFAULT.sha256},
                "requests":requests,"frames_sha256":frame_hashes,
                "different_seed_pairs":differences,"same_seed_replays":replays,
                "first_batch_seconds":first_batch_seconds,
                "seconds_including_models_and_checks":started.elapsed().as_secs_f64(),
                "limits":{"diversity_minimum_root_local_joint_rms_m":0.001,
                    "minimum_fraction_frames_over_1mm":0.5,"replay":"exact frame content"},
                "scope":"Native real-weight randomness check, before scene admission; not action fidelity or cross-device determinism."
            }))?,
        )?;
        std::fs::write(output.join("clips.json"), serde_json::to_vec(&clips)?)?;
        eprintln!(
            "Passed: six distinct-seed comparisons, eleven exact replays; {}",
            output.display()
        );
        Ok(())
    })
}
