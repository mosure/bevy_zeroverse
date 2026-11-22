//! Lossless, single-backend oracle. Readback occurs here only to compare the returned tensors.
use super::*;
use crate::generator::admission::CaptureRequest;
use bevy_zeroverse_capture::{calibration::FisheyeFit, request::RgbArtifactReceipt};

#[test]
#[ignore = "requires a Vulkan adapter; runs on software Vulkan as well as hardware"]
fn mixed_roles_match_both_rgb_references() -> Result<()> {
    const SEED: u64 = 70;
    const CAMERAS: usize = 8;
    const STEPS: usize = 3;
    setup_test_assets();
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(SEED),
        indoor_density: 0.25,
        indoor_human_density: 0.,
        indoor_quality: procedural_indoor::IndoorQuality::Portable,
        indoor_camera: Some(
            r#"{"lenses":{"probability":0.6,"circular_probability":1,"elliptical_probability":0}}"#
                .into(),
        ),
        indoor_appearance: Some(r#"{"sensor":{"probability":1,"noise":1}}"#.into()),
        num_cameras: CAMERAS,
        width: 64.,
        height: 64.,
        playback_steps: STEPS as u32,
        playback_step: 0.25,
        ssaa_factor: 2,
        depth_format: DepthFormat::Linear,
        z_depth: true,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
            RenderMode::CoVisibility,
            RenderMode::OpticalFlow,
            RenderMode::MotionVectors,
        ],
        ..default()
    };
    let mixed_rows = vec![vec![0, 2, 5, 7], vec![1, 3, 4, 6], vec![0, 1, 6, 7]];
    let all = vec![(0..CAMERAS).collect::<Vec<_>>(); STEPS];
    let empty = vec![vec![]; STEPS];
    let requests: Vec<_> = [empty.clone(), mixed_rows.clone(), all.clone()]
        .into_iter()
        .enumerate()
        .map(|(sequence, rows)| CaptureRequest {
            sequence,
            seed: SEED,
            rgb_artifact_eligible_cameras_by_timestep: Some(rows),
        })
        .collect();
    let spec = CaptureSpec::from_scene(config).resolve()?;
    let mut session = GpuCaptureBuilder::new(spec).build_session()?;
    let invalid = CaptureRequest {
        rgb_artifact_eligible_cameras_by_timestep: Some(vec![vec![]]),
        ..requests[0].clone()
    };
    assert!(session.try_submit(invalid).is_err());
    assert_eq!(session.outstanding_rooms(), 0);
    session.plan_requests(&requests)?;
    let mut results = Vec::new();
    for req in requests {
        assert!(session.try_submit(req.clone())?);
        let completion = session
            .receive_timeout(Duration::from_secs(180))?
            .context("missing capture")?;
        assert_eq!(completion.request, req);
        let sample = completion.result?;
        let metadata = sample.metadata.indoor_render_metadata.as_ref().unwrap();
        let receipt: RgbArtifactReceipt =
            serde_json::from_value(metadata["rgb_artifacts"].clone())?;
        assert_eq!(
            receipt.eligible_cameras_by_timestep,
            req.rgb_artifact_eligible_cameras_by_timestep.unwrap()
        );
        assert_eq!(
            receipt.applied_cameras_by_timestep,
            receipt.eligible_cameras_by_timestep
        );
        assert_eq!(
            receipt.optics_applied_cameras_by_timestep,
            receipt.eligible_cameras_by_timestep
        );
        let transport = &metadata["gpu_tensor_contract"]["transport"];
        assert_eq!(transport["host_image_bytes"], 0);
        assert_eq!(transport["allocated_host_readback_bytes"], 0);
        assert!(sample.metadata.views.iter().all(|v| v.color.is_empty()));
        results.push(sample);
    }
    let [clean, mixed, augmented]: [GpuSample; 3] = results.try_into().ok().unwrap();
    let scene = clean.metadata.indoor.as_ref().unwrap();
    assert!(
        scene
            .cameras
            .iter()
            .any(|c| c.fisheye.is_some_and(|f| f.fit == FisheyeFit::Circular))
    );
    for sample in [&mixed, &augmented] {
        assert_eq!(
            sample.metadata.views, clean.metadata.views,
            "poses, time and intrinsics"
        );
        assert_eq!(sample.metadata.object_obbs, clean.metadata.object_obbs);
        let metadata = sample.metadata.indoor_render_metadata.as_ref().unwrap();
        let reference = clean.metadata.indoor_render_metadata.as_ref().unwrap();
        assert_eq!(
            metadata["camera_sensor"], reference["camera_sensor"],
            "optics and noise sequence"
        );
        assert_ne!(
            metadata["rgb_artifact_eligibility_sha256"],
            reference["rgb_artifact_eligibility_sha256"]
        );
    }
    let mut transparent = 0;
    for i in 0..CAMERAS * STEPS {
        let c = &clean.views[i];
        let m = &mixed.views[i];
        let a = &augmented.views[i];
        let rgb = |view: &GpuView| values(view.color.clone().unwrap());
        let (baseline, actual, augmented) = (rgb(c), rgb(m), rgb(a));
        assert_ne!(
            baseline, augmented,
            "augmentation had no signal at view {i}"
        );
        for (p, q) in baseline
            .as_chunks::<4>()
            .0
            .iter()
            .zip(augmented.as_chunks::<4>().0)
        {
            assert_eq!(
                p[3].to_bits(),
                q[3].to_bits(),
                "sensor footprint at view {i}"
            );
            if p[3] == 0. {
                assert_eq!(*q, [0.; 4]);
                transparent += 1;
            }
        }
        let eligible = mixed_rows[i / CAMERAS].contains(&(i % CAMERAS));
        equal_plane(
            actual,
            if eligible { augmented } else { baseline },
            &format!("RGB view {i}"),
        );
        for other in [m, a] {
            for (expected, actual) in [
                (&c.world_depth, &other.world_depth),
                (&c.normal_semantic, &other.normal_semantic),
                (&c.co_visibility, &other.co_visibility),
                (&c.optical_flow, &other.optical_flow),
                (&c.motion_vectors, &other.motion_vectors),
            ] {
                equal_plane(
                    values(actual.clone().unwrap()),
                    values(expected.clone().unwrap()),
                    &format!("geometry view {i}"),
                );
            }
        }
    }
    assert!(transparent > 0, "circular lens has off-sensor pixels");
    Ok(())
}
