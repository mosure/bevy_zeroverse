//! Production rejection seeds: eight views, fixed lens/mount assignments and
//! moving ceiling rails. No renderer, device, assets or learned models are used.
use super::*;

#[test]
fn production_65716664_mixed_baseline() {
    qualify_training_replay(65716664, 32_768);
}

#[test]
fn production_65721965_mixed_baseline() {
    use crate::sample::failure::{CaptureError, SceneRejectionKind};
    let seed = 65721965;
    let policy =
        CameraSettings::parse(include_str!("mixed_training_camera_settings.json")).unwrap();
    let mut scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let before = scene.clone();
    // This sparse primary room still exhausts the fixed 34--39 degree peers'
    // mixed-mount search. A future valid solution is welcome; weakening the
    // lens assignments, motion, overlap or geometry to force one is not.
    match scene.resample_cameras_typed(8, policy.clone(), 1.) {
        Ok(()) => {
            assert_eq!(scene.cameras.len(), 8);
            assert_eq!(scene.camera_settings, policy);
            validate_layout(&scene).unwrap();
        }
        Err(CaptureError::Rejected { rejection, .. }) => {
            eprintln!("BOUNDED {seed}: {}", rejection.reason);
            assert_eq!(rejection.kind, SceneRejectionKind::CameraPlacement);
            assert_eq!(rejection.seed, seed);
            assert_eq!(
                scene, before,
                "no partial rig or modified policy may escape"
            );
        }
        Err(error) => panic!("unexpected failure for {seed}: {error}"),
    }
    crate::scene::procedural_indoor::validation::validate_geometry(&scene).unwrap();
}

#[test]
fn production_65697418_mixed_baseline() {
    qualify_training_replay(65697418, 32_768);
}

#[test]
fn production_65540002_mixed_baseline() {
    qualify_mixed_training_replay(65_540_002);
}

#[test]
fn production_65538610_mixed_baseline() {
    qualify_mixed_training_replay(65_538_610);
}

#[test]
fn production_65547454_mixed_baseline() {
    qualify_mixed_training_replay(65_547_454);
}

#[test]
fn production_65547441_mixed_baseline() {
    qualify_mixed_training_replay(65_547_441);
}

#[test]
fn production_65573964_mixed_baseline() {
    qualify_mixed_training_replay(65_573_964);
}

#[test]
fn production_65573956_mixed_baseline() {
    qualify_mixed_training_replay(65_573_956);
}

#[test]
fn production_65565210_mixed_baseline() {
    qualify_mixed_training_replay(65_565_210);
}

#[test]
fn production_65575252_mixed_baseline() {
    qualify_mixed_training_replay(65575252);
}

#[test]
fn production_65575086_mixed_baseline() {
    qualify_mixed_training_replay(65575086);
}

#[test]
fn production_65574761_mixed_baseline() {
    qualify_mixed_training_replay(65574761);
}

#[test]
fn production_65566556_mixed_baseline() {
    qualify_mixed_training_replay(65566556);
}

fn qualify_mixed_training_replay(seed: u64) {
    qualify_training_replay(seed, 8_192);
}

fn qualify_training_replay(seed: u64, peer_budget: usize) {
    let scene = qualify_mixed_training_seed(seed);
    // Ordinary rigs have 64 anchor attempts; joint mixed rigs use a different
    // bounded search. Assert their proposal counters rather than wall-clock time
    // on shared CI machines. Recovery also has a fixed per-view budget.
    assert!(scene.camera_diagnostics.peer_proposals < peer_budget);
    assert!(scene.camera_diagnostics.reference_proposals < 32_768);
    let mut replay = scene.clone();
    replay
        .resample_cameras(8, scene.camera_settings.clone(), 1.)
        .unwrap();
    assert_eq!(replay, scene);
    crate::scene::procedural_indoor::validation::validate_geometry(&scene).unwrap();
}

// Includes reported outliers plus adjacent scenes, without rendering or
// models. Kept explicit so the ordinary regression suite remains small.
#[test]
#[ignore = "160-room CPU placement/latency qualification"]
fn mixed_training_neighbor_cohort() {
    for seed in (65_538_592..65_538_624)
        .chain(65_539_984..65_540_016)
        .chain(65_547_424..65_547_456)
        .chain(65_565_200..65_565_232)
        .chain(65_573_944..65_573_976)
    {
        qualify_mixed_training_seed(seed);
    }
}

#[test]
#[ignore = "12-room CPU qualification around the October production rejections"]
fn reported_training_neighbor_cohort() {
    for seed in (65716662..65716666)
        .chain(65721963..65721967)
        .chain(65697416..65697420)
        .filter(|s| *s != 65721965)
    {
        qualify_mixed_training_seed(seed);
    }
}

fn qualify_mixed_training_seed(seed: u64) -> IndoorManifest {
    let policy =
        CameraSettings::parse(include_str!("mixed_training_camera_settings.json")).unwrap();
    let mut scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let before = scene.clone();
    let started = std::time::Instant::now();
    scene.resample_cameras(8, policy.clone(), 1.).unwrap();
    eprintln!(
        "MIXED {seed} seconds={:.4} diagnostics={:?}",
        started.elapsed().as_secs_f64(),
        scene.camera_diagnostics
    );
    validate_layout(&scene).unwrap();
    assert_eq!(scene.cameras.len(), 8);
    for (slot, camera) in scene.cameras.iter().enumerate() {
        assert_eq!(
            camera.is_ceiling_mounted(),
            policy.ceiling_mounted(seed, slot)
        );
        assert_eq!(camera.fisheye, policy.lens_for(seed, slot));
        assert!(policy.accepts_motion(seed, slot, camera));
        assert!(scene.camera_curve_clear(camera));
        if slot > 0 && scene.mixed_rig_contract().is_some() {
            let mut prior =
                crate::scene::procedural_indoor::layout::stream(seed, 0x4343_0000 + slot as u64);
            assert_eq!(
                camera.fov_degrees,
                policy.intrinsics_for(seed, slot).sample(&mut prior)
            );
        }
    }
    let mut geometry = scene.clone();
    geometry.cameras = before.cameras.clone();
    geometry.camera_diagnostics = before.camera_diagnostics.clone();
    geometry.camera_settings = before.camera_settings.clone();
    geometry.camera_aspect_ratio = before.camera_aspect_ratio;
    assert_eq!(
        geometry, before,
        "camera search must not change geometry or appearance"
    );
    scene
}

#[test]
fn rejected_training_rooms_build_valid_meshes() {
    for seed in [
        65341346, 65341171, 65340887, 65348959, 65348197, 65339328, 65333746, 65325411, 65332074,
        65573964, 65573956, 65565210, 65575252, 65575086, 65574761, 65566556,
    ] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let report = crate::scene::procedural_indoor::validation::validate_geometry(&scene)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        eprintln!(
            "GEOMETRY {seed}: triangles={} degenerate_triangles={} object_envelopes_checked={}",
            report.triangles, report.degenerate_triangles, report.object_envelopes_checked
        );
    }
}

#[test]
fn difficult_training_rigs_preserve_geometry_and_transactional_rejection() {
    use crate::sample::failure::{CaptureError, SceneRejectionKind};
    let policy = CameraSettings::parse(include_str!("training_camera_settings.json")).unwrap();
    // 65194824 solves on Linux but reaches the bounded search limit on the
    // macOS CI runner. A feasible base room does not guarantee an admissible eight-view
    // rig within this budget on every platform. Both outcomes must preserve
    // the full placement contract; never require a relaxed or partial rig.
    for seed in [65_185_819, 65_193_669, 65_194_134, 65_194_694, 65_194_824] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validate_layout(&scene).unwrap();
        let before = scene.clone();
        match scene.resample_cameras_typed(8, policy.clone(), 1.) {
            Ok(()) => {
                // Further sampler improvements may legitimately solve these.
                assert_eq!(scene.cameras.len(), 8);
                validate_layout(&scene).unwrap();
                for (index, camera) in scene.cameras.iter().enumerate() {
                    assert_eq!(
                        camera.is_ceiling_mounted(),
                        policy.ceiling_mounted(seed, index)
                    );
                    assert_eq!(camera.fisheye, policy.lens_for(seed, index));
                    assert!(policy.accepts_motion(seed, index, camera));
                    assert!(scene.camera_curve_clear(camera));
                    assert!(scene.cameras[..index]
                        .iter()
                        .all(|c| !bodies_overlap(c, camera)));
                }
            }
            Err(CaptureError::Rejected { rejection, .. }) => {
                assert_eq!(rejection.kind, SceneRejectionKind::CameraPlacement);
                assert_eq!(rejection.seed, seed);
                assert_eq!(
                    scene, before,
                    "no partial rig or modified policy may escape"
                );
            }
            Err(error) => panic!("unexpected failure for {seed}: {error}"),
        }
    }
}

// Captured from the production rejection ledger: eight mixed-lens views,
// moving ceiling rails, no people, density 0.65 and the exact archived policy.
fn qualify_production_seed(seed: u64) {
    let policy = CameraSettings::parse(include_str!("training_camera_settings.json")).unwrap();
    let started = std::time::Instant::now();
    let mut scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    validate_layout(&scene).unwrap();
    let before = scene.clone();
    scene
        .resample_cameras(8, policy.clone(), 1.)
        .unwrap_or_else(|e| panic!("{e}"));
    validate_layout(&scene).unwrap();
    assert_eq!(scene.cameras.len(), 8);
    assert_eq!(scene.camera_settings, policy);
    assert!(scene.camera_diagnostics.peer_proposals <= 32 * 384 * 8);
    assert!(
        scene.camera_diagnostics.reference_proposals
            <= 32 * scene.camera_diagnostics.peer_proposals
    );
    for (index, camera) in scene.cameras.iter().enumerate() {
        assert_eq!(
            camera.is_ceiling_mounted(),
            policy.ceiling_mounted(seed, index)
        );
        assert_eq!(camera.fisheye, policy.lens_for(seed, index));
        assert!(policy.accepts_motion(seed, index, camera));
        assert!(scene.camera_curve_clear(camera));
    }
    eprintln!(
        "COHORT {seed} ok {:.3}s proposals={}",
        started.elapsed().as_secs_f64(),
        scene.camera_diagnostics.peer_proposals
    );
    if [65_341_346, 65_333_746, 65_185_309, 65_194_325, 65_459_150].contains(&seed) {
        let cameras = scene.cameras.clone();
        scene.resample_cameras(8, policy, 1.).unwrap();
        assert_eq!(
            scene.cameras, cameras,
            "complete mixed rigs must replay deterministically"
        );
    }
    scene.cameras = before.cameras.clone();
    scene.camera_diagnostics = before.camera_diagnostics.clone();
    scene.camera_settings = before.camera_settings.clone();
    scene.camera_aspect_ratio = before.camera_aspect_ratio;
    assert_eq!(
        scene, before,
        "camera recovery must not change geometry or appearance"
    );
}

macro_rules! production_cases {
    ($($name:ident: $seed:literal),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                qualify_production_seed($seed);
            }
        )*
    };
}
production_cases! {
    production_65459150: 65459150,
    production_65304870: 65304870,
    production_65305524: 65305524,
    production_65305576: 65305576,
    production_65307200: 65307200,
    production_65307756: 65307756,
    production_65307897: 65307897,
    production_65308010: 65308010,
    production_65308064: 65308064,
    production_65308167: 65308167,
    production_65313128: 65313128,
    production_65313598: 65313598,
    production_65313920: 65313920,
    production_65313961: 65313961,
    production_65314768: 65314768,
    production_65315377: 65315377,
    production_65315604: 65315604,
    production_65316301: 65316301,
    production_65321624: 65321624,
    production_65322286: 65322286,
    production_65324214: 65324214,
    production_65325098: 65325098,
    production_65325377: 65325377,
    production_65325411: 65325411,
    production_65330468: 65330468,
    production_65330752: 65330752,
    production_65330786: 65330786,
    production_65331559: 65331559,
    production_65332066: 65332066,
    production_65332074: 65332074,
    production_65332297: 65332297,
    production_65332518: 65332518,
    production_65332524: 65332524,
    production_65332852: 65332852,
    production_65333746: 65333746,
    production_65339328: 65339328,
    production_65340887: 65340887,
    production_65341171: 65341171,
    production_65341346: 65341346,
    production_65342199: 65342199,
    production_65342587: 65342587,
    production_65348197: 65348197,
    production_65348959: 65348959,
}

production_cases! {
    historical_65185309: 65185309,
    historical_65186191: 65186191,
    historical_65193636: 65193636,
    historical_65193935: 65193935,
    historical_65193994: 65193994,
    historical_65194005: 65194005,
    historical_65194057: 65194057,
    historical_65194277: 65194277,
    historical_65194325: 65194325,
    historical_65194372: 65194372,
    historical_65194622: 65194622,
    historical_65194664: 65194664,
    historical_65194816: 65194816,
}
