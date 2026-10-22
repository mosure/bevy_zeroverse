//! Production rejection seeds: eight views, fixed lens/mount assignments and
//! moving ceiling rails. No renderer, device, assets or learned models are used.
use super::*;

#[test]
fn training_rejections_complete_with_original_roles_and_constraints() {
    let policy = CameraSettings::parse(include_str!("training_camera_settings.json")).unwrap();
    for seed in [
        65_185_309, 65_186_191, 65_193_636, 65_193_935, 65_193_994, 65_194_005, 65_194_057,
        65_194_277, 65_194_325, 65_194_372, 65_194_622, 65_194_664, 65_194_816, 65_194_824,
    ] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let original = scene.clone();
        validate_layout(&scene).unwrap();
        scene
            .resample_cameras(8, policy.clone(), 1.)
            .unwrap_or_else(|e| panic!("{e}"));
        validate_layout(&scene).unwrap_or_else(|e| panic!("{e}"));
        assert!(scene.camera_diagnostics.peer_proposals <= 32 * 384 * 8);
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
        if [65_185_309, 65_194_325].contains(&seed) {
            let cameras = scene.cameras.clone();
            scene.resample_cameras(8, policy.clone(), 1.).unwrap();
            assert_eq!(scene.cameras, cameras, "retry must be seed-deterministic");
        }
        // Search must leave architecture, objects, people, motion and materials
        // untouched, including when it explores and backtracks many prefixes.
        scene.cameras = original.cameras.clone();
        scene.camera_diagnostics = original.camera_diagnostics.clone();
        scene.camera_settings = original.camera_settings.clone();
        scene.camera_aspect_ratio = original.camera_aspect_ratio;
        assert_eq!(scene, original);
    }
}

#[test]
fn difficult_training_rigs_preserve_geometry_and_transactional_rejection() {
    use crate::sample::failure::{CaptureError, SceneRejectionKind};
    let policy = CameraSettings::parse(include_str!("training_camera_settings.json")).unwrap();
    for seed in [65_185_819, 65_193_669, 65_194_134, 65_194_694] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validate_layout(&scene).unwrap();
        let before = scene.clone();
        match scene.resample_cameras_typed(8, policy.clone(), 1.) {
            Ok(()) => {
                // Further sampler improvements may legitimately solve these.
                assert_eq!(scene.cameras.len(), 8);
                validate_layout(&scene).unwrap();
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
