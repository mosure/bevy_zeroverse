//! Public camera-factor API across the four-view reconstruction default.
use bevy_zeroverse::scene::procedural_indoor::{
    cameras::{handheld::HandheldSettings, mixture::OverlapMixture, CameraSettings},
    layout::{IndoorLayout, IndoorManifest},
    validation::validate_layout,
};

#[test]
fn four_view_mixtures_and_handheld_paths_keep_the_requested_strata() {
    for seed in 0..16 {
        for handheld in [None, Some(HandheldSettings::default())] {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            let settings = CameraSettings {
                overlap_mixture: Some(OverlapMixture::default()),
                handheld,
                duration_seconds: Some(1.5),
                path_length_min: 0.01,
                path_length_max: 0.6,
                long_path_fraction: 0.,
                ..Default::default()
            };
            let requested =
                serde_json::to_value(settings.overlap_mixture.as_ref().unwrap().targets(seed, 4))
                    .unwrap();
            scene
                .resample_cameras(4, settings, if seed % 2 == 0 { 1.6 } else { 0.75 })
                .unwrap();
            validate_layout(&scene).unwrap();
            assert_eq!(scene.cameras.len(), 4);
            assert_eq!(
                serde_json::to_value(scene.camera_pair_targets()).unwrap(),
                requested
            );
            assert!(scene
                .camera_overlap()
                .iter()
                .all(|pair| scene.accepts_camera_pair(pair)));
        }
    }
}
