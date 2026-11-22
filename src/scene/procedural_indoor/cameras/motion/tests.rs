use super::*;
use crate::scene::procedural_indoor::{
    cameras::{handheld::HandheldSettings, multiview::OverlapGraph},
    layout::{stream, IndoorLayout, IndoorManifest},
    validation::validate_layout,
};

#[test]
fn peer_turns_respect_metric_budget_at_every_path_scale() {
    for extent in [0.03, 0.3, 1., 8.] {
        for radius in [0.35, 2., 10.] {
            let mut last = 0.;
            for variation in [0., 0.1, 0.5, 1.] {
                let angle = peer_yaw_limit(extent, radius, variation);
                assert!(angle >= last && angle <= std::f32::consts::PI);
                assert!(2. * extent * (angle * 0.5).sin() <= 0.65 * radius * variation + 1e-6);
                last = angle;
            }
        }
    }
    assert_eq!(peer_yaw_limit(0.2, 5., 0.), 0.);
    assert!(peer_yaw_limit(0.2, 5., 0.5).to_degrees() > 90.);
}

#[test]
fn short_admitted_room_trajectories_include_parallel_and_crossing_headings() {
    let mut angles = Vec::new();
    let settings = CameraSettings {
        path_length_min: 0.10,
        path_length_max: 0.35,
        ..default()
    };
    for seed in 1_400..1_416 {
        let mut scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 0).unwrap();
        scene.resample_cameras(4, settings.clone(), 1.2).unwrap();
        assert!(scene.accepts_camera_geometry() && scene.accepts_camera_visibility());
        assert!(scene.cameras.iter().all(|c| scene.camera_curve_clear(c)));
        for (i, camera) in scene.cameras.iter().enumerate() {
            let travel =
                |c: &IndoorCamera| c.transform_at(1.).translation - c.transform_at(0.).translation;
            for peer in &scene.cameras[..i] {
                let (a, b) = (travel(camera), travel(peer));
                if a.length().min(b.length()) > 0.01 {
                    angles.push(a.angle_between(b).to_degrees());
                }
            }
        }
    }
    assert!(
        angles.iter().filter(|&&a| a < 30.).count() >= 8,
        "{angles:?}"
    );
    assert!(
        angles.iter().filter(|&&a| a > 80.).count() >= 8,
        "{angles:?}"
    );
}

#[test]
fn seeded_roles_are_stable_unbiased_and_monotone() {
    let mut settings = CameraSettings::default();
    assert_eq!(settings.motion_fraction, 1.);
    for fraction in [0., 0.1, 0.5, 0.9, 1.] {
        settings.motion_fraction = fraction;
        let mut moving = 0;
        for seed in 0..2000 {
            for index in 0..12 {
                let role = settings.camera_moves(seed, index);
                moving += usize::from(role);
                let more = CameraSettings {
                    motion_fraction: (fraction + 0.1).min(1.),
                    ..settings.clone()
                };
                assert!(!role || more.camera_moves(seed, index));
                assert_eq!(role, settings.clone().camera_moves(seed, index));
            }
        }
        assert!(
            (moving as f32 / 24000. - fraction).abs() < 0.012,
            "{fraction}: {moving}"
        );
    }
    for fraction in [-1., 1.1, f32::NAN] {
        settings.motion_fraction = fraction;
        assert!(settings.validate().is_err());
    }
    assert_eq!(CameraSettings::parse("{}").unwrap().motion_fraction, 1.);
}

#[test]
fn fixed_camera_has_no_handheld_rotation_drift_or_intrinsic_change() {
    let settings = CameraSettings {
        motion_fraction: 0.,
        handheld: Some(HandheldSettings::default()),
        ..default()
    };
    let mut camera = IndoorCamera {
        roll_radians: 0.,
        ceiling_mount: None,
        fisheye: None,
        start: Vec3::new(1., 2., 3.),
        end: Vec3::new(2., 3., 4.),
        target: Vec3::ZERO,
        principal_point_offset: Some(Vec2::new(0.1, -0.05)),
        fov_degrees: 110.,
        motion: None,
    };
    settings.prepare_motion(0, 0, &mut camera, &mut stream(0, 3));
    let pose = camera.transform_at(0.);
    let calibration = (camera.fov_degrees, camera.principal_point_offset);
    for i in 0..=100 {
        assert_eq!(pose, camera.transform_at(i as f32 / 100.));
        assert_eq!(
            calibration,
            (camera.fov_degrees, camera.principal_point_offset)
        );
    }
    assert_eq!(camera.path_length(), 0.);
    assert!(settings.accepts_motion(0, 0, &camera));
}

#[test]
fn mixed_rigs_preserve_roles_replay_clearance_and_overlap() {
    let mut saw_fixed_anchor = false;
    let mut saw_moving_anchor = false;
    for seed in 0..6 {
        let source =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        for graph in [
            None,
            Some(OverlapGraph::Reference),
            Some(OverlapGraph::Connected),
        ] {
            for fraction in [0., 0.5, 1.] {
                let mut settings = CameraSettings {
                    motion_fraction: fraction,
                    ..default()
                };
                settings.multiview = graph
                    .map(|graph| super::super::multiview::MultiViewSettings { graph, ..default() });
                if seed % 2 == 1 {
                    settings.handheld = Some(default());
                }
                let mut scene = source.clone();
                scene.resample_cameras(4, settings.clone(), 1.6).unwrap();
                validate_layout(&scene).unwrap();
                assert_eq!(scene.objects, source.objects);
                assert_eq!(scene.humans, source.humans);
                for (i, camera) in scene.cameras.iter().enumerate() {
                    assert!(settings.accepts_motion(seed, i, camera));
                    if settings.camera_moves(seed, i) {
                        assert!(camera.path_length() >= settings.path_length_min - 1e-4);
                    } else {
                        assert_eq!(camera.transform_at(0.), camera.transform_at(0.73));
                        assert_eq!(camera.transform_at(0.), camera.transform_at(1.));
                    }
                }
                if fraction == 0. {
                    assert_eq!(scene.camera_diagnostics.long_reference_requests, 0);
                    assert_eq!(scene.camera_diagnostics.endpoint_adjustments, 0);
                }
                if fraction == 0.5 {
                    saw_fixed_anchor |= !settings.camera_moves(seed, 0)
                        && (1..4).any(|i| settings.camera_moves(seed, i));
                    saw_moving_anchor |= settings.camera_moves(seed, 0)
                        && (1..4).any(|i| !settings.camera_moves(seed, i));
                }
                let cameras = scene.cameras.clone();
                scene.resample_cameras(4, settings, 1.6).unwrap();
                assert_eq!(cameras, scene.cameras);
            }
        }
    }
    assert!(saw_fixed_anchor && saw_moving_anchor);
}

#[test]
fn fixed_and_mixed_twelve_view_rigs_keep_indexed_roles() {
    let source =
        IndoorManifest::generate_with_humans(61458299, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
    for fraction in [0., 0.5] {
        let mut scene = source.clone();
        let settings = CameraSettings {
            motion_fraction: fraction,
            ..default()
        };
        scene.resample_cameras(12, settings.clone(), 1.).unwrap();
        validate_layout(&scene).unwrap();
        for (index, camera) in scene.cameras.iter().enumerate() {
            assert!(settings.accepts_motion(scene.seed, index, camera));
        }
    }
}

#[test]
fn reordered_overlap_strata_preserve_per_view_motion_and_fixed_lenses() {
    use crate::scene::procedural_indoor::cameras::mixture::OverlapMixture;
    for seed in 0..4 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut settings = CameraSettings {
            motion_fraction: 0.5,
            overlap_mixture: Some(OverlapMixture {
                weights: [0.45, 0.35, 0.2],
                ..default()
            }),
            path_length_max: 0.6,
            long_path_fraction: 0.,
            ..default()
        };
        settings.intrinsics.spread = 0.;
        let fov = settings.intrinsics.quantile(0.5);
        scene.resample_cameras(4, settings.clone(), 1.6).unwrap();
        validate_layout(&scene).unwrap();
        for (index, camera) in scene.cameras.iter().enumerate() {
            assert!(settings.accepts_motion(seed, index, camera));
            assert_eq!(camera.fov_degrees, fov);
        }
        assert!(scene
            .camera_overlap()
            .iter()
            .all(|pair| scene.accepts_camera_pair(pair)));
    }
}
