use super::*;
use crate::scene::procedural_indoor::{
    cameras::ceiling::proposal::Prototype,
    layout::{IndoorLayout, IndoorManifest},
};

fn angle(rotation: Quat) -> f32 {
    let forward = rotation * Vec3::NEG_Z;
    forward
        .cross(Vec3::NEG_Z)
        .length()
        .atan2(forward.dot(Vec3::NEG_Z))
        .to_degrees()
}

#[test]
fn mounting_error_is_continuous_isotropic_seeded_and_slot_stable() {
    let policy = CeilingCameraSettings::default();
    let mut angles = Vec::new();
    let mut bins = [0usize; 8];
    let mut direction_sum = Vec2::ZERO;
    for seed in 0..2048 {
        for index in 0..8 {
            let rotation = sample(&policy, seed, index);
            assert_eq!(rotation, sample(&policy, seed, index));
            assert!((rotation.length_squared() - 1.).abs() < 1e-6);
            let a = angle(rotation);
            assert!(a > 0. && a < 12.);
            angles.push(a);
            let direction = (rotation * Vec3::NEG_Z).xy().normalize();
            direction_sum += direction;
            let azimuth = direction
                .y
                .atan2(direction.x)
                .rem_euclid(std::f32::consts::TAU);
            bins[(azimuth / std::f32::consts::TAU * 8.) as usize % 8] += 1;
        }
    }
    angles.sort_by(f32::total_cmp);
    assert!(bins.iter().all(|&n| (1800..2300).contains(&n)), "{bins:?}");
    assert!(direction_sum.length() / (angles.len() as f32) < 0.02);
    let quantiles = [angles[819], angles[8192], angles[15564]];
    assert!((1.0..1.6).contains(&quantiles[0]));
    assert!((4.4..5.0).contains(&quantiles[1]));
    assert!((9.2..9.9).contains(&quantiles[2]));
    eprintln!(
        "mount tilt prior: {} samples, p05/p50/p95={quantiles:?}, azimuth bins={bins:?}",
        angles.len()
    );
}

#[test]
fn mounting_error_respects_zero_fixed_and_bounded_ranges() {
    for range in [[0., 0.], [8., 8.], [10., 35.], [45., 45.]] {
        let policy = CeilingCameraSettings {
            tilt_degrees: range,
            ..default()
        };
        policy.validate().unwrap();
        for seed in 0..128 {
            let rotation = sample(&policy, seed, 3);
            let a = angle(rotation);
            assert!(
                a >= range[0] - 1e-4 && a <= range[1] + 1e-4,
                "{range:?}: {a}"
            );
            if range[1] == 0. {
                assert_eq!(rotation, Quat::IDENTITY);
            }
        }
    }
    for range in [
        [-0.1, 1.],
        [1., 0.],
        [0., 45.1],
        [0., f32::NAN],
        [f32::INFINITY, f32::INFINITY],
    ] {
        assert!(CeilingCameraSettings {
            tilt_degrees: range,
            ..default()
        }
        .validate()
        .is_err());
    }
}

#[test]
fn tilt_composes_with_roof_slope_independently_of_image_clocking() {
    let policy = CeilingCameraSettings::default();
    for normal in [Vec3::NEG_Y, Vec3::new(0.3, -1., -0.2).normalize()] {
        for seed in 0..32 {
            let local = sample(&policy, seed, 1);
            let base = Transform::IDENTITY.looking_to(normal, Vec3::Z).rotation;
            let mut original_axis = None;
            for roll in [0., 0.7, 2.1, 4.8] {
                let mount = CeilingCameraMount {
                    surface: Vec3::new(0., 3., 0.),
                    inward_normal: normal,
                    rotation: base * local * Quat::from_rotation_z(roll),
                    standoff_m: 0.2,
                    body_radius_m: 0.055,
                };
                assert!((mount.tilt_degrees() - angle(local)).abs() < 1e-4);
                assert!(mount.optical_axis().y < -0.8);
                if let Some(axis) = original_axis {
                    assert!(mount.optical_axis().distance(axis) < 1e-6);
                } else {
                    original_axis = Some(mount.optical_axis());
                }
            }
        }
    }
}

#[test]
fn zero_tilt_retains_legacy_draws_and_tilt_does_not_change_lenses_or_roles() {
    let mut scene =
        IndoorManifest::generate_with_humans(123, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut policy = CeilingCameraSettings {
        tilt_degrees: [0.; 2],
        ..default()
    };
    scene.camera_settings.ceiling_mount = Some(policy.clone());
    for index in 0..12 {
        let prototype = Prototype::sample(&scene, index, &policy);
        let mut prior = stream(scene.seed, 0x4343_0000 + index as u64);
        let fov = scene
            .camera_settings
            .intrinsics_for(scene.seed, index)
            .sample(&mut prior);
        let standoff = prior.random_range(policy.standoff_m[0]..=policy.standoff_m[1]);
        let roll = prior.random_range(0.0..std::f32::consts::TAU);
        assert_eq!(prototype.fov_degrees, fov);
        assert_eq!(prototype.standoff_m, standoff);
        assert_eq!(prototype.local_rotation, Quat::from_rotation_z(roll));
        policy.tilt_degrees = [0., 12.];
        let tilted = Prototype::sample(&scene, index, &policy);
        assert_eq!(prototype.lens, tilted.lens);
        assert_eq!(prototype.fov_degrees, tilted.fov_degrees);
        assert_eq!(prototype.standoff_m, tilted.standoff_m);
        assert_ne!(prototype.local_rotation, tilted.local_rotation);
        policy.tilt_degrees = [0.; 2];
    }
    // Unmounted slots pay no tilt-sampling cost and retain their old draws.
    scene
        .camera_settings
        .ceiling_mount
        .as_mut()
        .unwrap()
        .probability = 0.;
    let a = Prototype::sample(&scene, 2, &policy);
    policy.tilt_degrees = [10., 30.];
    let b = Prototype::sample(&scene, 2, &policy);
    assert_eq!(a.local_rotation, b.local_rotation);
    assert_eq!(a.lens, b.lens);
}

#[test]
fn guided_mount_intersects_shared_surface_without_changing_tilt() {
    let mut scene =
        IndoorManifest::generate_with_humans(12, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.camera_settings.ceiling_mount = Some(CeilingCameraSettings::default());
    scene.envelope.as_mut().unwrap().soffits.clear();
    for slope in [Vec2::ZERO, Vec2::new(0.7, -0.4)] {
        scene.envelope.as_mut().unwrap().ceiling_drop = slope;
        let prototype = Prototype::sample(
            &scene,
            1,
            scene.camera_settings.ceiling_mount.as_ref().unwrap(),
        );
        for target in [
            Vec3::ZERO,
            Vec3::new(0.5, 0.7, -0.3),
            Vec3::new(-0.4, 1.1, 0.1),
        ] {
            let p = prototype.overlooking(&scene, target).unwrap();
            let camera = prototype.mounted_at(&scene, 1, p, &mut stream(12, 99));
            let mount = camera.ceiling_mount.as_ref().unwrap();
            let delta = target - camera.start;
            assert!(delta.cross(mount.optical_axis()).length() < 1e-4);
            assert!(delta.dot(mount.optical_axis()) > 0.);
            assert!((mount.tilt_degrees() - angle(prototype.local_rotation)).abs() < 1e-4);
        }
        assert!(prototype
            .overlooking(&scene, Vec3::Y * (scene.room_size.y + 5.))
            .is_none());
    }
}

#[test]
fn configured_tilt_ranges_survive_actual_static_and_moving_placement() {
    for seed in 0..8 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        for range in [[0., 0.], [10., 35.]] {
            let policy = CameraSettings {
                ceiling_mount: Some(CeilingCameraSettings {
                    tilt_degrees: range,
                    motion_fraction: (seed % 2) as f32,
                    travel_m: [0.1, 0.5],
                    lenses: LensSampling {
                        probability: ((seed / 2) % 2) as f32,
                        ..CeilingCameraSettings::default().lenses
                    },
                    ..default()
                }),
                ..default()
            };
            scene.resample_cameras(1, policy.clone(), 1.).unwrap();
            let camera = &scene.cameras[0];
            let mount = camera.ceiling_mount.as_ref().unwrap();
            assert!(
                mount.tilt_degrees() >= range[0] - 1e-4 && mount.tilt_degrees() <= range[1] + 1e-4
            );
            assert!(super::super::mount_clear(&scene, camera));
            assert!(policy.accepts_motion(seed, 0, camera));
        }
    }
}
