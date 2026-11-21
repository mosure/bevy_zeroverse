use super::*;
use crate::{
    annotation::schematic::Schematic,
    calibration::CameraCalibration,
    scene::procedural_indoor::{
        cameras::{ceiling::CeilingCameraSettings, handheld::HandheldSettings, CameraMotion},
        layout::{stream, IndoorLayout, IndoorManifest},
    },
};

fn room(seed: u64) -> IndoorManifest {
    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap()
}

#[test]
fn admitted_fixed_and_moving_views_have_balanced_horizon_roll() {
    // Same room contents and default reference-overlap policy for both cohorts.
    for (role, fraction) in [("fixed", 0.), ("moving", 1.)] {
        let mut start_degrees = Vec::new();
        let mut middle_degrees = Vec::new();
        for seed in 8000..8064 {
            let mut scene = room(seed);
            let policy = CameraSettings {
                motion_fraction: fraction,
                ..default()
            };
            scene.resample_cameras(4, policy.clone(), 1.).unwrap();
            for edge in scene.camera_overlap() {
                assert!(
                    policy.multiview.as_ref().unwrap().accepts(&edge.samples),
                    "seed {seed} {role}"
                );
            }
            for c in &scene.cameras {
                assert!(scene.camera_curve_clear(c));
                let mut path = c.runtime_trajectory();
                let start = path.sample(0.);
                let middle = path.sample(0.5);
                let end = path.sample(1.);
                for pose in [start, end] {
                    let roll = horizon_roll_radians(pose.rotation).unwrap().to_degrees();
                    assert!(roll.abs() <= policy.roll.max_degrees + 1e-3);
                }
                if fraction == 0. {
                    assert_eq!(start, middle);
                    assert_eq!(start, end);
                }
                start_degrees.push(horizon_roll_radians(start.rotation).unwrap().to_degrees());
                middle_degrees.push(horizon_roll_radians(middle.rotation).unwrap().to_degrees());
            }
        }
        for (when, sample) in [("start", start_degrees), ("midpoint", middle_degrees)] {
            let count = sample.len();
            let negative = sample.iter().filter(|r| **r < -0.5).count();
            let positive = sample.iter().filter(|r| **r > 0.5).count();
            let mean = sample.iter().sum::<f32>() / count as f32;
            let low = sample.iter().copied().fold(f32::INFINITY, f32::min);
            let high = sample.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let over30 = sample.iter().filter(|r| r.abs() > 30.).count();
            eprintln!("{role} {when}: mean={mean:.3}° min={low:.3}° max={high:.3}° above30={over30}/{count}");
            assert!(mean.abs() < 2.0);
            let mut absolute: Vec<_> = sample.into_iter().map(f32::abs).collect();
            absolute.sort_by(f32::total_cmp);
            let median = (absolute[count / 2 - 1] + absolute[count / 2]) * 0.5;
            let p90 = absolute[(count as f32 * 0.9).ceil() as usize - 1];
            eprintln!("{role} {when}: n={count}, |roll| median={median:.3}°, p90={p90:.3}°, left={negative}, right={positive}, within_half_degree={}", count - negative - positive);
            assert!(negative > count / 3 && positive > count / 3);
            let far_left_right = absolute.iter().filter(|r| **r >= 25.).count();
            eprintln!(
                "absolute roll above 25°: {far_left_right}/{count}, maximum {:.2}°",
                absolute[count - 1]
            );
            assert!((1.4..5.0).contains(&median) && p90 > 6.);
            assert!(far_left_right >= 2 && absolute[count - 1] > 35.);
        }
    }
}

#[test]
fn old_archives_and_zero_policy_keep_their_orientations() {
    let mut scene = room(8011);
    let settings =
        CameraSettings::parse(r#"{"motion_fraction":0,"roll":{"max_degrees":0}}"#).unwrap();
    scene.resample_cameras(2, settings, 1.).unwrap();
    let mut archived = serde_json::to_value(&scene).unwrap();
    archived["camera_settings"]
        .as_object_mut()
        .unwrap()
        .remove("roll");
    for c in archived["cameras"].as_array_mut().unwrap() {
        c.as_object_mut().unwrap().remove("roll_radians");
    }
    let previous: IndoorManifest = serde_json::from_value(archived).unwrap();
    assert_eq!(previous.camera_settings.roll.max_degrees, 0.);
    for c in &previous.cameras {
        assert_eq!(c.roll_radians, 0.);
        assert!(
            horizon_roll_radians(c.transform_at(0.).rotation)
                .unwrap()
                .abs()
                < 1e-6
        );
    }
    // Archived moving cameras already recorded absolute bank at their knots.
    let mut c = previous.cameras[0].clone();
    c.motion = Some(CameraMotion {
        control: [c.start; 2],
        target_end: c.target,
        orientations: None,
        roll: [-0.055, 0.045],
        route: Vec::new(),
    });
    for (t, expected) in [(0., -0.055), (1., 0.045)] {
        assert!(
            (horizon_roll_radians(c.transform_at(t).rotation).unwrap() - expected).abs() < 1e-6
        );
    }
    assert_eq!(CameraSettings::parse("{}").unwrap().roll.max_degrees, 45.);
    for deg in [-1., 45.01, f32::NAN] {
        assert!(CameraSettings {
            roll: RollDistribution {
                max_degrees: deg,
                ..default()
            },
            ..default()
        }
        .validate()
        .is_err());
    }
    assert!(horizon_roll_radians(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)).is_none());
}

#[test]
fn room_roll_preserves_calibrated_rays_and_handheld_local_axes() {
    let mut scene = room(8009);
    let mut camera = serde_json::from_value::<IndoorCamera>(serde_json::json!({
        "start":[0.,1.,0.], "end":[0.,1.,0.], "target":[0.,1.,-3.], "fov_degrees":70.
    }))
    .unwrap();
    camera.roll_radians = 0.09;
    let handheld = HandheldSettings {
        translation_m: [[0.2; 2], [0.; 2], [0.; 2]],
        rotation_degrees: [[0.; 2], [0.; 2], [2.; 2]],
        reverse_probability: 0.,
    };
    handheld.apply(&mut camera, &mut stream(4, 9));
    let (start, end) = (camera.transform_at(0.), camera.transform_at(1.));
    assert!(end
        .translation
        .abs_diff_eq(start.translation + start.rotation * (0.2 * Vec3::X), 1e-6));
    assert!((horizon_roll_radians(end.rotation).unwrap() - 0.09 - 2_f32.to_radians()).abs() < 1e-6);

    let lens = crate::camera::lens::LensSampling {
        probability: 1.,
        ..default()
    }
    .sample(11, 0)
    .unwrap();
    scene.world_yaw = 0.47;
    scene.camera_aspect_ratio = 1.;
    for fisheye in [None, Some(lens)] {
        camera.fisheye = fisheye;
        scene.cameras = vec![camera.clone()];
        let t = 0.5;
        let schematic = Schematic::from_manifest(&scene, t).unwrap();
        let label = &schematic.cameras[0];
        let saved = Mat4::from_cols_array_2d(&label.world_from_view);
        let pose = camera.runtime_trajectory().sample(t);
        let optics = label.calibration.clone().unwrap_or_else(|| {
            CameraCalibration::centered_pinhole(1024, 1024, label.fov_y, 1.).unwrap()
        });
        let pixel = [610., 455.];
        let ray = Vec3::from_array(optics.unproject(pixel, 3.).unwrap());
        let world = Quat::from_rotation_y(scene.world_yaw) * pose.transform_point(ray);
        let projected = optics
            .project(saved.inverse().transform_point3(world).to_array())
            .unwrap();
        assert!(Vec2::from_array(projected).distance(Vec2::from_array(pixel)) < 1e-3);
    }
}

#[test]
fn explicit_handheld_bank_survives_clearance_and_rebinding() {
    let mut scene = room(8030);
    let handheld = HandheldSettings {
        translation_m: [[0.02; 2], [0.01; 2], [0.03; 2]],
        rotation_degrees: [[5.; 2], [-3.; 2], [20.; 2]],
        reverse_probability: 1.,
    };
    let policy = CameraSettings {
        motion_fraction: 0.,
        handheld: Some(handheld.clone()),
        ..default()
    };
    scene.resample_cameras(1, policy.clone(), 1.).unwrap();
    let mut camera = scene.cameras[0].clone();
    handheld.apply(&mut camera, &mut stream(3, 4));
    assert!(scene.camera_curve_clear(&camera));
    let relative_motion = |c: &IndoorCamera| {
        let a = c.transform_at(0.);
        let b = c.transform_at(1.);
        (
            a.rotation.inverse() * (b.translation - a.translation),
            a.rotation.inverse() * b.rotation,
        )
    };
    let (displacement, rotation) = relative_motion(&camera);
    let origin = camera.transform_at(0.);
    let old_roll = camera.roll_radians;
    policy.bind_room_roll(scene.seed, 1, &mut camera);
    assert!((camera.roll_radians - old_roll).abs() > 1e-4);
    let rebound_origin = camera.transform_at(0.);
    assert_eq!(rebound_origin.translation, origin.translation);
    let q = origin.rotation * Quat::from_rotation_z(camera.roll_radians - old_roll);
    assert!(rebound_origin.rotation.abs_diff_eq(q, 1e-6));
    let (new_displacement, new_rotation) = relative_motion(&camera);
    assert!(new_displacement.abs_diff_eq(displacement, 1e-6));
    assert!(new_rotation.abs_diff_eq(rotation, 1e-6));
    assert!(scene.camera_curve_clear(&camera));
    let rebound = camera.clone();
    policy.bind_room_roll(scene.seed, 1, &mut camera);
    assert_eq!(camera, rebound);
}

#[test]
fn room_bank_does_not_reorient_ceiling_mounts() {
    let mut scene = room(8000);
    let mut policy = CameraSettings {
        ceiling_mount: Some(CeilingCameraSettings::default()),
        ..default()
    };
    scene.resample_cameras(1, policy.clone(), 1.).unwrap();
    let original = scene.cameras.clone();
    policy.roll.max_degrees = 0.;
    scene.resample_cameras(1, policy, 1.).unwrap();
    assert!(scene.cameras[0].is_ceiling_mounted());
    assert_eq!(scene.cameras, original);
}
