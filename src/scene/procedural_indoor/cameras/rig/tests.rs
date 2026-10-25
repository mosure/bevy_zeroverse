use super::*;
use crate::scene::procedural_indoor::{
    cameras::{ceiling::CeilingCameraMount, CameraSettings},
    layout::IndoorLayout,
    validation::validate_layout,
};

fn camera(position: Vec3, mounted: bool) -> IndoorCamera {
    IndoorCamera {
        ceiling_mount: mounted.then(|| CeilingCameraMount {
            surface: position + Vec3::Y * 0.2,
            inward_normal: Vec3::NEG_Y,
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            standoff_m: 0.2,
            body_radius_m: 0.055,
        }),
        fisheye: None,
        principal_point_offset: None,
        start: position,
        end: position,
        target: position + Vec3::NEG_Z,
        fov_degrees: 90.,
        motion: None,
    }
}

#[test]
fn partial_group_spread_can_improve_without_weakening_pair_constraints() {
    let policy = MultiViewSettings::from_baseline(0.5).unwrap();
    let cameras = [0., 2., 4.].map(|x| camera(Vec3::new(x, 1.5, 0.), false));
    let tracks = cameras.each_ref().map(Track::new);
    // Collinear prefixes are valid search states, but not completed rigs.
    assert!(accepts_geometry(
        &policy, None, &cameras, &tracks, None, false
    ));
    assert!(!accepts_geometry(
        &policy, None, &cameras, &tracks, None, true
    ));
    let lateral = camera(Vec3::new(2., 1.5, 3.), false);
    let track = Track::new(&lateral);
    assert!(accepts_geometry(
        &policy,
        None,
        &cameras,
        &tracks,
        Some((&lateral, &track)),
        true
    ));
    let close = camera(cameras[1].start + Vec3::X * 0.01, false);
    let track = Track::new(&close);
    assert!(!accepts_geometry(
        &policy,
        None,
        &cameras,
        &tracks,
        Some((&close, &track)),
        false
    ));
}

#[test]
fn reference_content_eligibility_is_restricted_to_the_requested_room() {
    let mut scene =
        IndoorManifest::generate_with_humans(5, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
    assert!(scene.humans.iter().any(|h| !h.neighbor));
    assert!(!super::super::sampling::requires_person(&scene, 0));
    scene.camera_settings.primary_room = false;
    assert!(super::super::sampling::requires_person(&scene, 0));
    assert!(!super::super::sampling::requires_person(&scene, 1));
    scene.camera_settings.primary_room = true;
    let (lo, hi) = scene.primary_room_bounds();
    let p = lo.lerp(hi, 0.5);
    scene.humans[0].position = Vec3::new(p.x, scene.floor_height(p), p.y);
    assert!(super::super::sampling::requires_person(&scene, 0));
}

#[test]
fn legacy_custom_and_stratified_contracts_remain_exact() {
    for baseline in [0., 0.25, 0.5, 0.75, 1.] {
        let policy = MultiViewSettings::from_baseline(baseline).unwrap();
        assert!(MixedRigContract::new(&policy).is_some());
        let mut wire = serde_json::to_value(&policy).unwrap();
        wire.as_object_mut().unwrap().remove("mixed_rig");
        let old: MultiViewSettings = serde_json::from_value(wire).unwrap();
        assert!(old.baseline().is_some());
        assert!(MixedRigContract::new(&old).is_none());
        let roundtrip: MultiViewSettings =
            serde_json::from_slice(&serde_json::to_vec(&policy).unwrap()).unwrap();
        assert_eq!(roundtrip, policy);
        let mut custom = policy;
        custom.min_overlap += 0.01;
        assert!(MixedRigContract::new(&custom).is_none());
    }
    let mut scene = IndoorManifest::generate(0, IndoorLayout::Mixed, 0.65, 0).unwrap();
    scene.cameras = vec![camera(Vec3::ZERO, false), camera(Vec3::Y, true)];
    assert!(scene.mixed_rig_contract().is_some());
    scene.camera_settings.overlap_mixture = Some(Default::default());
    assert!(scene.mixed_rig_contract().is_none());
}

#[test]
fn mount_height_never_weakens_physical_clearance_or_within_role_radius() {
    let policy = MultiViewSettings::from_baseline(0.).unwrap();
    let contract = MixedRigContract::new(&policy);
    let check = |cameras: &[IndoorCamera]| {
        let tracks: Vec<_> = cameras.iter().map(Track::new).collect();
        accepts_geometry(&policy, contract, cameras, &tracks, None, true)
    };
    let room = camera(Vec3::new(0., 1.5, 0.), false);
    let ceiling = camera(Vec3::new(0.3, 3., 0.), true);
    assert!(check(&[room.clone(), ceiling.clone()]));
    assert!(!check(&[
        room.clone(),
        camera(Vec3::new(0.81, 3., 0.), true)
    ]));
    assert!(!check(&[
        room.clone(),
        camera(room.start + Vec3::X * 0.02, true)
    ]));
    assert!(!check(&[
        room.clone(),
        camera(room.start + Vec3::X * 0.36, false)
    ]));
    // Full 3D export remains physical, including the mounting-height difference.
    let tracks = [Track::new(&room), Track::new(&ceiling)];
    assert!(geometry(&tracks, None).unwrap().max_reference_baseline_m > 1.5);
    assert!(!accepts_geometry(
        &policy,
        None,
        &[room, ceiling],
        &tracks,
        None,
        true
    ));
}

#[test]
fn containment_requires_both_directions_at_every_time_and_keeps_raw_values() {
    let policy = MultiViewSettings::from_baseline(0.).unwrap();
    let contract = MixedRigContract::new(&policy).unwrap();
    let a = camera(Vec3::ZERO, false);
    let b = camera(Vec3::Y, true);
    let mut samples: Vec<_> = super::super::multiview::CHECK_TIMES
        .into_iter()
        .map(|time| OverlapSample {
            time,
            reference_to_view: 0.08,
            view_to_reference: 0.50,
            baseline_m: 1.,
            mean_triangulation_degrees: 25.,
        })
        .collect();
    assert!(contract.accepts_visibility(&policy, &a, &b, &samples, 0.05));
    assert!(!policy.accepts(&samples));
    assert_eq!(samples[0].reference_to_view, 0.08);
    samples[2].reference_to_view = 0.;
    assert!(!contract.accepts_visibility(&policy, &a, &b, &samples, 0.05));
    samples[2].reference_to_view = 0.08;
    samples[4].view_to_reference = 0.3;
    assert!(!contract.accepts_visibility(&policy, &a, &b, &samples, 0.05));
    assert!(!contract.accepts_visibility(&policy, &a, &b, &[], 0.05));
}

#[test]
fn calibrated_nested_sensors_have_feasible_nonzero_floors() {
    let policy = MultiViewSettings::from_baseline(0.).unwrap();
    let contract = MixedRigContract::new(&policy).unwrap();
    let mut a = camera(Vec3::ZERO, false);
    let mut b = a.clone();
    a.fov_degrees = 140.;
    b.fov_degrees = 45.;
    let expected = 0.5 * (22.5_f32.to_radians().tan() / 70_f32.to_radians().tan()).powi(2);
    for aspect in [0.75, 1., 1.6] {
        assert!((contract.pair_overlap_floor(&a, &b, aspect) - expected).abs() < 1e-6);
        assert_eq!(
            contract.pair_overlap_floor(&a, &b, aspect),
            contract.pair_overlap_floor(&b, &a, aspect)
        );
        assert_eq!(contract.pair_overlap_floor(&a, &a, aspect), 0.05);
    }
    for fit in [
        crate::calibration::FisheyeFit::Circular,
        crate::calibration::FisheyeFit::Diagonal,
        crate::calibration::FisheyeFit::Elliptical,
    ] {
        a.fisheye = Some(crate::calibration::FisheyeSensor {
            fit,
            anisotropy: 1.,
            lens: crate::calibration::Fisheye {
                coefficients: [0.; 4],
                max_angle: 1.75,
            },
        });
        assert!((0.005..=0.05).contains(&contract.pair_overlap_floor(&a, &b, 1.6)));
    }
}

// A bounded deterministic cohort, independent of GPU, models and asset loading.
#[test]
fn mixed_baseline_regression_cohort_preserves_roles_paths_and_scene() {
    for seed in 0..8 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        let original = scene.clone();
        for baseline in [0., 0.5, 1.] {
            let mut settings = CameraSettings::parse(r#"{"lenses":{"probability":0.5},"ceiling_mount":{"probability":0.2,"motion_fraction":1.0,"travel_m":[0.1,1.6]}}"#).unwrap();
            settings.multiview = Some(MultiViewSettings::from_baseline(baseline).unwrap());
            for count in [2, 4, 8, 12] {
                eprintln!(
                    "mixed baseline regression seed={seed} baseline={baseline} count={count}"
                );
                scene
                    .resample_cameras(count, settings.clone(), 1.)
                    .unwrap_or_else(|e| panic!("baseline={baseline}, count={count}: {e}"));
                validate_layout(&scene).unwrap();
                assert_eq!(scene.objects, original.objects);
                assert_eq!(scene.humans, original.humans);
                assert_eq!(scene.envelope, original.envelope);
                assert_eq!(scene.camera_settings, settings);
                for (i, c) in scene.cameras.iter().enumerate() {
                    assert_eq!(c.is_ceiling_mounted(), settings.ceiling_mounted(seed, i));
                    assert_eq!(c.fisheye, settings.lens_for(seed, i));
                    assert!(settings.accepts_motion(seed, i, c));
                    assert!(scene.camera_curve_clear(c));
                    assert!(scene.cameras[..i]
                        .iter()
                        .all(|a| !super::super::ceiling::bodies_overlap(a, c)));
                }
            }
            let cameras = scene.cameras.clone();
            scene.resample_cameras(12, settings, 1.).unwrap();
            assert_eq!(scene.cameras, cameras);
            let archive: IndoorManifest =
                serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
            assert_eq!(archive, scene);
        }
    }
}
