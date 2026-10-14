use super::super::{
    layout::{IndoorLayout, IndoorManifest},
    preparation::SceneGeometry,
};
use super::*;

fn policy() -> IndoorMotionConfig {
    IndoorMotionConfig {
        chair_fraction: 1.,
        door_fraction: 1.,
        cabinet_fraction: 1.,
        covering_fraction: 1.,
        max_mechanisms: 32,
        ..default()
    }
}

#[test]
fn policy_is_explicit_bounded_and_roundtrips_through_config() {
    let c = crate::app::BevyZeroverseConfig::default();
    assert!(c.indoor_motion.is_none());
    let mut json = serde_json::to_value(&c).unwrap();
    json["indoor_motion"] = serde_json::json!({"door_fraction":1.0});
    let decoded: crate::app::BevyZeroverseConfig = serde_json::from_value(json).unwrap();
    assert_eq!(
        IndoorMotionConfig::parse(decoded.indoor_motion.as_ref().unwrap())
            .unwrap()
            .door_fraction,
        1.
    );
    for bad in [
        r#"{"door_fraction":1.1}"#,
        r#"{"max_mechanisms":0}"#,
        r#"{"chair_travel_m":3}"#,
        r#"{"unrecognized":true}"#,
    ] {
        assert!(IndoorMotionConfig::parse(bad).is_err());
    }
    let mut c = decoded;
    c.ovoxel_mode = crate::app::OvoxelMode::CpuAsync;
    assert!(c.validate_ovoxel().unwrap_err().contains("static"));
}

#[test]
fn absolute_sampling_is_reversible_and_preserves_rigid_hinges() {
    let j = construction::rigid(
        MotionKind::Door,
        8,
        Vec3::new(2., 1., -3.),
        Vec3::Y,
        1.3,
        Vec3::ZERO,
    );
    for time in [1., 0., 0.7, 0.1, 0.7] {
        let tf = j.transform(j.amount(time));
        assert!(tf.transform_point(j.pivot).distance(j.pivot) < 1e-5);
        assert!(
            (tf.transform_point(Vec3::X)
                .distance(tf.transform_point(Vec3::Z))
                - 2_f32.sqrt())
            .abs()
                < 1e-5
        );
    }
    assert_eq!(j.amount(0.), 0.);
    assert_eq!(j.amount(1.), 1.);
}

#[test]
fn disabled_builds_preserve_parts_and_have_no_motion_work() {
    let scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
    let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
    assert!(geometry.architecture.joints.is_empty());
    assert!(geometry.objects.iter().all(|a| a.joints.is_empty()));
    let mut manifest = scene.clone();
    prepare(&mut manifest, &mut geometry);
    assert_eq!(scene, manifest);
    for (o, a) in scene.objects.iter().zip(geometry.objects) {
        let reference = super::super::objects::build_object(o);
        assert_eq!(
            a.parts.keys().collect::<Vec<_>>(),
            reference.parts.keys().collect::<Vec<_>>()
        );
        for (k, g) in a.parts {
            assert_eq!(g.positions, reference.parts[&k].positions);
            assert_eq!(g.indices, reference.parts[&k].indices);
        }
    }
}

#[test]
fn rooms_admit_motion_and_export_shared_navigation_reservations() {
    let mut counts = [0; 4];
    let mut reasons = std::collections::BTreeMap::new();
    for seed in 0..12 {
        let layout = if seed % 2 == 0 {
            IndoorLayout::Kitchen
        } else {
            IndoorLayout::Conference
        };
        let mut scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap();
        scene.indoor_motion = Some(IndoorMotionReport::new(policy()));
        let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
        prepare(&mut scene, &mut geometry);
        let report = scene.indoor_motion.as_ref().unwrap();
        for m in &report.mechanisms {
            if let Some(reason) = &m.rejection {
                *reasons.entry(reason.clone()).or_insert(0) += 1;
            } else {
                counts[m.kind as usize] += 1;
                assert!(!m.swept_bounds.is_empty());
            }
        }
        let boxes = crate::human_motion::planning::obstacles(&scene, usize::MAX, None);
        for (a, b) in report.reservations() {
            assert!(boxes
                .iter()
                .any(|(lo, hi)| lo.cmple(a).all() && hi.cmpge(b).all()));
        }
        let previous = report.clone();
        prepare(&mut scene, &mut geometry);
        assert_eq!(&previous, scene.indoor_motion.as_ref().unwrap());
    }
    eprintln!("accepted {counts:?}; rejected {reasons:?}");
    assert!(
        counts.iter().all(|&c| c > 0),
        "each requested mechanism category must actually move: {counts:?}"
    );
}

#[test]
fn occupied_chairs_remain_static_and_populated_camera_scenes_keep_motion() {
    let mut accepted = 0;
    let mut total = 0;
    let start = std::time::Instant::now();
    for seed in 0..32 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        scene.indoor_motion = Some(IndoorMotionReport::new(policy()));
        let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
        prepare(&mut scene, &mut geometry);
        for m in &scene.indoor_motion.as_ref().unwrap().mechanisms {
            total += 1;
            if m.rejection.is_none() {
                accepted += 1;
                if m.kind == MotionKind::Chair {
                    assert!(!scene.humans.iter().any(|p| p.chair == m.object_id));
                }
            }
        }
    }
    eprintln!(
        "populated 32-room admission: {accepted}/{total}, {:.3}s geometry+clearance",
        start.elapsed().as_secs_f64()
    );
    assert!(
        accepted >= 32,
        "motion must remain usable with real cameras and people"
    );
}
