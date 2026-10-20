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
        fan_fraction: 1.,
        appliance_fraction: 1.,
        prop_fraction: 1.,
        cart_fraction: 1.,
        max_mechanisms: 128,
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
        r#"{"fan_fraction":-0.1}"#,
        r#"{"max_mechanisms":129}"#,
        r#"{"prop_fraction":1.1}"#,
        r#"{"appliance_fraction":-0.1}"#,
        r#"{"cart_fraction":-0.1}"#,
        r#"{"prop_travel_m":0.76}"#,
        r#"{"prop_rotation_degrees":181}"#,
        r#"{"cart_travel_m":2.1}"#,
        r#"{"fan_turns":[8,2]}"#,
        r#"{"fan_turns":[0,121]}"#,
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
fn fractions_select_nested_seeded_sets_and_zero_policy_has_no_mechanisms() {
    let kinds = [
        MotionKind::Chair,
        MotionKind::Door,
        MotionKind::Cabinet,
        MotionKind::Covering,
        MotionKind::Fan,
        MotionKind::Appliance,
        MotionKind::Prop,
        MotionKind::Cart,
    ];
    for kind in kinds {
        let mut previous = std::collections::BTreeSet::new();
        for fraction in [0., 0.25, 0.5, 0.75, 1.] {
            let a = Assembly {
                motion_policy: Some(IndoorMotionConfig {
                    chair_fraction: fraction,
                    door_fraction: fraction,
                    cabinet_fraction: fraction,
                    covering_fraction: fraction,
                    fan_fraction: fraction,
                    appliance_fraction: fraction,
                    prop_fraction: fraction,
                    cart_fraction: fraction,
                    ..default()
                }),
                ..default()
            };
            let selected: std::collections::BTreeSet<_> = (0..1024)
                .filter(|&seed| a.motion_selected(kind, seed))
                .collect();
            assert!(previous.is_subset(&selected));
            assert!((selected.len() as f32 / 1024. - fraction).abs() < 0.06);
            previous = selected;
        }
        assert_eq!(previous.len(), 1024);
    }
    let policy = IndoorMotionConfig {
        chair_fraction: 0.,
        door_fraction: 0.,
        cabinet_fraction: 0.,
        covering_fraction: 0.,
        fan_fraction: 0.,
        appliance_fraction: 0.,
        prop_fraction: 0.,
        cart_fraction: 0.,
        ..default()
    };
    let mut scene =
        IndoorManifest::generate_with_humans(4, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
    scene.indoor_motion = Some(IndoorMotionReport::new(policy));
    let (mut g, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
    assert!(g.architecture.joints.is_empty() && g.objects.iter().all(|a| a.joints.is_empty()));
    prepare(&mut scene, &mut g);
    assert!(scene.indoor_motion.unwrap().mechanisms.is_empty());
}

#[test]
fn motion_configuration_matrix_reports_real_admission_and_bounded_cost() {
    use std::collections::BTreeMap;
    let mut summary = vec![];
    for (name, policy) in [
        ("default", IndoorMotionConfig::default()),
        (
            "maximum",
            IndoorMotionConfig {
                chair_travel_m: 1.5,
                chair_swivel_degrees: 120.,
                prop_travel_m: 0.75,
                prop_rotation_degrees: 180.,
                cart_travel_m: 2.,
                fan_turns: [120., 120.],
                ..policy()
            },
        ),
    ] {
        let mut row = serde_json::json!({"policy":name});
        let mut accepted = [0usize; 8];
        let mut selected = [0usize; 8];
        let mut rejected = BTreeMap::<String, usize>::new();
        let mut adapted = BTreeMap::<String, usize>::new();
        let mut seconds = vec![];
        let mut moving_rooms = 0;
        let mut rooms = vec![];
        for seed in 0..12 {
            let density = [0.25, 0.65, 1.][seed as usize % 3];
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, density, 3, 0.25)
                    .unwrap();
            scene.indoor_motion = Some(IndoorMotionReport::new(policy.clone()));
            super::super::validation::validate_layout(&scene).unwrap();
            let objects = scene.objects.clone();
            let (mut geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
            let start = std::time::Instant::now();
            prepare(&mut scene, &mut geometry);
            seconds.push(start.elapsed().as_secs_f64());
            assert_eq!(scene.objects, objects);
            let report = scene.indoor_motion.as_ref().unwrap();
            let count = report
                .mechanisms
                .iter()
                .filter(|m| m.rejection.is_none())
                .count();
            moving_rooms += usize::from(count > 0);
            assert!(count <= policy.max_mechanisms);
            rooms.push(serde_json::json!({"seed":seed,"selected":report.mechanisms.len(),
                "accepted":count,"noticeable":report.mechanisms.iter().filter(|m| m.rejection.is_none()
                    && (m.translation_m.length() >= 0.02 || m.angle_radians.abs() >= 5_f32.to_radians())).count()}));
            for m in &report.mechanisms {
                selected[m.kind as usize] += 1;
                assert!(m.attempts <= 24);
                if let Some(reason) = &m.rejection {
                    *rejected.entry(reason.clone()).or_default() += 1;
                    continue;
                }
                accepted[m.kind as usize] += 1;
                if let Some(adaptation) = &m.adaptation {
                    *adapted.entry(adaptation.clone()).or_default() += 1;
                }
                assert!(m.translation_m.length() <= m.requested_translation_m.length() + 1e-5);
                assert!(m.angle_radians.abs() <= m.requested_angle_radians.abs() + 1e-5);
                if m.kind != MotionKind::Covering {
                    assert!(m.translation_m.length() > 0. || m.angle_radians.abs() > 0.);
                    let a = m.rigid_transform_at(0.37).unwrap();
                    let b = m.rigid_transform_at(0.37).unwrap();
                    assert_eq!(a, b);
                }
            }
            let serialized = serde_json::to_string(report).unwrap();
            assert_eq!(
                *report,
                serde_json::from_str::<IndoorMotionReport>(&serialized).unwrap()
            );
        }
        row["selected_by_kind"] = serde_json::json!(selected);
        row["accepted_by_kind"] = serde_json::json!(accepted);
        row["rejections"] = serde_json::json!(rejected);
        row["adaptations"] = serde_json::json!(adapted);
        row["motion_planning_seconds"] = serde_json::json!(seconds);
        row["moving_rooms"] = serde_json::json!(moving_rooms);
        row["rooms"] = serde_json::json!(rooms);
        assert!(accepted[MotionKind::Chair as usize] > 0, "{row}");
        assert!(moving_rooms >= 8, "{row}");
        summary.push(row);
    }
    std::fs::create_dir_all("out/object_motion_review").unwrap();
    std::fs::write(
        "out/object_motion_review/configuration-matrix.json",
        serde_json::to_vec_pretty(&serde_json::json!({"rooms_per_policy":12,"cameras":3,
            "densities":[0.25,0.65,1.],"human_density":0.25,"cpu_only":true,
            "kinds":["chair","door","cabinet","covering","fan","appliance","prop","cart"],"results":summary}))
        .unwrap(),
    )
    .unwrap();
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
    let mut counts = [0; 8];
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
        counts[..7].iter().all(|&c| c > 0),
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
