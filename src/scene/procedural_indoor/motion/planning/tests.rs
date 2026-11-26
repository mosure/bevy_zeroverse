use super::*;
pub(super) fn empty() -> (IndoorManifest, SceneGeometry) {
    let mut scene = IndoorManifest::generate_with_humans(0, default(), 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    scene.humans.clear();
    scene.cameras.clear();
    scene.indoor_motion = Some(IndoorMotionReport::new(default()));
    (
        scene,
        SceneGeometry {
            architecture: default(),
            objects: vec![],
            humans: vec![],
        },
    )
}
fn block(center: Vec3, size: Vec3) -> Assembly {
    let mut a = Assembly::default();
    a.box_part(Surface::Wood, "door", center, size, 0.);
    a
}
#[test]
fn full_arc_clearance_catches_obstacles_between_clear_endpoints() {
    let (mut scene, mut geometry) = empty();
    let a = &mut geometry.architecture;
    a.append_assembly(block(Vec3::new(0.71, 1., -0.71), Vec3::splat(0.10)));
    a.articulated(
        block(Vec3::new(1., 1., 0.), Vec3::splat(0.04)),
        construction::rigid(
            MotionKind::Door,
            1,
            Vec3::new(0., 1., 0.),
            Vec3::Y,
            std::f32::consts::FRAC_PI_2,
            Vec3::ZERO,
        ),
    );
    prepare(&mut scene, &mut geometry);
    let m = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    assert!(m.rejection.is_none());
    assert_eq!(m.adaptation.as_deref(), Some("shortened"));
    assert!(m.angle_radians < std::f32::consts::FRAC_PI_4);
    let tree = Tree::new(&scene, &geometry);
    assert!(sweeps(
        &geometry.architecture,
        &geometry.architecture.joints[0],
        Transform::IDENTITY
    )
    .into_iter()
    .all(|s| !tree.hits(s, None, &[0], None, None)));
}
#[test]
fn crossing_mechanisms_shorten_without_overlapping_reservations() {
    let (mut scene, mut geometry) = empty();
    for (p, delta) in [
        (Vec3::new(-1., 1., 0.), Vec3::X * 2.),
        (Vec3::new(0., 1., -1.), Vec3::Z * 2.),
    ] {
        geometry.architecture.articulated(
            block(p, Vec3::splat(0.08)),
            construction::rigid(MotionKind::Door, 5, Vec3::ZERO, Vec3::Y, 0., delta),
        );
    }
    prepare(&mut scene, &mut geometry);
    let report = scene.indoor_motion.unwrap();
    assert!(report.mechanisms.iter().all(|m| m.rejection.is_none()));
    assert!(report.mechanisms.iter().any(|m| m.adaptation.is_some()));
    assert!(report.mechanisms[0].swept_bounds.iter().all(|&a| {
        report.mechanisms[1]
            .swept_bounds
            .iter()
            .all(|&b| !overlaps(a, b))
    }));
}

#[test]
fn crowded_chair_falls_back_to_swivel_and_preparation_is_idempotent() {
    let (mut scene, mut g) = empty();
    scene.envelope = None;
    for (p, size) in [
        (Vec3::new(0.124, 1., 0.), Vec3::new(0.04, 0.8, 0.4)),
        (Vec3::new(-0.124, 1., 0.), Vec3::new(0.04, 0.8, 0.4)),
        (Vec3::new(0., 1., 0.124), Vec3::new(0.4, 0.8, 0.04)),
        (Vec3::new(0., 1., -0.124), Vec3::new(0.4, 0.8, 0.04)),
    ] {
        g.architecture.append_assembly(block(p, size));
    }
    g.architecture.articulated(
        block(Vec3::Y, Vec3::splat(0.20)),
        construction::rigid(MotionKind::Chair, 3, Vec3::Y, Vec3::Y, 2., Vec3::X * 1.5),
    );
    prepare(&mut scene, &mut g);
    let report = scene.indoor_motion.clone().unwrap();
    let m = &report.mechanisms[0];
    assert!(m.rejection.is_none(), "{m:?}");
    assert_eq!(m.adaptation.as_deref(), Some("swivel_only"));
    assert_eq!(m.translation_m, Vec3::ZERO);
    assert!(m.angle_radians.abs() > 0.01);
    prepare(&mut scene, &mut g);
    assert_eq!(scene.indoor_motion.as_ref().unwrap(), &report);
}

#[test]
fn fully_blocked_motion_stays_static_and_budget_is_enforced() {
    let (mut scene, mut g) = empty();
    g.architecture
        .append_assembly(block(Vec3::new(0.061, 1., 0.), Vec3::new(0.02, 0.4, 0.4)));
    g.architecture.articulated(
        block(Vec3::Y, Vec3::splat(0.1)),
        construction::rigid(MotionKind::Door, 2, Vec3::Y, Vec3::Y, 0., Vec3::X),
    );
    prepare(&mut scene, &mut g);
    assert_eq!(
        scene.indoor_motion.as_ref().unwrap().mechanisms[0]
            .rejection
            .as_deref(),
        Some("static_geometry")
    );
    assert!(!g.architecture.joints[0].accepted);
    let (mut scene, mut g) = empty();
    scene.indoor_motion.as_mut().unwrap().policy.max_mechanisms = 1;
    for x in [-2., 2.] {
        g.architecture.articulated(
            block(Vec3::new(x, 1., 0.), Vec3::splat(0.1)),
            construction::rigid(MotionKind::Door, 2, Vec3::Y, Vec3::Y, 0., Vec3::Z * 0.2),
        );
    }
    prepare(&mut scene, &mut g);
    let m = &scene.indoor_motion.as_ref().unwrap().mechanisms;
    assert_eq!(m.iter().filter(|m| m.rejection.is_none()).count(), 1);
    assert_eq!(
        m.iter()
            .filter(|m| m.rejection.as_deref() == Some("mechanism_limit") && m.attempts == 0)
            .count(),
        1
    );
}

#[test]
fn camera_path_reservations_follow_route_instead_of_filling_its_bounding_box() {
    use crate::scene::procedural_indoor::{cameras::CameraMotion, layout::IndoorCamera};
    let (mut scene, mut g) = empty();
    let a = Vec3::new(-2., 1., 2.);
    let b = Vec3::new(2., 1., -2.);
    scene.cameras.push(IndoorCamera {
        roll_radians: 0.,
        ceiling_mount: None,
        fisheye: None,
        start: a,
        end: b,
        target: Vec3::Y,
        fov_degrees: 60.,
        principal_point_offset: None,
        motion: Some(CameraMotion {
            control: [a, b],
            orientations: None,
            target_end: Vec3::Y,
            roll: [0.; 2],
            route: vec![a, Vec3::new(2., 1., 2.), b],
        }),
    });
    g.architecture.articulated(
        block(Vec3::Y, Vec3::splat(0.1)),
        construction::rigid(MotionKind::Door, 2, Vec3::Y, Vec3::Y, 0., Vec3::X * 0.25),
    );
    prepare(&mut scene, &mut g);
    assert!(scene.indoor_motion.as_ref().unwrap().mechanisms[0]
        .rejection
        .is_none());
    let reservations = camera_reservations(&scene);
    for i in 0..=1000 {
        let p = scene.cameras[0].transform_at(i as f32 / 1000.).translation;
        assert!(reservations
            .iter()
            .any(|(a, b)| p.cmpge(*a).all() && p.cmple(*b).all()));
    }
    let m = scene.cameras[0].motion.as_mut().unwrap();
    m.route.clear();
    m.control = [Vec3::new(-3., 1., -3.), Vec3::new(3., 1., 3.)];
    let reservations = camera_reservations(&scene);
    for i in 0..=1000 {
        let p = scene.cameras[0].transform_at(i as f32 / 1000.).translation;
        assert!(reservations
            .iter()
            .any(|(a, b)| p.cmpge(*a).all() && p.cmple(*b).all()));
    }
}

#[test]
fn fast_fan_reserves_every_blade_angle_and_never_accepts_partial_rotations_when_blocked() {
    let (mut scene, mut g) = empty();
    let o = super::super::super::objects::audit::fixture(
        super::super::super::layout::ObjectKind::CeilingFan,
        4,
    );
    let policy = IndoorMotionConfig {
        fan_fraction: 1.,
        fan_turns: [120., 120.],
        ..default()
    };
    g.architecture = super::super::super::objects::build_object_with_motion(&o, Some(&policy));
    let j = &g.architecture.joints[0];
    let bb = sweep(&g.architecture, j, Transform::IDENTITY);
    for i in 0..=4096 {
        let tf = j.transform(j.amount(i as f32 / 4096.));
        for key in &j.parts {
            for p in &g.architecture.parts[key].positions {
                let p = tf.transform_point(Vec3::from_array(*p));
                assert!(bb
                    .iter()
                    .any(|(a, b)| p.cmpge(*a).all() && p.cmple(*b).all()));
            }
        }
    }
    prepare(&mut scene, &mut g);
    assert!(g.architecture.joints[0].accepted);
    let p = super::super::super::objects::fans::parameters(&o);
    g.architecture.append_assembly(block(
        Vec3::new(p.radius_m * 0.75, p.rotor_y, 0.),
        Vec3::splat(0.03),
    ));
    prepare(&mut scene, &mut g);
    assert!(!g.architecture.joints[0].accepted);
    assert_eq!(scene.indoor_motion.unwrap().mechanisms[0].attempts, 1);
}
#[test]
fn coupled_rotor_leaves_are_admitted_atomically() {
    let (mut scene, mut geometry) = empty();
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .operation_return_fraction = 1.;
    for i in 0..4 {
        let p = Quat::from_rotation_y(i as f32 * std::f32::consts::FRAC_PI_2) * Vec3::X + Vec3::Y;
        let mut joint = construction::rigid(MotionKind::Door, 5, Vec3::Y, Vec3::Y, 2.0, Vec3::ZERO);
        joint.coupled = true;
        joint.attachment = Some("portal0".into());
        geometry
            .architecture
            .articulated(block(p, Vec3::splat(0.05)), joint);
    }
    prepare(&mut scene, &mut geometry);
    let records = scene.indoor_motion.unwrap().mechanisms;
    assert_eq!(records.len(), 1);
    assert!(records[0].operation_cycle.is_none());
    assert!(geometry.architecture.joints.iter().all(|j| j.accepted));
}

#[test]
fn an_obstruction_in_one_telescoping_track_stops_both_leaves() {
    use crate::scene::procedural_indoor::architecture::doors::{
        DoorInstallation, DoorMechanism, DoorPanel, DoorProgram,
    };
    let (mut scene, mut geometry) = empty();
    let mut program = DoorProgram::sample(11, DoorMechanism::Telescoping);
    program.handedness = 1.;
    program.swing_side = 1.;
    program.opening = 0.;
    program.panel = DoorPanel::Flush;
    let door = DoorInstallation {
        id: 7,
        center: Vec3::ZERO,
        yaw: 0.,
        width: 1.4,
        height: 2.25,
        wall_depth: 0.16,
        program,
    };
    geometry.architecture.motion_policy = Some(IndoorMotionConfig {
        door_fraction: 1.,
        ..default()
    });
    door.build(&mut geometry.architecture, true);
    scene.indoor_motion.as_mut().unwrap().policy.max_mechanisms = 1;
    prepare(&mut scene, &mut geometry);
    assert_eq!(geometry.architecture.joints.len(), 2);
    assert!(geometry.architecture.joints.iter().all(|j| j.accepted));
    let records = &scene.indoor_motion.as_ref().unwrap().mechanisms;
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.linked_slides.len(), 1);
    assert_ne!(record.translation_m, record.linked_slides[0].translation_m);
    assert!(!record.swept_bounds.is_empty());
    for (i, joint) in geometry.architecture.joints.iter().enumerate() {
        for t in [0., 0.5, 1.] {
            assert_eq!(
                record.rigid_transform_for_joint_at(i, t).unwrap(),
                joint.transform(joint.amount(t))
            );
        }
    }
    let mut archived = serde_json::to_value(record).unwrap();
    archived.as_object_mut().unwrap().remove("linked_slides");
    let legacy: MechanismReport = serde_json::from_value(archived).unwrap();
    assert!(legacy.linked_slides.is_empty());
    assert_eq!(
        legacy.rigid_transform_at(0.5),
        record.rigid_transform_at(0.5)
    );
    assert!(legacy.rigid_transform_for_joint_at(1, 0.5).is_none());

    let one_way = record.clone();
    assert!(one_way.operation_cycle.is_none());
    assert_eq!(one_way.stroke_fraction_at(1.), 1.);
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .operation_return_fraction = 1.;
    let version = scene.generator_version;
    scene.generator_version = OPERATION_CYCLE_VERSION - 1;
    prepare(&mut scene, &mut geometry);
    assert_eq!(scene.indoor_motion.as_ref().unwrap().mechanisms[0], one_way);
    scene.generator_version = version;
    prepare(&mut scene, &mut geometry);
    let cyclic = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
    let cycle = cyclic.operation_cycle.unwrap();
    assert_eq!(cyclic.swept_bounds, one_way.swept_bounds);
    assert_eq!(cyclic.attempts, one_way.attempts);
    assert_eq!(cyclic.linked_slides, one_way.linked_slides);
    for (id, joint) in geometry.architecture.joints.iter().enumerate() {
        assert_eq!(joint.operation_cycle, Some(cycle));
        for t in [cycle.leaves_stop_at, 1., cycle.reaches_stop_at, 0.] {
            let delta = cyclic.rigid_transform_for_joint_at(id, t).unwrap();
            assert_eq!(delta, joint.transform(joint.amount(t)));
            if t == 0. || t == 1. {
                assert_eq!(delta, Transform::IDENTITY);
            } else {
                assert_eq!(Some(delta), one_way.rigid_transform_for_joint_at(id, 1.));
            }
        }
    }

    let leaf = door.leaves()[0];
    let edge = leaf.transform.translation.x + leaf.size.x * 0.5;
    geometry.architecture.append_assembly(block(
        Vec3::new(edge + 0.005, 0.4, leaf.transform.translation.z),
        Vec3::new(0.006, 0.06, 0.010),
    ));
    prepare(&mut scene, &mut geometry);
    assert!(geometry.architecture.joints.iter().all(|j| !j.accepted));
    let reports = &scene.indoor_motion.as_ref().unwrap().mechanisms;
    assert_eq!(reports.len(), 1);
    assert!(reports[0].rejection.is_some() && reports[0].operation_cycle.is_none());
    assert!((0..2).all(|i| reports[0].rigid_transform_for_joint_at(i, 1.).is_none()));
}

#[test]
fn ceiling_camera_mount_body_blocks_fan_even_when_lens_is_below_rotor() {
    use super::super::super::{cameras::ceiling::CeilingCameraMount, layout::IndoorCamera};
    let (mut scene, mut geometry) = empty();
    let object = super::super::super::objects::audit::fixture(
        super::super::super::layout::ObjectKind::CeilingFan,
        4,
    );
    let policy = IndoorMotionConfig {
        fan_fraction: 1.,
        fan_turns: [20., 20.],
        ..default()
    };
    geometry.architecture =
        super::super::super::objects::build_object_with_motion(&object, Some(&policy));
    let fan = super::super::super::objects::fans::parameters(&object);
    let surface = Vec3::new(fan.radius_m * 0.75, fan.rotor_y + 0.10, 0.);
    let eye = surface - Vec3::Y * 0.8;
    scene.cameras.push(IndoorCamera {
        roll_radians: 0.,
        ceiling_mount: Some(CeilingCameraMount {
            surface,
            inward_normal: Vec3::NEG_Y,
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            standoff_m: 0.8,
            body_radius_m: 0.05,
        }),
        fisheye: None,
        principal_point_offset: None,
        start: eye,
        end: eye,
        target: eye - Vec3::Y * 2.,
        fov_degrees: 60.,
        motion: None,
    });
    prepare(&mut scene, &mut geometry);
    assert!(!geometry.architecture.joints[0].accepted);
    assert_eq!(
        scene.indoor_motion.unwrap().mechanisms[0]
            .rejection
            .as_deref(),
        Some("camera_path")
    );
}
#[test]
fn arc_enclosures_contain_every_intermediate_vertex() {
    let mut a = Assembly::default();
    let j = construction::rigid(
        MotionKind::Door,
        1,
        Vec3::new(0.15, 0.5, -0.2),
        Vec3::Y,
        2.1,
        Vec3::X * 0.4,
    );
    a.articulated(
        block(Vec3::new(0.6, 0.8, 0.1), Vec3::new(0.9, 1.2, 0.06)),
        j,
    );
    let j = &a.joints[0];
    let envelopes = sweep(&a, j, Transform::IDENTITY);
    for i in 0..=8192 {
        let tf = j.transform(i as f32 / 8192.);
        for g in a.parts.values() {
            for p in &g.positions {
                let p = tf.transform_point(Vec3::from_array(*p));
                assert!(envelopes
                    .iter()
                    .any(|(a, b)| p.cmpge(*a).all() && p.cmple(*b).all()));
            }
        }
    }
}
#[test]
fn collapsed_mid_morph_is_rejected_even_when_endpoints_are_valid() {
    let a = Geometry {
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        indices: vec![0, 1, 2],
        ..default()
    };
    let mut b = a.clone();
    b.positions[1] = [-1., 0., 0.];
    b.positions[2] = [0., -1., 0.];
    assert!(!validate_morph(&a, &b));
    assert!(validate_morph(&a, &a));
}

#[test]
fn track_coverings_can_use_their_own_head_but_still_collide_with_other_hardware() {
    use crate::scene::procedural_indoor::architecture::facade::{treatments, Shade};
    for (kind, seed) in [
        (Shade::Roller, 44210),
        (Shade::Curtain, 75110),
        (Shade::Vertical, 470),
        (Shade::Pleated, 426),
    ] {
        let source = IndoorManifest::initial_mixed_fixture(seed, 0.65, 2, 0.25).unwrap();
        let mut facade = source
            .envelope
            .as_ref()
            .unwrap()
            .walls
            .iter()
            .filter_map(|w| w.facade.as_ref())
            .find(|f| f.shade == kind)
            .unwrap()
            .clone();
        facade.shade_coverage = 0.85;
        if kind == Shade::Vertical {
            facade.shade_tilt = 0.16;
            facade.shade_coverage = 0.98;
        }
        facade.treatment.as_mut().unwrap().tieback = 0.;
        let opening = &facade.openings[0];
        for obstructed in [false, true] {
            let (mut scene, mut g) = empty();
            g.architecture.motion_policy = Some(IndoorMotionConfig {
                covering_fraction: 1.,
                ..default()
            });
            treatments::build(&mut g.architecture, Transform::IDENTITY, &facade, opening);
            if obstructed {
                // A paired curtain may have clear glass at the centre. Place
                // the foreign fitting directly across an actual fabric face.
                let joint = &g.architecture.joints[0];
                let key = joint
                    .parts
                    .iter()
                    .find(|k| k.0 == facade.treatment.as_ref().unwrap().fabric)
                    .unwrap();
                let cloth = &g.architecture.parts[key];
                let tri = cloth.indices.as_chunks::<3>().0[0];
                let [a, b, c] = tri.map(|i| Vec3::from(cloth.positions[i as usize]));
                g.architecture.box_part(
                    Surface::Metal,
                    "blinds#contact_other_head",
                    (a + b + c) / 3.,
                    Vec3::splat(0.04),
                    0.,
                );
            }
            prepare(&mut scene, &mut g);
            let report = &scene.indoor_motion.as_ref().unwrap().mechanisms;
            assert_eq!(report.len(), 1);
            assert_eq!(
                report[0].rejection.as_deref(),
                obstructed.then_some("static_geometry"),
                "{kind:?} obstructed={obstructed}"
            );
        }
    }
}
