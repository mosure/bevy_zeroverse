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
    assert!(report.mechanisms[0]
        .swept_bounds
        .iter()
        .all(|&a| report.mechanisms[1]
            .swept_bounds
            .iter()
            .all(|&b| !overlaps(a, b))));
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
    assert_eq!(scene.indoor_motion.unwrap().mechanisms.len(), 1);
    assert!(geometry.architecture.joints.iter().all(|j| j.accepted));
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
