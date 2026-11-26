use super::*;

#[test]
fn desk_fan_spins_inside_its_fixed_guard_at_every_mount_angle() {
    use objects::accessories::fan;
    let (blank, mut g) = room();
    let mut styles = [0_usize; 2];
    let mut selected_at_half = 0;
    for seed in 0..24 {
        let mut scene = blank.clone();
        let mut o = fixture(ObjectKind::DeskFan, seed);
        o.size = [Vec3::new(0.18, 0.27, 0.15), Vec3::new(0.32, 0.48, 0.23)][seed as usize % 2];
        o.yaw = seed as f32 * 0.71;
        o.position = Vec3::new(1.3, 0.78, -2.6);
        let fan = fan::parameters(&o);
        let head = o.transform().mul_transform(fan.head_transform());
        let turns = if seed % 2 == 0 { 0.25 } else { 120. };
        let policy = IndoorMotionConfig {
            desk_fan_fraction: 1.,
            fan_turns: [turns; 2],
            ..policy()
        };
        let rest = objects::build_manifest_object(&scene, &o);
        assert!(rest.joints.is_empty());
        let half = IndoorMotionConfig {
            desk_fan_fraction: 0.5,
            ..policy.clone()
        };
        selected_at_half += usize::from(
            !objects::build_object_with_motion(&o, Some(&half))
                .joints
                .is_empty(),
        );
        if seed == 0 {
            let disabled = IndoorMotionConfig {
                fan_turns: [0.; 2],
                ..policy.clone()
            };
            assert!(objects::build_object_with_motion(&o, Some(&disabled))
                .joints
                .is_empty());
        }
        let mut archived = scene.clone();
        archived.generator_version = ROTOR_CARGO_VERSION - 1;
        archived.indoor_motion = Some(IndoorMotionReport::new(policy.clone()));
        assert!(objects::build_manifest_object(&archived, &o)
            .joints
            .is_empty());
        scene.indoor_motion = Some(IndoorMotionReport::new(policy));
        g.objects.clear();
        add(&mut scene, &mut g, o);
        assert_eq!(rest_vertices(&rest), rest_vertices(&g.objects[0]));
        assert_eq!(
            rest.parts.values().map(|g| g.indices.len()).sum::<usize>(),
            g.objects[0]
                .parts
                .values()
                .map(|g| g.indices.len())
                .sum::<usize>()
        );
        prepare(&mut scene, &mut g);
        let m = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
        assert!(m.rejection.is_none(), "seed {seed} {m:?}");
        assert_eq!(m.kind, MotionKind::Fan);
        assert_eq!(m.semantic_labels, [ObjectKind::DeskFan.class_name()]);
        assert_eq!((m.attempts, m.interval), (1, [0., 1.]));
        assert!((m.angle_radians.abs() / std::f32::consts::TAU - turns).abs() < 0.0001);
        assert!(m.axis.distance(head.rotation * Vec3::Z) < 1e-5);
        assert!(m.pivot.distance(head.translation) < 1e-5);
        let j = &g.objects[0].joints[0];
        let vertices = &g.objects[0].parts[&j.parts[0]].positions;
        let (lo, hi) = m.swept_bounds[0];
        for t in [0., 0.017, 0.197, 0.67, 1., 0.197, 0.] {
            let delta = m.rigid_transform_at(t).unwrap();
            for point in vertices.iter().step_by(7) {
                let p = Vec3::from_array(*point);
                let actual = scene.objects[0]
                    .transform()
                    .transform_point(j.transform(j.amount(t)).transform_point(p));
                let expected =
                    delta.transform_point(scene.objects[0].transform().transform_point(p));
                assert!(actual.distance(expected) < 1e-5);
                assert!(actual.cmpge(lo).all() && actual.cmple(hi).all());
            }
        }
        // A stationary intruder between sampled blade phases still vetoes rotation.
        g.architecture.box_part(
            Surface::Plastic,
            "obstruction",
            head.transform_point(Vec3::X * (fan.radius * 0.55)),
            Vec3::splat(0.008),
            0.,
        );
        prepare(&mut scene, &mut g);
        assert_eq!(
            scene.indoor_motion.as_ref().unwrap().mechanisms[0]
                .rejection
                .as_deref(),
            Some("static_geometry")
        );
        g.architecture = default();
        styles[fan.yoke as usize] += 1;
    }
    assert!(styles.into_iter().all(|n| n > 0));
    assert!((1..24).contains(&selected_at_half));
}

#[test]
fn a_tethered_desk_fan_can_spin_on_a_cart_that_can_no_longer_roll() {
    let (mut scene, mut g) = room();
    scene
        .indoor_motion
        .as_mut()
        .unwrap()
        .policy
        .desk_fan_fraction = 1.;
    let cart = fixture(ObjectKind::MobileCart, 7);
    let top = cart.size.y;
    add(&mut scene, &mut g, cart);
    let mut fan = fixture(ObjectKind::DeskFan, 5);
    fan.support = Some(0);
    fan.position = Vec3::Y * top;
    add(&mut scene, &mut g, fan);
    prepare(&mut scene, &mut g);
    let carried = scene.indoor_motion.clone().unwrap();
    assert_eq!(carried.mechanisms.len(), 1);
    assert_eq!(
        carried.mechanism_for_object(1).unwrap().kind,
        MotionKind::Cart
    );
    assert_eq!(g.objects[1].joints[0].kind, MotionKind::Cart);

    let mut cord = fixture(ObjectKind::Cable, 27);
    cord.id = 2;
    cord.cable_target = Some(1);
    scene.objects.push(cord);
    g.objects.push(default());
    prepare(&mut scene, &mut g);
    let parked = scene.indoor_motion.as_ref().unwrap();
    assert_eq!(
        parked.mechanisms[0].rejection.as_deref(),
        Some("connected_cable")
    );
    assert!(parked.mechanisms[0].carried_object_ids.is_empty());
    assert_eq!(
        parked.mechanism_for_object(1).unwrap().kind,
        MotionKind::Fan
    );
    let fan = &g.objects[1];
    assert!(fan.joints[0].accepted && fan.joints[0].parts.len() < fan.parts.len());
    scene.objects.pop();
    g.objects.pop();
    prepare(&mut scene, &mut g);
    assert_eq!(scene.indoor_motion.as_ref(), Some(&carried));
}
