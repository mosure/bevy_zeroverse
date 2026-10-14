use super::*;

fn object(id: usize, kind: ObjectKind, position: Vec3, size: Vec3) -> IndoorObject {
    IndoorObject {
        finish_seed: None,
        id,
        kind,
        position,
        size,
        yaw: 0.,
        variant: 0,
        seed: 123 + id as u64,
        solid: id == 0,
        support: (id > 0).then_some(0),
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    }
}

fn connected_scene() -> (IndoorManifest, IndoorObject) {
    let mut scene =
        IndoorManifest::generate_with_humans(7, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let table = object(
        0,
        ObjectKind::Table,
        Vec3::ZERO,
        Vec3::new(1.60, 0.75, 1.00),
    );
    let laptop = object(
        1,
        ObjectKind::Laptop,
        Vec3::new(-0.30, 0.75, 0.),
        Vec3::new(0.30, 0.27, 0.35),
    );
    let strip = object(
        2,
        ObjectKind::PowerStrip,
        Vec3::new(0.35, 0.75, -0.07),
        Vec3::new(0.25, 0.03, 0.055),
    );
    let start = equipment::power_port(&laptop, true).unwrap();
    let end = equipment::power_port(&strip, true).unwrap();
    let world = [
        start,
        start + Vec3::X * 0.039,
        Vec3::new(0.03, 0.761, 0.10),
        Vec3::new(0.25, 0.761, 0.10),
        end + Vec3::Z * 0.039,
        end,
    ];
    let mut cable = object(3, ObjectKind::Cable, Vec3::ZERO, Vec3::ONE);
    cable.interaction_target = Some(1);
    cable.cable_target = Some(2);
    let world = wires::fillet_path(&world, wires::radius(&cable)).unwrap();
    let r = wires::bound_radius(&cable);
    let lo = world
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
        - Vec3::splat(r);
    let hi = world
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
        + Vec3::splat(r);
    cable.position = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    cable.size = hi - lo;
    cable.cable_path = Some(world.iter().map(|p| *p - cable.position).collect());
    scene.objects = vec![table, laptop, strip];
    scene.program = None;
    scene.envelope = None;
    scene.floor_plan = super::super::super::floorplan::FloorPlan::OpenHall;
    (scene, cable)
}

#[test]
fn cables_validate_ports_bounds_support_and_segment_crossings() {
    let (mut scene, cable) = connected_scene();
    assert!(cable_route_clear(&scene, &cable));
    let replay: IndoorObject =
        serde_json::from_slice(&serde_json::to_vec(&cable).unwrap()).unwrap();
    assert_eq!(cable, replay);
    assert!(cable_route_clear(&scene, &replay));
    let mut changed = cable.clone();
    changed.cable_path.as_mut().unwrap()[0].x += 0.003;
    assert!(
        !cable_route_clear(&scene, &changed),
        "floating endpoint must fail"
    );
    changed = cable.clone();
    changed.cable_target = Some(1);
    assert!(!cable_route_clear(&scene, &changed));
    changed = cable.clone();
    changed.size.x += 0.003;
    assert!(
        !cable_route_clear(&scene, &changed),
        "dummy bounds must fail"
    );
    changed = cable.clone();
    changed.cable_path.as_mut().unwrap()[1].y = f32::NAN;
    assert!(!cable_route_clear(&scene, &changed));
    changed = cable.clone();
    changed.cable_path.as_mut().unwrap().push(Vec3::ZERO);
    assert!(!cable_route_clear(&scene, &changed));
    scene.objects.push(object(
        3,
        ObjectKind::Mug,
        Vec3::new(0.14, 0.75, 0.10),
        Vec3::new(0.05, 0.08, 0.05),
    ));
    let mut late = cable.clone();
    late.id = 4;
    assert!(
        !cable_route_clear(&scene, &late),
        "wire cannot pass through existing clutter"
    );
}

#[test]
fn prop_clear_uses_wire_segments_and_preserves_connected_ports() {
    let (mut scene, cable) = connected_scene();
    scene.objects.push(cable.clone());
    let support = &scene.objects[0];
    for id in [1, 2] {
        assert!(scene.prop_clear(&scene.objects[id], support, 0.001));
        let mut moved = scene.objects[id].clone();
        moved.position.x += 0.006;
        assert!(
            !scene.prop_clear(&moved, support, 0.001),
            "moving a wired endpoint must invalidate its stale connection"
        );
        let mut rotated = scene.objects[id].clone();
        rotated.yaw += 0.35;
        assert!(
            !scene.prop_clear(&rotated, support, 0.001),
            "rotating a wired endpoint must retain its physical port"
        );
        let mut detached = scene.objects[id].clone();
        detached.support = None;
        assert!(!scene.prop_clear(&detached, support, 0.001));
    }
    let crossing = object(
        4,
        ObjectKind::Mug,
        Vec3::new(0.14, 0.75, 0.10),
        Vec3::new(0.025, 0.04, 0.025),
    );
    assert!(
        !scene.prop_clear(&crossing, support, 0.001),
        "new props cannot intersect an existing wire jacket"
    );
    let mut vacant = crossing;
    vacant.position.z = -0.025;
    let (wire_lo, wire_hi) = cable.bounds();
    assert!(
        vacant.position.x > wire_lo.x
            && vacant.position.x < wire_hi.x
            && vacant.position.z > wire_lo.z
            && vacant.position.z < wire_hi.z,
        "fixture must exercise vacant space inside the wire's enclosing rectangle"
    );
    assert!(scene.prop_clear(&vacant, support, 0.001));

    // Replaying the same connected system in a rotated support frame must not
    // change its port contacts or clearance result.
    let rotation = Quat::from_rotation_y(0.7);
    for o in &mut scene.objects {
        o.position = rotation * o.position + Vec3::new(0.12, 0., 0.15);
        o.yaw += 0.7;
    }
    for id in [1, 2] {
        assert!(scene.prop_clear(&scene.objects[id], &scene.objects[0], 0.001));
    }
}

#[test]
fn archived_wire_reversal_is_rejected_before_geometry_construction() {
    let (scene, mut cable) = connected_scene();
    assert!(cable_route_clear(&scene, &cable));
    let points = cable.cable_path.as_mut().unwrap();
    let p = points[2];
    // Both added segments remain nonzero and inside the original tight bounds;
    // at the new middle vertex the tube's centered tangent would be zero.
    points.insert(3, p + Vec3::X * 0.04);
    points.insert(4, p);
    assert!(points.windows(2).all(|pair| pair[0] != pair[1]));
    assert!(points.windows(3).any(|triple| triple[0] == triple[2]));
    assert!(!cable_route_clear(&scene, &cable));
    let replay: IndoorObject =
        serde_json::from_slice(&serde_json::to_vec(&cable).unwrap()).unwrap();
    assert!(!cable_route_clear(&scene, &replay));
}

#[test]
fn equipment_systems_have_repeatable_nonzero_presence() {
    let mut powered_monitors = 0;
    let mut counts = std::collections::BTreeMap::new();
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        for o in scene.objects.iter().filter(|o| !o.neighbor) {
            if o.kind == ObjectKind::Cable {
                assert!(cable_route_clear(&scene, o), "seed {seed} cable {}", o.id);
                powered_monitors += usize::from(
                    scene.objects[o.interaction_target.unwrap()].kind == ObjectKind::Monitor,
                );
            }
            if o.kind == ObjectKind::CeilingPendant {
                assert!(
                    ceiling_fixture_clear(&scene, o),
                    "seed {seed} pendant {}",
                    o.id
                );
            }
            if o.kind == ObjectKind::Projector {
                let parent = &scene.objects[o.support.expect("tabletop projector is supported")];
                assert!((0.60..=1.22).contains(&(parent.position.y + parent.size.y)));
                assert!(scene.prop_clear(o, parent, 0.001));
                assert!(
                    scene
                        .objects
                        .iter()
                        .filter(|s| s.kind == ObjectKind::ProjectionScreen)
                        .any(|screen| projector_axis(&scene, o, screen).is_some()),
                    "seed {seed}: projector must face a plausible unblocked architectural target"
                );
                let port = equipment::power_port(o, false).unwrap();
                let (lo, hi) = o.bounds();
                assert!(
                    port.cmpge(lo - Vec3::splat(0.002)).all()
                        && port.cmple(hi + Vec3::splat(0.002)).all()
                );
                let replay: IndoorObject =
                    serde_json::from_slice(&serde_json::to_vec(o).unwrap()).unwrap();
                assert_eq!(port, equipment::power_port(&replay, false).unwrap());
            }
            *counts.entry(o.kind as usize).or_insert(0usize) += 1;
        }
    }
    assert!(
        powered_monitors > 0,
        "accepted workspaces must exercise mains-connected monitor leads"
    );

    for kind in [
        ObjectKind::WallShelf,
        ObjectKind::NoticeBoard,
        ObjectKind::ProjectionScreen,
        ObjectKind::WallSconce,
        ObjectKind::CeilingPendant,
        ObjectKind::PowerStrip,
        ObjectKind::Projector,
        ObjectKind::Cable,
    ] {
        assert!(
            counts.get(&(kind as usize)).copied().unwrap_or(0) >= 4,
            "{kind:?} unreachable: {counts:?}"
        );
    }
}

#[test]
fn projector_installations_reserve_supported_space_and_reject_bad_optical_axes() {
    let (mut scene, _) = connected_scene();
    scene.room_size = Vec3::new(8., 3.2, 8.);
    let table = object(
        0,
        ObjectKind::Table,
        Vec3::ZERO,
        Vec3::new(1.60, 0.75, 1.00),
    );
    let mut screen = object(
        1,
        ObjectKind::ProjectionScreen,
        Vec3::new(0., 0.95, 3.),
        Vec3::new(1.6, 1.2, 0.07),
    );
    screen.support = None;
    screen.yaw = PI;
    let mut installed = 0;
    for seed in 0..16 {
        screen.seed = seed;
        scene.objects = vec![table.clone(), screen.clone()];
        let mut replay = scene.clone();
        scene.projector_equipment(std::slice::from_ref(&table));
        replay.projector_equipment(std::slice::from_ref(&table));
        assert_eq!(scene.objects, replay.objects);
        if let Some(projector) = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Projector)
            .cloned()
        {
            installed += 1;
            assert!(scene.prop_clear(&projector, &table, 0.012));
            assert!(projector_axis_clear(&scene, &projector, &screen));
            let mut backward = projector.clone();
            backward.yaw += PI;
            assert!(!projector_axis_clear(&scene, &backward, &screen));
            let mut back_screen = screen.clone();
            back_screen.yaw = 0.;
            assert!(!projector_axis_clear(&scene, &projector, &back_screen));
            let mut obstruction = object(
                scene.objects.len(),
                ObjectKind::Cabinet,
                Vec3::new(0., 0., 1.4),
                Vec3::new(0.8, 2., 0.5),
            );
            obstruction.support = None;
            obstruction.solid = true;
            scene.objects.push(obstruction);
            assert!(!projector_axis_clear(&scene, &projector, &screen));
        }
    }
    assert!(
        installed >= 8,
        "dedicated presentation supports must be reachable: {installed}/16"
    );
}

#[test]
fn projector_optics_reach_physical_wall_screens_without_camera_clearance() {
    use super::super::super::envelope::{EnvelopeProgram, EnvelopeWall};
    let (mut scene, _) = connected_scene();
    scene.room_size = Vec3::new(8., 3.2, 8.);
    scene.envelope = Some(EnvelopeProgram {
        soffits: vec![],
        footprint: vec![
            Vec2::new(-4., -4.),
            Vec2::new(4., -4.),
            Vec2::new(4., 4.),
            Vec2::new(-4., 4.),
        ],
        ceiling_drop: Vec2::ZERO,
        floor_patches: Vec::new(),
        pillars: Vec::new(),
        mezzanine: None,
        walls: [0, 1, 3]
            .map(|edge| EnvelopeWall { edge, facade: None })
            .into(),
    });
    scene.envelope.as_ref().unwrap().validate(&scene).unwrap();
    let tf = scene.envelope.as_ref().unwrap().wall_transform(0);
    let mut table = object(
        0,
        ObjectKind::Table,
        tf.transform_point(Vec3::new(0., 0., 2.4)),
        Vec3::new(1.6, 0.75, 1.0),
    );
    table.yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
    let mut screen = object(
        1,
        ObjectKind::ProjectionScreen,
        tf.transform_point(Vec3::new(0., 0.95, 0.07 * 0.5 + 0.001)),
        Vec3::new(1.6, 1.2, 0.07),
    );
    screen.yaw = table.yaw;
    screen.support = None;
    assert!(scene.wall_attachment_clear(&screen));
    let mut installed = 0;
    for seed in 0..16 {
        screen.seed = seed;
        scene.objects = vec![table.clone(), screen.clone()];
        scene.projector_equipment(std::slice::from_ref(&table));
        let Some(projector) = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Projector)
        else {
            continue;
        };
        installed += 1;
        let (origin, target) = projector_axis(&scene, projector, &screen).unwrap();
        let e = scene.envelope.as_ref().unwrap();
        assert!(projector_axis_clear(&scene, projector, &screen));
        assert!(
            !e.segment_clear(scene.room_size, origin, target, 0.008),
            "camera-body wall margin cannot serve as a physical optical ray test"
        );
        let mut obstructed = scene.clone();
        // A concave exterior notch interrupts the optical ray while both ends
        // remain in the room. The physical first-surface test must still reject
        // this installation: removing the camera rail cannot remove walls.
        obstructed.envelope.as_mut().unwrap().footprint = vec![
            Vec2::new(-4., -4.),
            Vec2::new(4., -4.),
            Vec2::new(4., -3.),
            Vec2::new(-0.4, -3.),
            Vec2::new(-0.4, -2.),
            Vec2::new(4., -2.),
            Vec2::new(4., 4.),
            Vec2::new(-4., 4.),
        ];
        assert!(projector_axis(&obstructed, projector, &screen).is_none());
    }
    assert!(
        installed >= 8,
        "wall-screen installations are reachable: {installed}/16"
    );
}

#[test]
fn monitor_power_leads_use_the_rendered_pedestal_port_and_reject_housing_crossings() {
    let (mut scene, _) = connected_scene();
    scene.objects[1].kind = ObjectKind::Monitor;
    scene.objects[1].size = Vec3::new(0.40, 0.45, 0.24);
    scene.objects[1].position.z = -0.15;
    let monitor = &scene.objects[1];
    let strip = &scene.objects[2];
    let start = equipment::power_port(monitor, false).unwrap();
    let source_normal = equipment::power_port_normal(monitor, false).unwrap();
    let end = equipment::power_port(strip, false).unwrap();
    let target_normal = equipment::power_port_normal(strip, false).unwrap();
    assert!(equipment::power_port(monitor, true).is_none());
    assert!(equipment::power_port_normal(monitor, true).is_none());
    let world = [
        start,
        start + source_normal * 0.039,
        Vec3::new(-0.07, start.y, start.z - 0.039),
        Vec3::new(0.10, end.y + 0.039, -0.25),
        end + target_normal * 0.039 - Vec3::Z * 0.10,
        end + target_normal * 0.039,
        end,
    ];
    let mut cable = object(3, ObjectKind::Cable, Vec3::ZERO, Vec3::ONE);
    cable.interaction_target = Some(1);
    cable.cable_target = Some(2);
    let world = wires::fillet_path(&world, wires::radius(&cable)).unwrap();
    let r = wires::bound_radius(&cable);
    let lo = world
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
        - Vec3::splat(r);
    let hi = world
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
        + Vec3::splat(r);
    cable.position = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    cable.size = hi - lo;
    cable.cable_path = Some(world.into_iter().map(|p| p - cable.position).collect());
    assert!(cable_route_clear(&scene, &cable));
    scene.objects.push(cable.clone());
    assert!(scene.prop_clear(&scene.objects[1], &scene.objects[0], 0.001));
    let mut moved = scene.objects[1].clone();
    moved.yaw += 0.2;
    assert!(!scene.prop_clear(&moved, &scene.objects[0], 0.001));
    let invalid_controls = [
        start,
        start + source_normal * 0.039,
        Vec3::new(-0.07, start.y, start.z - 0.039),
        Vec3::new(-0.30, 0.79, -0.15),
        Vec3::new(-0.60, 0.79, 0.05),
        Vec3::new(-0.60, 0.79, 0.20),
        Vec3::new(-0.20, 0.79 + (end.y + 0.039 - 0.79) * (4. / 7.), 0.20),
        Vec3::new(0.10, end.y + 0.039, 0.20),
        end + target_normal * 0.039,
        end,
    ];
    let invalid_world = wires::fillet_path(&invalid_controls, wires::radius(&cable)).unwrap();
    let mut crossing = cable.clone();
    let lo = invalid_world
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
        - Vec3::splat(r);
    let hi = invalid_world
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
        + Vec3::splat(r);
    crossing.position = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    crossing.size = hi - lo;
    crossing.cable_path = Some(
        invalid_world
            .into_iter()
            .map(|p| p - crossing.position)
            .collect(),
    );
    assert!(
        wires::path_qualified(&crossing),
        "crossing fixture must retain valid bend geometry"
    );
    assert!(!cable_route_clear(&scene, &crossing));
    let rotation = Quat::from_rotation_y(0.7);
    for o in &mut scene.objects {
        o.position = rotation * o.position + Vec3::new(0.12, 0., 0.15);
        o.yaw += 0.7;
    }
    assert!(cable_route_clear(&scene, &scene.objects[3]));
    assert!(scene.prop_clear(&scene.objects[1], &scene.objects[0], 0.001));
    crossing.position = rotation * crossing.position + Vec3::new(0.12, 0., 0.15);
    crossing.yaw += 0.7;
    assert!(wires::path_qualified(&crossing));
    assert!(!cable_route_clear(&scene, &crossing));
}

#[test]
fn rotated_wire_clearance_broad_phase_contains_the_padded_reserved_box() {
    let radius = 0.004;
    let mut phone = object(
        1,
        ObjectKind::Phone,
        Vec3::new(-3.3, 0.74, 3.4),
        Vec3::new(0.072, 0.010, 0.164),
    );
    for yaw in [0., 0.31, std::f32::consts::FRAC_PI_4, 2.2, 4.397459] {
        phone.yaw = yaw;
        let tf = phone.transform();
        let hits_phone = cable_prop_intersector(&phone, radius);
        for x in [-1., 1.] {
            for z in [-1., 1.] {
                // A millimetric vertical lead at a padded local corner must
                // agree with the exact rotated box, even outside the old
                // world-AABB-plus-radius broad phase.
                let corner = Vec3::new(
                    x * (phone.size.x * 0.5 + radius - 0.0002),
                    0.003,
                    z * (phone.size.z * 0.5 + radius - 0.0002),
                );
                let start = tf.transform_point(corner);
                let end = tf.transform_point(corner + Vec3::Y * 0.004);
                assert!(hits_phone(start, end), "yaw {yaw} corner {x}/{z}");
                if yaw == std::f32::consts::FRAC_PI_4 && x == -1. && z == -1. {
                    let (lo, _) = phone.bounds();
                    assert!(start.x < lo.x - radius);
                }
            }
        }
        let start = tf.transform_point(Vec3::new(phone.size.x * 0.5 + radius + 0.01, 0., 0.));
        assert!(!hits_phone(start, start + Vec3::Y * phone.size.y));
    }
}

#[test]
fn reported_dense_and_rotated_equipment_routes_retain_layout_and_mesh_validity() {
    for (seed, layout, density, cameras, humans) in [
        (49, IndoorLayout::Mixed, 0.65, 0, 0.),
        (192, IndoorLayout::Conference, 1., 0, 0.),
        (1280, IndoorLayout::Mixed, 0.65, 3, 0.25),
    ] {
        let scene =
            IndoorManifest::generate_with_humans(seed, layout, density, cameras, humans).unwrap();
        for prop in scene.objects.iter().filter(|o| o.support.is_some()) {
            let support = &scene.objects[prop.support.unwrap()];
            assert!(
                scene.prop_clear(prop, support, 0.),
                "seed {seed} {:?} {} on {:?} {}",
                prop.kind,
                prop.id,
                support.kind,
                support.id
            );
        }
        for cable in scene.objects.iter().filter(|o| o.kind == ObjectKind::Cable) {
            assert!(
                cable_route_clear(&scene, cable),
                "seed {seed} cable {}",
                cable.id
            );
        }
        super::super::super::validation::validate_layout(&scene).unwrap();
        super::super::super::validation::validate_geometry(&scene).unwrap();
    }
}

#[test]
fn wall_power_routes_dangle_from_real_ports_without_filling_their_empty_envelope() {
    let (mut routes, mut hanging) = (0, 0);
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        for cable in scene.objects.iter().filter(|o| {
            o.kind == ObjectKind::Cable
                && o.cable_target
                    .is_some_and(|id| scene.objects[id].kind == ObjectKind::WallOutlet)
        }) {
            routes += 1;
            assert!(cable_route_clear(&scene, cable), "seed {seed}");
            let parent = &scene.objects[cable.support.unwrap()];
            hanging += usize::from(cable.bounds().0.y < parent.position.y + parent.size.y - 0.10);
            let (lo, hi) = cable.bounds();
            let occupied: f32 = cable
                .obstacle_bounds()
                .map(|(a, b)| (b - a).element_product())
                .sum();
            assert!(
                occupied < (hi - lo).element_product(),
                "wire must not reserve its empty enclosing box"
            );
            let mut moved_scene = scene.clone();
            let target = cable.cable_target.unwrap();
            moved_scene.objects[target].position.x += 0.05;
            assert!(
                !cable_route_clear(&moved_scene, cable),
                "moving a receptacle must invalidate its cable"
            );
        }
    }
    assert!(
        routes >= 12 && hanging >= 10,
        "wall routes={routes}, hanging={hanging}"
    );
}
