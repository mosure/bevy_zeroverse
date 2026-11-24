use super::*;
use crate::scene::procedural_indoor::objects::{audit, equipment as product};
use std::collections::BTreeSet;

fn furnish(room: &mut IndoorManifest) {
    if let Some(job) = room.furnish_worktop_printer() {
        room.furnish_print_stand(job);
    }
}

fn fixture(kind: ObjectKind) -> IndoorManifest {
    let mut room =
        IndoorManifest::generate_with_humans(18, IndoorLayout::OpenOffice, 0.65, 0, 0.).unwrap();
    room.envelope = None;
    room.ceiling_installation = None;
    room.program = None;
    room.room_size = Vec3::new(10., 3., 9.);
    room.clutter.probability = 1.;
    let mut support = audit::fixture(kind, 97);
    support.id = 0;
    support.position = Vec3::new(-1.3, 0., 0.8);
    support.size = Vec3::new(1.52, 0.75, 0.80);
    support.yaw = 0.42;
    room.objects = vec![support];
    room
}

#[test]
fn print_stations_use_work_tops_leave_collection_space_and_respect_opt_outs() {
    for kind in [
        ObjectKind::Desk,
        ObjectKind::DrawerUnit,
        ObjectKind::MobileCart,
        ObjectKind::Cabinet,
    ] {
        let base = fixture(kind);
        let mut headings = BTreeSet::new();
        let mut positions = BTreeSet::new();
        let mut wired = 0;
        for seed in 0..80 {
            let mut room = base.clone();
            room.seed = seed;
            room.objects[0].seed = seed;
            let mut replay = room.clone();
            furnish(&mut room);
            furnish(&mut replay);
            assert_eq!(room.objects, replay.objects);
            let Some(printer) = room.objects.get(1).cloned() else {
                continue;
            };
            assert_eq!(printer.support, Some(0));
            assert!(room.prop_clear(&printer, &room.objects[0], 0.012));
            assert!(room.print_station_clear(
                &printer,
                0,
                &room.structural_obstacles_with_ceiling(false)
            ));
            assert!((printer.position.y - 0.75).abs() < 1e-5);
            let local = room.objects[0]
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(printer.position);
            positions.insert((local.x > 0., local.z > 0.));
            headings.insert(
                ((printer.yaw - room.objects[0].yaw).rem_euclid(std::f32::consts::TAU) * 2.) as i32,
            );
            furnish(&mut room);
            assert_eq!(room.objects.len(), 2);
            let mut block = audit::fixture(ObjectKind::StorageBox, seed);
            block.position = printer.transform().transform_point(Vec3::new(
                0.,
                0.03,
                printer.size.z * 0.5 + 0.08,
            ));
            block.size = Vec3::splat(0.07);
            block.id = 2;
            room.objects.push(block);
            assert!(!room.print_station_clear(&printer, 0, &[]));
            room.objects.pop();
            let mut target = audit::fixture(ObjectKind::PowerStrip, seed);
            target.support = Some(0);
            target.id = 2;
            target.size = Vec3::new(0.22, 0.045, 0.09);
            target.position = room.objects[0].transform().transform_point(Vec3::new(
                if local.x > 0. { -0.56 } else { 0.56 },
                0.75,
                0.,
            ));
            target.yaw = room.objects[0].yaw;
            room.objects.push(target);
            room.connect_equipment();
            for cable in room
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::Cable && o.interaction_target == Some(printer.id))
            {
                assert!(equipment::cable_route_clear(&room, cable));
                let path = cable.cable_path.as_ref().unwrap();
                assert!(
                    cable
                        .transform()
                        .transform_point(path[0])
                        .distance(product::power_port(&printer, false).unwrap())
                        < 0.0001
                );
                wired += 1;
            }
        }
        assert!(
            headings.len() >= 4 && positions.len() >= 3,
            "{kind:?}: {headings:?}, {positions:?}"
        );
        assert!(wired > 0, "{kind:?}: no compatible power route");
    }
    let mut clean = fixture(ObjectKind::Desk);
    clean.clutter.probability = 0.;
    let mut dining = fixture(ObjectKind::Cabinet);
    dining.layout = IndoorLayout::Cafeteria;
    for seed in 0..50 {
        for room in [&mut clean, &mut dining] {
            room.seed = seed;
            furnish(room);
            assert_eq!(room.objects.len(), 1);
        }
    }
}

#[test]
fn freestanding_print_stations_are_paired_and_clear_of_the_power_outlet() {
    let mut base = fixture(ObjectKind::Desk);
    let mut outlet = audit::fixture(ObjectKind::WallOutlet, 32);
    outlet.position = Vec3::new(0., 0.32, -4.42);
    outlet.yaw = 0.;
    outlet.id = 0;
    let mut blocked = outlet.clone();
    blocked.position.x = 8.;
    base.objects = vec![outlet];
    let mut types = BTreeSet::new();
    for seed in 0..160 {
        let mut room = base.clone();
        room.seed = seed;
        furnish(&mut room);
        if room.objects.len() == 1 {
            continue;
        }
        assert_eq!(room.objects.len(), 3);
        let stand = &room.objects[1];
        let printer = &room.objects[2];
        assert_eq!(printer.support, Some(stand.id));
        assert_eq!(stand.support, None);
        let (a, b) = stand.bounds();
        let (c, d) = room.objects[0].bounds();
        assert!(!((a.cmplt(d)).all() && b.cmpgt(c).all()));
        assert!(room.prop_clear(printer, stand, 0.012));
        assert!(room.print_station_clear(
            printer,
            stand.id,
            &room.structural_obstacles_with_ceiling(false)
        ));
        types.insert(stand.kind.class_name());
        let mut invalid = base.clone();
        invalid.objects[0] = blocked.clone();
        invalid.seed = seed;
        furnish(&mut invalid);
        assert_eq!(invalid.objects.len(), 1);
    }
    assert_eq!(types.len(), 2);
}
