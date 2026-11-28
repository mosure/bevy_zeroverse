use super::*;
use crate::scene::procedural_indoor::{
    architecture::doors::DoorMechanism,
    program::hallway::{across, front_normal},
    validation::validate_layout,
};
use std::collections::BTreeSet;

#[test]
fn hallway_ports_routes_camera_paths_and_furniture_describe_one_building() {
    let mut arrangements = BTreeSet::new();
    let mut mechanisms = BTreeSet::new();
    let mut glazing = [false; 2];
    let mut banks_with_seats = 0;
    let mut bay_count = 0;
    let mut furnished = 0;
    for seed in 0..36 {
        let scene = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Hallway,
            [0., 0.65, 1.][seed as usize % 3],
            3,
            0.25,
        )
        .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        validate_layout(&scene).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let p = scene.program.as_ref().unwrap();
        let hall = &p.zones[0];
        let a = across(hall);
        let top = scene.room_topology();
        assert_eq!(scene.primary_room_bounds(), (hall.min, hall.max));
        assert_eq!(
            top.regions
                .iter()
                .filter(|z| z.primary)
                .map(|z| z.zone)
                .collect::<Vec<_>>(),
            [0]
        );
        assert!(top.boundary_partitions.is_empty() && top.open_connections.is_empty());
        let doors = scene.door_installations();
        let mut sides = [false; 2];
        let mut reached = BTreeSet::new();
        for portal in top.portals.iter().filter(|p| p.regions.contains(&0)) {
            let w = &p.partitions[portal.partition];
            let n = front_normal(hall, w).unwrap();
            sides[usize::from(n > 0.)] = true;
            assert!(reached.insert(*portal.regions.iter().find(|&&i| i != 0).unwrap()));
            glazing[usize::from(w.glazing_fraction > 0.)] = true;
            let d = doors.iter().find(|d| d.id == portal.partition + 1).unwrap();
            mechanisms.insert(format!("{:?}", d.program.mechanism));
            if matches!(
                d.program.mechanism,
                DoorMechanism::Hinged | DoorMechanism::Paired
            ) && d.program.opening > 0.42
            {
                // Test the actual transformed leaves including axis sign and yaw.
                for leaf in d.leaves() {
                    assert!(
                        (leaf.transform.translation.xz()[a] - w.coordinate) * n < 0.,
                        "{seed}: door swung into passage"
                    );
                }
            }
        }
        assert_eq!(reached.len(), p.zones.len() - 1);
        let half = 0.5 * scene.entry_width();
        assert!(scene.door_x - half >= hall.min.x && scene.door_x + half <= hall.max.x);
        arrangements.insert((a, sides));
        let receipt = &scene.functional_programs()[0];
        let spine = &receipt.reservations[0];
        assert!((spine.max - spine.min)[a] >= 1.33);
        assert_eq!(
            receipt
                .reservations
                .iter()
                .filter(|r| r.role != ReservationRole::SeatingAccess)
                .count(),
            reached.len() + 2
        );
        let structure = scene.columns();
        for route in &receipt.reservations {
            assert!(
                structure
                    .iter()
                    .all(|&(lo, hi)| lo.y >= 1.90 || hi.y <= 0.001 || !route.overlaps(lo, hi)),
                "{seed}: architectural obstruction on {route:?}"
            );
        }
        for branch in receipt
            .reservations
            .iter()
            .filter(|r| r.role == ReservationRole::DoorApproach)
        {
            assert!(branch.min.cmplt(spine.max).all() && branch.max.cmpgt(spine.min).all());
            let mid = (branch.min + branch.max) * 0.5;
            assert!(scene.ceiling_height(mid) - scene.floor_height(mid) > 2.0);
        }
        for obj in scene.objects.iter().filter(|o| o.solid && !o.neighbor) {
            let (lo, hi) = obj.bounds();
            assert!(!receipt.reservations.iter().any(|r| r.overlaps(lo, hi)));
        }
        banks_with_seats += usize::from(receipt.accepted_capacity > 0);
        bay_count += reached.len();
        furnished += scene.functional_programs()[1..]
            .iter()
            .filter(|p| p.accepted_capacity > 0)
            .count();
        assert_eq!(scene.cameras.len(), 3);
        for cam in &scene.cameras {
            assert!(scene.camera_curve_clear(cam));
            let mut trajectory = cam.runtime_trajectory();
            for t in 0..=16 {
                let point = trajectory.sample(t as f32 / 16.).translation.xz();
                assert!(
                    point.cmpge(hall.min).all() && point.cmple(hall.max).all(),
                    "{seed}: camera outside passage"
                );
            }
        }
        let pairs = scene.camera_pair_overlaps();
        assert_eq!(pairs.len(), 3);
        assert!(pairs.iter().all(|p| p
            .samples
            .iter()
            .all(|s| s.reference_to_view.is_finite() && s.view_to_reference.is_finite())));
        if seed == 0 {
            let replay =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Hallway, 0., 3, 0.25)
                    .unwrap();
            assert_eq!(scene, replay);
        }
    }
    assert_eq!(
        arrangements.len(),
        4,
        "two orientations and both outer walls"
    );
    assert_eq!(glazing, [true, true]);
    assert_eq!(mechanisms.len(), 4);
    assert!((3..34).contains(&banks_with_seats));
    assert!(
        furnished * 100 > bay_count * 85,
        "{furnished} occupied / {bay_count} suites"
    );
}

#[test]
fn hallway_validation_rejects_disconnected_frontage_route_or_incorrect_primary() {
    let scene =
        IndoorManifest::generate_with_humans(42200, IndoorLayout::Hallway, 0.65, 0, 0.).unwrap();
    let mut invalid = scene.clone();
    invalid.assembly_diagnostics.programs[0]
        .reservations
        .remove(1);
    assert!(invalid.validate_public_programs().is_err());
    let mut program = scene.program.clone().unwrap();
    program.partitions.remove(0);
    assert!(program.validate(&scene).is_err());
    let mut program = scene.program.clone().unwrap();
    program.primary_zone = Some(1);
    assert!(program.validate(&scene).is_err());
}
