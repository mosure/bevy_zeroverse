use super::*;
use crate::scene::procedural_indoor::{
    architecture::{doors::DoorPanel, interior_glazing},
    materials::{flooring::FloorLayout, Surface},
    objects::{public::education, services::bench},
    validation::validate_layout,
};
use std::collections::BTreeSet;

#[test]
fn changing_rooms_have_working_locker_fronts_supported_seats_and_privacy_glazing() {
    let mut capacities = BTreeSet::new();
    let mut saw_island = false;
    let mut saw_perimeter_only = false;
    let mut saw_towel = false;
    let mut saw_bag = false;
    let mut saw_tile = false;
    let mut saw_terrazzo = false;
    for seed in 0..48 {
        let density = [0., 0.65, 1.][seed as usize % 3];
        let s =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::LockerRoom, density, 0, 0.)
                .unwrap();
        let p = &s.functional_programs()[0];
        validate_layout(&s).unwrap_or_else(|e| {
            panic!(
                "{e}; zone {:?} .. {:?}, reservations {:?}, rejected {} candidates",
                p.min, p.max, p.reservations, p.rejected_candidates
            )
        });
        let members = || p.members.iter().map(|&id| &s.objects[id]);
        let banks = members().filter(|o| o.kind == ObjectKind::Locker).count();
        assert!(banks >= 1 && banks <= p.requested_capacity);
        assert_eq!(p.accepted_capacity, banks);
        assert_eq!(p.capacity_unit, CapacityUnit::LockerBanks);
        assert!(members().any(|o| o.kind == ObjectKind::Bench));
        capacities.insert(banks);
        let islands = members()
            .filter(|o| o.variant == education::ISLAND_BANK && o.kind == ObjectKind::Locker)
            .count();
        assert!(islands == 0 || islands == 2);
        saw_island |= islands == 2;
        saw_perimeter_only |= islands == 0;
        for o in members() {
            match o.kind {
                ObjectKind::Locker => {
                    let bank = education::parameters(o);
                    assert_eq!(
                        bank.compartments(),
                        bank.bays.iter().map(|bay| bay.open.len()).sum::<usize>()
                    );
                    let q = o
                        .transform()
                        .transform_point(Vec3::Z * (o.size.z * 0.5 + 0.24))
                        .xz();
                    assert!(
                        p.reservations
                            .iter()
                            .any(|r| r.role == ReservationRole::DoorApproach
                                && q.cmpge(r.min).all()
                                && q.cmple(r.max).all()),
                        "{seed}"
                    );
                }
                ObjectKind::Bench => {
                    let b = bench::parameters(o);
                    assert!(!b.arms && !b.backed && b.seat_height == o.size.y);
                    assert!(matches!(b.seat, Surface::Wood | Surface::Plastic));
                    let bank = &s.objects[o.interaction_target.unwrap()];
                    assert!(p.members.contains(&bank.id) && bank.kind == ObjectKind::Locker);
                    let (a, b) = o.bounds();
                    assert!(p.reservations.iter().all(|r| !r.overlaps(a, b)));
                }
                ObjectKind::KitchenTowel | ObjectKind::Bag => {
                    let seat = &s.objects[o.support.unwrap()];
                    assert_eq!(seat.kind, ObjectKind::Bench);
                    assert_eq!(s.support_plane(o), Some(SupportPlane::Top));
                    assert!(!seat.supports_item(ObjectKind::Monitor));
                    assert!(s.prop_clear(o, seat, 0.));
                    assert!((o.position.y - seat.position.y - seat.size.y).abs() < 0.001);
                    saw_towel |= o.kind == ObjectKind::KitchenTowel;
                    saw_bag |= o.kind == ObjectKind::Bag;
                }
                _ => {}
            }
        }
        let floor = &s.program.as_ref().unwrap().materials[Surface::Floor as usize];
        let layout = floor.flooring.as_ref().unwrap().layout;
        saw_terrazzo |= layout == FloorLayout::Seamless;
        saw_tile |= layout == FloorLayout::Hexagonal;
        assert!(floor.roughness >= 0.47);
        let e = s.envelope.as_ref().unwrap();
        let shared = interior_glazing::shared_facade(&s, interior_glazing::partition_height(&s));
        assert!(!matches!(
            s.entry_door.as_ref().unwrap().program.panel,
            DoorPanel::Glazed | DoorPanel::VisionLite
        ));
        for f in e
            .walls
            .iter()
            .filter_map(|w| w.facade.as_ref())
            .chain(&s.exterior.as_ref().unwrap().facades)
            .chain([&shared])
        {
            assert!(f
                .openings
                .iter()
                .all(|w| w.min.y >= 1.94 && w.max.y - w.min.y >= 0.37));
        }
    }
    assert!(saw_island && saw_perimeter_only && saw_towel && saw_bag && saw_tile && saw_terrazzo);
    assert!(capacities.len() >= 3);
}

#[test]
fn changing_room_rejects_a_bench_rotated_away_from_its_required_locker() {
    let mut s =
        IndoorManifest::generate_with_humans(7, IndoorLayout::LockerRoom, 0.65, 0, 0.).unwrap();
    let id = s.functional_programs()[0]
        .members
        .iter()
        .copied()
        .find(|&id| s.objects[id].kind == ObjectKind::Bench)
        .unwrap();
    s.objects[id].yaw += 0.6;
    assert!(s
        .validate_public_programs()
        .unwrap_err()
        .contains("changing bench"));
}

#[test]
fn changing_room_aliases_and_mixed_sampling_preserve_other_profiles() {
    for arg in ["changing-room", "locker-room"] {
        assert_eq!(
            <IndoorLayout as clap::ValueEnum>::from_str(arg, true).unwrap(),
            IndoorLayout::LockerRoom
        );
    }
    for value in ["ChangingRoom", "LockerRoom"] {
        assert_eq!(
            serde_json::from_value::<IndoorLayout>(serde_json::json!(value)).unwrap(),
            IndoorLayout::LockerRoom
        );
    }
    let all = &IndoorLayout::PROFILES;
    let previous = &all[..all.len() - 1];
    let mut count = 0;
    for seed in 0..4096 {
        let pick = IndoorLayout::sample_profile(seed, all, &mut stream(seed, 0));
        if pick == IndoorLayout::LockerRoom {
            count += 1;
        } else {
            assert_eq!(
                pick,
                IndoorLayout::sample_profile(seed, previous, &mut stream(seed, 0))
            );
        }
    }
    assert!((70..180).contains(&count));
}

#[test]
fn people_can_walk_from_the_changing_room_entry_to_the_installed_locker_fronts() {
    use crate::human_motion::planning::{obstacles, route};
    for seed in 0..12 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::LockerRoom, 0.8, 0, 0.)
                .unwrap();
        let entry = Vec3::new(scene.door_x, 0., scene.room_size.z * 0.5 - 0.73);
        let boxes: Vec<_> = obstacles(&scene, usize::MAX, None)
            .into_iter()
            .filter(|(lo, hi)| hi.y > 0.04 && lo.y < 1.68)
            .collect();
        let p = &scene.functional_programs()[0];
        for &id in &p.members {
            let bank = &scene.objects[id];
            if bank.kind != ObjectKind::Locker {
                continue;
            }
            let tf = bank.transform();
            let reached = [0., -0.25, 0.25].into_iter().any(|x| {
                [0.36, 0.49].into_iter().any(|z| {
                    let standing =
                        tf.transform_point(Vec3::new(x * bank.size.x, 0., bank.size.z * 0.5 + z));
                    route(&scene, entry, standing, 1.68, 0.20, &boxes).is_some()
                })
            });
            assert!(
                reached,
                "seed {seed}, locker {id}, installation {:?}",
                education::installation(bank)
            );
        }
    }
}
