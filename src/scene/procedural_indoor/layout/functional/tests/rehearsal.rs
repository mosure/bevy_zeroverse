use super::*;
use crate::scene::procedural_indoor::{objects::public::music, validation::validate_layout};
use std::collections::BTreeSet;

#[test]
fn ensemble_banks_admit_complete_player_stations_and_preserve_the_aisle() {
    let mut fronts = BTreeSet::new();
    let mut piano_rooms = 0;
    let mut capacities = BTreeSet::new();
    for seed in 0..48 {
        let s = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::RehearsalRoom,
            [0., 0.65, 1.][seed as usize % 3],
            0,
            0.25,
        )
        .unwrap();
        validate_layout(&s).unwrap_or_else(|e| panic!("room {seed}: {e}"));
        let p = &s.functional_programs()[0];
        let aisle = &p.reservations[0];
        assert_eq!(aisle.role, ReservationRole::Circulation);
        let across = aisle.min.x == p.min.x && aisle.max.x == p.max.x;
        assert!(across || (aisle.min.y == p.min.y && aisle.max.y == p.max.y));
        let mut players = BTreeSet::new();
        let mut stands = 0;
        for &id in &p.members {
            let o = &s.objects[id];
            let (lo, hi) = o.bounds();
            assert!(!aisle.overlaps(lo, hi));
            if matches!(o.kind, ObjectKind::MusicStand | ObjectKind::StagePiano) {
                let player = &s.objects[o.interaction_target.unwrap()];
                assert!(music::faces_player(o, player));
                assert!(players.insert(player.id) && player.interaction_target.is_none());
                if o.kind == ObjectKind::StagePiano {
                    piano_rooms += 1;
                } else {
                    stands += 1;
                }
            }
        }
        assert!(stands >= 2);
        assert_eq!(players.len(), p.accepted_capacity);
        assert!(p.accepted_capacity <= p.requested_capacity);
        capacities.insert(p.accepted_capacity);
        let center = (p.max + p.min) * 0.5;
        let front = p.focus.xz() - center;
        fronts.insert((across, if across { front.x > 0. } else { front.y > 0. }));
    }
    assert_eq!(fronts.len(), 4);
    assert!(capacities.len() >= 6 && (8..42).contains(&piano_rooms));
}

#[test]
fn reversed_music_desk_is_rejected() {
    let mut scene =
        IndoorManifest::generate_with_humans(9, IndoorLayout::RehearsalRoom, 0.6, 0, 0.).unwrap();
    assert!(scene.validate_public_programs().is_ok());
    let stand = scene
        .objects
        .iter_mut()
        .find(|o| o.kind == ObjectKind::MusicStand)
        .unwrap();
    stand.yaw += std::f32::consts::PI;
    assert!(scene
        .validate_public_programs()
        .unwrap_err()
        .contains("facing player"));
}

#[test]
fn percussion_sets_keep_their_player_side_clear_and_share_a_product_line() {
    let mut counts = BTreeSet::new();
    let mut diameters = BTreeSet::new();
    let mut bays = BTreeSet::new();
    let mut changed = false;
    for seed in 0..96 {
        let scene = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::RehearsalRoom,
            [0., 0.65, 1.][seed as usize % 3],
            0,
            0.25,
        )
        .unwrap();
        validate_layout(&scene).unwrap();
        let zone = &scene.functional_programs()[0];
        let drums: Vec<_> = zone
            .members
            .iter()
            .map(|&id| &scene.objects[id])
            .filter(|o| o.kind == ObjectKind::Timpani)
            .collect();
        counts.insert(drums.len());
        let Some(access) = zone
            .reservations
            .iter()
            .find(|r| r.role == ReservationRole::EquipmentAccess)
        else {
            assert!(drums.is_empty());
            continue;
        };
        let aisle = &zone.reservations[0];
        let across = aisle.min.x == zone.min.x && aisle.max.x == zone.max.x;
        let long = usize::from(!across);
        let wide = 1 - long;
        let center = (zone.min + zone.max) * 0.5;
        assert!(zone.members.iter().all(|&id| {
            let (l, h) = scene.objects[id].bounds();
            !access.overlaps(l, h)
        }));
        assert!(
            (access.max[wide] - aisle.min[wide]).abs() < 1e-4
                || (access.min[wide] - aisle.max[wide]).abs() < 1e-4
        );
        let mut sizes = Vec::new();
        for &o in &drums {
            assert!(o.solid && o.support.is_none() && o.finish_seed == drums[0].finish_seed);
            let player = o
                .transform()
                .transform_point(Vec3::Z * (o.size.z / 2. + 0.36));
            assert!(
                player.xz().cmpge(access.min + 0.05).all()
                    && player.xz().cmple(access.max - 0.05).all()
            );
            assert!(
                (zone.focus.xz()[long] - center[long]) * (player.xz()[long] - center[long]) < 0.
            );
            assert!((scene.floor_height(player.xz()) - o.position.y).abs() < 0.035);
            let size = music::Timpani::sample(o).nominal_inches;
            sizes.push(size);
            diameters.insert(size);
        }
        assert!(sizes.windows(2).all(|w| w[0].abs_diff(w[1]) == 3));
        if sizes.len() > 1 {
            bays.insert((wide, sizes[0] < sizes[1]));
        }
        if !changed {
            let mut broken = scene.clone();
            broken.objects[drums[0].id].yaw += std::f32::consts::PI;
            assert!(broken
                .validate_public_programs()
                .unwrap_err()
                .contains("pedal"));
            changed = true;
        }
    }
    assert_eq!(counts, BTreeSet::from([0, 1, 2, 3]));
    assert_eq!(diameters.len(), 5);
    assert!(bays.len() >= 3 && changed);
}
