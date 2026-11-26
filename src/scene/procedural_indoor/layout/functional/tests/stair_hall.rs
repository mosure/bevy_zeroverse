use super::*;
use crate::scene::procedural_indoor::{envelope::polygon, validation::validate_layout};
use std::collections::BTreeSet;

#[test]
fn stair_hall_cli_aliases_share_the_archived_program_name() {
    for name in ["stair-hall", "stairwell", "stairway"] {
        let parsed = <IndoorLayout as clap::ValueEnum>::from_str(name, true).unwrap();
        assert_eq!(parsed, IndoorLayout::StairHall);
        assert_eq!(serde_json::to_value(parsed).unwrap(), "StairHall");
    }
}

#[test]
fn stair_halls_connect_the_entrance_to_a_supported_upper_gallery() {
    let mut runs = BTreeSet::new();
    let mut structures = BTreeSet::new();
    let mut guards = BTreeSet::new();
    let mut landings = [false; 2];
    let mut uses = BTreeSet::new();
    let mut seats = 0;
    for seed in 0..36 {
        let count = if seed < 7 { 3 } else { 0 };
        let density = [0., 0.65, 1.][seed as usize % 3];
        let s = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::StairHall,
            density,
            count,
            0.25,
        )
        .unwrap();
        validate_layout(&s).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let e = s.envelope.as_ref().unwrap();
        let m = e.mezzanine.as_ref().unwrap();
        let c = m.construction.as_ref().unwrap();
        let axis = m.stair_axis;
        runs.insert((axis, c.ascending_positive));
        landings[usize::from(c.landing.is_some())] = true;
        guards.insert(format!("{:?}", c.guard.infill));
        structures.insert(format!("{:?}", c.stair_structure));
        assert_eq!(
            e.walls
                .iter()
                .filter(|w| {
                    let end = e.footprint[(w.edge + 1) % e.footprint.len()];
                    w.facade.is_none() && m.abuts_wall_segment(e.footprint[w.edge], end)
                })
                .count(),
            1
        );
        let regions = support_regions(&s);
        let tops: Vec<_> = regions
            .iter()
            .filter(|r| r.name.starts_with("stair_tread_"))
            .collect();
        assert_eq!(tops.len(), m.steps as usize);
        let circulation: Vec<_> = s.functional_programs()[0]
            .reservations
            .iter()
            .filter(|r| r.role == ReservationRole::Circulation)
            .collect();
        for r in circulation.windows(2) {
            assert!(r[0].min.cmplt(r[1].max).all() && r[0].max.cmpgt(r[1].min).all());
        }
        let entry = Vec2::new(s.door_x, s.room_size.z * 0.5 - 0.20);
        assert!(entry.cmpgt(circulation[0].min).all() && entry.cmplt(circulation[0].max).all());
        for (&top, (lo, hi)) in tops.iter().zip(m.treads()) {
            assert_eq!(top.min, lo.xz());
            assert_eq!(top.max, hi.xz());
            assert!((top.elevation_metres - hi.y).abs() < 1e-5 && top.triangles.len() == 2);
            let p = (lo.xz() + hi.xz()) * 0.5;
            assert!(circulation
                .last()
                .is_some_and(|r| p.cmpge(r.min).all() && p.cmple(r.max).all()));
            assert!(e.ceiling_height(s.room_size, p) - hi.y >= 2.2);
        }
        // Every leg up to the foot is on clear ground, including the side aisle
        // in halls where the elevated deck separates the entry from the stairs.
        let fixed = e.structural_boxes(s.room_size);
        for r in &circulation[..circulation.len() - 1] {
            assert!(polygon::box_inside(&e.footprint, r.min, r.max, 0.));
            assert!(
                fixed
                    .iter()
                    .all(|&(a, b)| a.y >= 1.90 || b.y <= 0.001 || !r.overlaps(a, b)),
                "ground route at seed {seed}: {r:?}"
            );
        }
        for o in s
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor && (o.position.y - m.deck.height).abs() < 0.001)
        {
            let (lo, hi) = o.bounds();
            assert!(e.support_clear(lo, hi));
            assert!(circulation.iter().all(|r| !r.overlaps(lo, hi)));
            uses.insert(o.kind as usize);
            seats += usize::from(matches!(o.kind, ObjectKind::Bench | ObjectKind::Banquette));
        }
        if count > 0 {
            assert!(s.accepts_camera_visibility());
            assert!(s.cameras.iter().all(|cam| s.camera_curve_clear(cam)));
        }
        if seed == 0 {
            let replay: IndoorManifest =
                serde_json::from_slice(&serde_json::to_vec(&s).unwrap()).unwrap();
            assert_eq!(s, replay);
            assert_eq!(
                s,
                IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::StairHall,
                    density,
                    count,
                    0.25
                )
                .unwrap()
            );
        }
    }
    assert_eq!(runs.len(), 4);
    assert_eq!(guards.len(), 4);
    assert_eq!(structures.len(), 3);
    assert_eq!(landings, [true, true]);
    assert!(seats >= 12, "only {seats} gallery seats");
    for kind in [
        ObjectKind::Bench,
        ObjectKind::Banquette,
        ObjectKind::ExhibitionPanel,
        ObjectKind::InformationStand,
    ] {
        assert!(
            uses.contains(&(kind as usize)),
            "missing gallery use {kind:?}"
        );
    }
}

#[test]
fn stair_hall_rejects_a_missing_flight_route_or_wall_behind_the_open_deck_edge() {
    let s = IndoorManifest::generate_with_humans(12, IndoorLayout::StairHall, 0.5, 0, 0.).unwrap();
    let mut missing = s.clone();
    missing.envelope.as_mut().unwrap().mezzanine = None;
    assert!(missing.validate_public_programs().is_err());
    let mut missing = s.clone();
    missing.assembly_diagnostics.programs[0]
        .reservations
        .remove(0);
    assert!(missing.validate_public_programs().is_err());
    let mut displaced = s;
    let e = displaced.envelope.as_mut().unwrap();
    let m = e.mezzanine.as_mut().unwrap();
    let cross = 1 - m.stair_axis;
    m.deck.min[cross] += 0.05;
    m.deck.max[cross] -= 0.05;
    assert!(displaced.validate_public_programs().is_err());
}
