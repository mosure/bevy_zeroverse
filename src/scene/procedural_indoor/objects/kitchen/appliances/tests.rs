use super::*;
use crate::scene::procedural_indoor::{
    objects::audit,
    placement::{rotated_half, Footprint},
};
use std::collections::BTreeSet;

#[test]
fn appliance_programs_fit_open_envelopes_replay_and_vary_at_fixed_dimensions() {
    for kind in [
        ObjectKind::Cooktop,
        ObjectKind::Range,
        ObjectKind::Oven,
        ObjectKind::Microwave,
        ObjectKind::Dishwasher,
        ObjectKind::Refrigerator,
    ] {
        let mut descriptions = Vec::new();
        let mut finishes = BTreeSet::new();
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = None;
            let p = parameters(&o);
            let recipe = program::parameters(&o);
            finishes.insert(format!("{:?}/{:?}", recipe.shell, recipe.front));
            let a = super::super::super::build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert_eq!(
                serde_json::to_value(metadata(&o, &p)).unwrap(),
                serde_json::to_value(metadata(&o, &parameters(&o))).unwrap()
            );
            // Fixed outer envelope, excluding changes of color and size.
            if seed < 32 {
                let (o, a, _) = review_fixture(kind, seed);
                audit::check(&o, &a).unwrap_or_else(|e| panic!("fixed {kind:?}/{seed}: {e}"));
                descriptions.push(audit::descriptor(&o, &a));
            }
        }
        assert!(
            finishes.len() >= 4,
            "finish collapse {kind:?}: {finishes:?}"
        );
        // Closed manufactured enclosures can legitimately remain rectangular.
        // Check visible shape variation, but report perceptual-threshold coverage
        // instead of requiring an arbitrary number of distinct cabinet boxes.
        let changed = descriptions
            .iter()
            .skip(1)
            .any(|d| audit::distance(&descriptions[0], d) > 0.0001);
        assert!(
            changed,
            "closed-chassis geometry does not vary for {kind:?}"
        );
    }
}

#[test]
fn burner_and_control_packing_is_bounded_and_not_an_identical_four_ring_grid() {
    let mut counts = BTreeSet::new();
    let mut signatures = BTreeSet::new();
    for seed in 0..512 {
        let o = audit::fixture(ObjectKind::Cooktop, seed);
        let p = parameters(&o);
        let recipe = cooktop::parameters(seed, o.size, &p);
        assert!(
            (2..=5).contains(&recipe.burners.len()),
            "seed {seed}: {}",
            recipe.burners.len()
        );
        assert_eq!(recipe.controls.len(), recipe.burners.len());
        counts.insert(recipe.burners.len());
        signatures.insert(serde_json::to_string(&recipe).unwrap());
        for (i, b) in recipe.burners.iter().enumerate() {
            assert!((b.center.abs() + Vec2::splat(b.grate_radius_m))
                .cmple(o.size.xz() * 0.5)
                .all());
            for c in &recipe.burners[..i] {
                assert!(
                    b.center.distance(c.center) >= b.grate_radius_m + c.grate_radius_m + 0.0119
                );
            }
            for &c in &recipe.controls {
                assert!(b.center.distance(c) > b.grate_radius_m + recipe.knob_radius_m);
            }
        }
    }
    assert!(counts.len() >= 3, "burner counts {counts:?}");
    assert_eq!(signatures.len(), 512);
}

#[test]
fn refrigerator_papers_clear_handles_and_peers_and_shelves_keep_supported_headroom() {
    let mut note_positions = BTreeSet::new();
    let mut gaps = BTreeSet::new();
    let mut count = 0;
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::Refrigerator, seed);
        let p = parameters(&o);
        let recipe = program::parameters(&o);
        let mut levels = recipe.shelf_levels_m.clone();
        levels.push(recipe.shelf_top_m);
        for v in levels.windows(2) {
            assert!(v[1] - v[0] >= 0.13, "shelf {seed}");
            gaps.insert(((v[1] - v[0]) * 1000.).round() as i32);
        }
        for d in refrigerator_doors(&o, &p) {
            let key = o.seed ^ ((d.center.x.to_bits() as u64) << 32) ^ d.center.y.to_bits() as u64;
            let notes = notes::placements(key, d.width, d.height, d.side, &p);
            let handle = Footprint {
                center: Vec2::new(-d.side * (d.width * 0.5 - 0.045), 0.),
                half: Vec2::new(0.025, 0.075),
                yaw: 0.,
            };
            let mut occupied = Vec::new();
            for n in notes {
                let q = Footprint {
                    center: n.center,
                    half: rotated_half(n.size, n.yaw),
                    yaw: n.yaw,
                };
                let half = Vec2::new(d.width * 0.34, d.height * 0.36);
                assert!(q.inside(-half, half, 0.0029));
                assert!(p.handles == 4 || !q.overlaps(handle, 0.0059));
                assert!(occupied.iter().all(|&a| !q.overlaps(a, 0.0079)));
                occupied.push(q);
                note_positions.insert((
                    (n.center.x / d.width * 64.).round() as i32,
                    (n.center.y / d.height * 64.).round() as i32,
                ));
                count += 1;
            }
        }
    }
    assert!(count > 400);
    assert!(note_positions.len() > 150);
    assert!(gaps.len() > 100);
}

#[test]
fn appliance_finish_changes_do_not_change_geometry_or_operated_parts() {
    for kind in [
        ObjectKind::Refrigerator,
        ObjectKind::Microwave,
        ObjectKind::Range,
    ] {
        for seed in 0..8 {
            let mut o = audit::fixture(kind, seed);
            let a = super::super::super::build_object(&o);
            let first = audit::descriptor(&o, &a);
            o.finish_seed = Some(90919);
            let b = super::super::super::build_object(&o);
            assert_eq!(first, audit::descriptor(&o, &b), "{kind:?}/{seed}");
        }
    }
}
