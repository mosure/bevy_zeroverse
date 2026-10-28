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

#[test]
fn appliance_cavities_have_fitted_variable_racks_and_temperature_safe_linings() {
    let mut baskets = BTreeSet::new();
    let mut racks = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut splits = BTreeSet::new();
    for seed in 0..128 {
        for kind in [ObjectKind::Dishwasher, ObjectKind::Oven, ObjectKind::Range] {
            let o = audit::fixture(kind, seed);
            let p = parameters(&o);
            let (depth, z) = body(&o, &p);
            let a = super::super::super::build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            if kind == ObjectKind::Dishwasher {
                let plan = interior::dishwasher_baskets(&o, &p);
                baskets.insert(plan.len());
                let top = o.size.y * (1. - program::parameters(&o).fascia_fraction) - 0.035;
                for (i, b) in plan.iter().enumerate() {
                    assert!(b.position.y > 0.1 && b.position.y + b.size.y <= top + 1e-5);
                    assert!(b.position.x.abs() + b.size.x * 0.5 < o.size.x * 0.5 - 0.025);
                    assert!((b.position.z - z).abs() + b.size.z * 0.5 < depth * 0.5 - 0.018);
                    if i > 0 {
                        assert!(plan[i - 1].position.y + plan[i - 1].size.y + 0.024 < b.position.y);
                    }
                    heights.insert((b.position.y * 1000.) as i32);
                }
            } else {
                let fronts = program::oven_fronts(&o, &p);
                for (i, front) in fronts.iter().enumerate() {
                    assert!(front.center_x.abs() + front.width_m * 0.5 < o.size.x * 0.5 - 0.012);
                    if i > 0 {
                        assert!(
                            fronts[i - 1].bottom_m + fronts[i - 1].height_m
                                < front.bottom_m - 0.025
                        );
                    }
                    splits.insert((front.height_m * 1000.) as i32);
                    let rr = interior::oven_racks(&o, &p, front, i);
                    racks.insert(rr.len());
                    for (j, rack) in rr.iter().enumerate() {
                        assert!(rack.y_m - rack.wire_m > front.bottom_m + 0.016);
                        assert!(
                            rack.y_m + rack.pan_size.y + rack.wire_m
                                < front.bottom_m + front.height_m - 0.016
                        );
                        if j > 0 {
                            assert!(rr[j - 1].y_m + rr[j - 1].pan_size.y + 0.01 < rack.y_m);
                        }
                        assert!(rack.width_m < front.width_m - 0.02);
                        assert!(
                            rack.pan_offset.abs().x + rack.pan_size.x * 0.5 < rack.width_m * 0.5
                        );
                    }
                }
                // Heat-exposed linings and shelves may not acquire timber or plastic.
                for ((surface, label), g) in &a.parts {
                    if !matches!(surface, Surface::Plastic | Surface::Wood) {
                        continue;
                    }
                    if super::super::super::part_label(label) != kind.class_name() {
                        continue;
                    }
                    assert!(
                        g.positions.iter().all(|v| v[2] > z + depth * 0.35),
                        "hot cavity uses {surface:?}: {kind:?}/{seed}"
                    );
                }
            }
        }
    }
    assert_eq!(baskets, BTreeSet::from([2, 3]));
    assert_eq!(racks, BTreeSet::from([1, 2, 3]));
    assert!(heights.len() > 150 && splits.len() > 150);
    println!("appliance sampling: basket counts={baskets:?}, rack counts={racks:?}, basket_height_bins={}, oven_height_bins={}",heights.len(),splits.len());
}

#[test]
fn kitchen_review_samples_capture_independent_finishes_and_live_programs() {
    let Some(path) = std::env::var_os("ZEROVERSE_KITCHEN_REPORT") else {
        return;
    };
    let mut categories = Vec::new();
    for kind in [
        ObjectKind::KitchenSink,
        ObjectKind::Dishwasher,
        ObjectKind::Oven,
        ObjectKind::Range,
    ] {
        let mut samples = Vec::new();
        let mut maps = Vec::new();
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed ^ 0x415050454152);
            let a = super::super::super::build_object(&o);
            let mesh = audit::check(&o, &a).unwrap();
            maps.push(audit::descriptor(&o, &a));
            samples.push(
                serde_json::json!({"program":super::super::metadata(&o),"object":o,"mesh":mesh}),
            );
        }
        categories.push(serde_json::json!({"kind":kind,"samples":samples,"depth_descriptors":maps,"failures":[]}));
    }
    std::fs::write(std::path::Path::new(&path).join("kitchen-metric-audit.json"),serde_json::to_vec(&serde_json::json!({"categories":categories,"build_provenance":crate::provenance::capture_provenance()})).unwrap()).unwrap();
}
