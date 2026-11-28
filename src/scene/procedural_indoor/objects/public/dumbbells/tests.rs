use super::*;
use crate::scene::procedural_indoor::{
    geometry::RevolvedOutline,
    objects::{audit, build_object, part_label},
};
use std::collections::BTreeSet;

#[test]
fn graduate_pairs_inside_the_installed_rack_and_keep_an_approach_to_each_tier() {
    let mut forms = BTreeSet::new();
    let mut heads = BTreeSet::new();
    let mut frames = BTreeSet::new();
    let mut levels = BTreeSet::new();
    let mut crosses = 0;
    let mut empty = 0;
    let mut unpaired = 0;
    let mut full_sets = 0;
    let mut unequal_tiers = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::DumbbellRack, seed);
        // Include the compact bay used for recovery in tightly furnished rooms.
        if seed % 3 == 0 {
            o.size.x = 1.05;
            o.size.z = 0.48;
        }
        let rack = FreeWeightRack::for_object(&o);
        forms.insert(format!("{:?}", rack.shelf));
        frames.insert(rack.cantilever);
        heads.insert(rack.head);
        levels.insert(rack.tiers.len());
        let mut full_pairs = 0;
        let mut installed_pairs = 0;
        let mut prev_top = 0.;
        for tier in &rack.tiers {
            assert!(
                tier.support_y - prev_top >= 0.117,
                "seed {seed}, tier {}",
                tier.support_y
            );
            let mut right = -o.size.x * 0.5 + 0.058;
            for p in &tier.pairs {
                installed_pairs += 1;
                assert!((2..=64).contains(&p.nominal_kg) && p.nominal_kg.is_multiple_of(2));
                match p.slots.iter().filter(|s| s.present).count() {
                    0 => empty += 1,
                    1 => unpaired += 1,
                    _ => full_pairs += 1,
                }
                for slot in &p.slots {
                    let (s, c) = slot.yaw.sin_cos();
                    let z = rack.half_grip + p.head_width;
                    let dx = p.radius * c + z * s.abs();
                    let dz = z * c + p.radius * s.abs();
                    assert!(slot.x - dx > right && slot.x + dx < o.size.x * 0.5 - 0.058);
                    right = slot.x + dx;
                    assert!((tier.center_z + slot.depth_offset).abs() + dz < o.size.z * 0.5);
                    if rack.shelf == Shelf::LinedTray {
                        assert!(
                            slot.depth_offset.abs() + dz
                                < FreeWeightRack::tray_depth(rack.half_grip) - 0.007
                        );
                    }
                }
            }
            crosses += usize::from(tier.pairs[0].radius > tier.pairs.last().unwrap().radius);
            prev_top = tier.support_y
                + tier
                    .pairs
                    .iter()
                    .map(|p| p.radius * (rack.contact + rack.head.contact(0.)))
                    .fold(0., f32::max);
        }
        assert!(prev_top <= o.size.y && full_pairs > 0);
        full_sets += usize::from(full_pairs == installed_pairs);
        unequal_tiers += usize::from(
            rack.tiers
                .windows(2)
                .any(|t| t[0].pairs.len() != t[1].pairs.len()),
        );
        if seed % 16 == 0 {
            let assembly = build_object(&o);
            let mesh = audit::check(&o, &assembly).unwrap();
            assert!(mesh.max_envelope_overrun_m < 0.0002 && mesh.batches <= 8);
            assert!(assembly
                .parts
                .keys()
                .all(|(_, label)| part_label(label) == o.kind.class_name()));
        }
    }
    assert_eq!(
        (forms.len(), heads.len(), frames.len(), levels.len()),
        (3, 3, 2, 2)
    );
    assert!(empty > 30 && unpaired > 30 && full_sets > 30 && crosses > 100 && unequal_tiers > 20);
}

// Double-sided vertical intersection of actual frame/weight triangles. A
// probe can lie on a polygon edge shared by two faces.
fn heights(a: &Assembly, point: Vec2) -> impl Iterator<Item = f32> + '_ {
    a.parts.values().flat_map(move |g| {
        g.indices.as_chunks::<3>().0.iter().filter_map(move |ix| {
            let [v, b, c] = ix.map(|i| Vec3::from_array(g.positions[i as usize]));
            let e = b - v;
            let f = c - v;
            let d = e.xz().perp_dot(f.xz());
            if d.abs() < 1e-9 {
                return None;
            }
            let delta = point - v.xz();
            let u = delta.perp_dot(f.xz()) / d;
            let w = e.xz().perp_dot(delta) / d;
            (u >= -1e-5 && w >= -1e-5 && u + w <= 1.00001).then_some(v.y + u * e.y + w * f.y)
        })
    })
}

#[test]
fn both_heads_touch_the_actual_support_triangles_and_empty_saddles_stay_installed() {
    let mut sampled = BTreeSet::new();
    let mut empty_saddles = 0usize;
    for seed in 0..40 {
        let o = audit::fixture(ObjectKind::DumbbellRack, seed);
        let rack = FreeWeightRack::for_object(&o);
        let mut frame = Assembly::default();
        rack.frame(&mut frame, o.size);
        let n = rack.head.sides();
        let outlined = if rack.head == Head::Round {
            RevolvedOutline::round(n, 0., 0)
        } else {
            RevolvedOutline::polygon(n)
        };
        for tier in &rack.tiers {
            let Some(pair) = tier
                .pairs
                .iter()
                .find(|p| p.slots.iter().any(|s| s.present))
            else {
                continue;
            };
            let slot = *pair.slots.iter().find(|s| s.present).unwrap();
            let mut weight = Assembly::default();
            rack.weights(&mut weight, &outlined, tier, pair);
            let yaw = Quat::from_rotation_y(slot.yaw);
            for side in [-1., 1.] {
                let head_z = side * (rack.half_grip + pair.head_width * 0.5);
                let axis = Vec3::new(slot.x, 0., tier.center_z + slot.depth_offset);
                // Rails/trays contact a flat. For the saddle find the vertex
                // tangent to each actual V flank independently of rack.contact.
                let dx = if rack.shelf == Shelf::Saddles {
                    (0..n)
                        .map(|i| {
                            let (y, x) = (i as f32 * TAU / n as f32).sin_cos();
                            (y + rack.saddle_slope * x, x)
                        })
                        .max_by(|a, b| a.0.total_cmp(&b.0))
                        .unwrap()
                        .1
                        * pair.radius
                } else {
                    0.
                };
                for probe_x in [-dx, dx] {
                    let xz = (axis + yaw * Vec3::new(probe_x, 0., head_z)).xz();
                    let bottom = heights(&weight, xz).min_by(f32::total_cmp).unwrap();
                    let rest = heights(&frame, xz)
                        .filter(|y| *y < tier.support_y + 0.75 * pair.radius)
                        .max_by(f32::total_cmp)
                        .unwrap();
                    assert!(
                        (bottom - rest).abs() < 0.00015,
                        "{seed}, {:?}: weight {}, support {}",
                        rack.shelf,
                        bottom,
                        rest
                    );
                }
            }
            if rack.shelf == Shelf::Saddles {
                for p in &tier.pairs {
                    for s in p.slots.iter().filter(|s| !s.present) {
                        for side in [-1., 1.] {
                            let dx = 0.55 * p.radius;
                            let probe =
                                Vec2::new(s.x + dx, tier.center_z + side * rack.rail_offset());
                            let height = tier.support_y + rack.saddle_slope * dx;
                            assert!(heights(&frame, probe).any(|y| (y - height).abs() < 0.0001));
                        }
                        empty_saddles += 1;
                    }
                }
            }
        }
        sampled.insert(rack.head);
    }
    assert_eq!(sampled.len(), 3);
    assert!(empty_saddles > 5);
}

#[test]
fn archive_replays_the_original_mesh_and_finish_seed_does_not_move_stock() {
    for seed in 0..12 {
        let mut o = audit::fixture(ObjectKind::DumbbellRack, seed);
        let mut old = Assembly::default();
        let mut replay = Assembly::default();
        super::super::fitness::build(&mut old, &o);
        super::super::build_at_version(&mut replay, &o, VERSION - 1);
        assert!(old.parts.keys().eq(replay.parts.keys()));
        for (key, g) in old.parts {
            let r = &replay.parts[&key];
            assert_eq!(
                (&g.positions, &g.normals, &g.uvs, &g.indices),
                (&r.positions, &r.normals, &r.uvs, &r.indices)
            );
        }
        let original = FreeWeightRack::for_object(&o);
        o.finish_seed = Some(215 + seed);
        let refinished = FreeWeightRack::for_object(&o);
        assert_eq!(
            serde_json::to_value(original.tiers).unwrap(),
            serde_json::to_value(refinished.tiers).unwrap()
        );
        assert_eq!(
            (original.shelf, original.head, original.cantilever),
            (refinished.shelf, refinished.head, refinished.cantilever)
        );
    }
}

#[test]
fn end_marks_read_outward_on_both_heads_and_stay_in_their_pairs_atlas_cell() {
    use crate::scene::procedural_indoor::materials::strength::end_marks;
    let o = audit::fixture(ObjectKind::DumbbellRack, 7);
    let mut rack = FreeWeightRack::for_object(&o);
    rack.head_finish = HeadFinish::ColourCodedEnds;
    let tier = &rack.tiers[0];
    let mut pair = tier.pairs[0].clone();
    pair.slots[0].present = true;
    pair.slots[0].face_turn = 0.;
    pair.slots[1].present = false;
    let slot = pair.slots[0];
    let mut a = Assembly::default();
    let outline = if rack.head == Head::Round {
        RevolvedOutline::round(rack.head.sides(), 0., 0)
    } else {
        RevolvedOutline::polygon(rack.head.sides())
    };
    rack.weights(&mut a, &outline, tier, &pair);
    let g = &a.parts[&(end_marks::KEY.0, end_marks::KEY.1.to_owned())];
    let center = Vec3::new(
        slot.x,
        tier.support_y + pair.radius * rack.contact,
        tier.center_z + slot.depth_offset,
    );
    let undo = Quat::from_rotation_y(-slot.yaw);
    let mut ends = [0; 2];
    for ((pt, n), uv) in g.positions.iter().zip(&g.normals).zip(&g.uvs) {
        let p = undo * (Vec3::from_array(*pt) - center);
        let n = undo * Vec3::from_array(*n);
        assert!(n.z * p.z.signum() > 0.999);
        ends[usize::from(p.z > 0.)] += 1;
        let unit = Vec2::new(p.x * p.z.signum(), -p.y) / (0.62 * pair.radius);
        let expected = end_marks::uv(pair.nominal_kg, unit);
        assert!(Vec2::from_array(*uv).abs_diff_eq(Vec2::from_array(expected), 0.00001));
    }
    assert!(ends[0] > 0 && ends[0] == ends[1]);
}
