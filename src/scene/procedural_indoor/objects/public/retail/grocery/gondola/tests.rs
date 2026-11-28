use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeMap;

fn rack(o: &IndoorObject) -> Gondola {
    let Installation::Gondola {
        levels,
        facings,
        stock,
        ..
    } = parameters(o)
    else {
        panic!("dry gondola fixture")
    };
    Gondola::sample(o, levels, facings, stock)
}

#[test]
fn dry_stock_fits_the_real_decks_including_both_faces_and_rear_copies() {
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        o.variant &= !REFRIGERATED;
        for size in [Vec3::new(1.0, 1.15, 0.64), Vec3::new(2.68, 1.94, 0.93)] {
            o.size = size;
            let rack = rack(&o);
            let mut boxes: Vec<(usize, Vec2, Vec2)> = Vec::new();
            let mut seen = [false; 4];
            rack.place(|p| {
                let size = rack.products[p.sku].1;
                let deck = rack.decks[p.row / 2];
                let (sn, cs) = p.yaw.sin_cos();
                let half = 0.5 * (size.xz() * cs.abs() + Vec2::new(size.z, size.x) * sn.abs());
                let lo = p.base.xz() - half;
                let hi = p.base.xz() + half;
                assert!(lo.x >= -rack.width * 0.5 && hi.x <= rack.width * 0.5);
                assert!(p.base.z.abs() - half.y > rack.spine * 0.5);
                assert!(p.base.z.abs() + half.y < rack.spine * 0.5 + deck.projection);
                assert_eq!(p.base.y, deck.top);
                assert!(size.y < deck.clearance && p.base.y + size.y <= o.size.y);
                assert!(
                    boxes.iter().all(|&(level, l, h)| level != p.row / 2
                        || !lo.cmplt(h).all()
                        || !hi.cmpgt(l).all()),
                    "two packs intersect at seed {seed}, row {}",
                    p.row
                );
                seen[p.sku] = true;
                boxes.push((p.row / 2, lo, hi));
            });
            assert!(seen.into_iter().all(|v| v));
            assert!(boxes.len() <= 4 * rack.decks.len() * rack.facings);
        }
    }
}

#[test]
fn edge_slots_can_sell_out_and_do_not_predict_the_product_lot() {
    let mut occupied = [0u32; 3];
    let mut totals = [0u32; 3];
    let mut first_lots = [0usize; 4];
    let mut seen_forms = std::collections::BTreeSet::new();
    let mut has_rear = false;
    let mut has_paired = false;
    let mut has_recess = false;
    for seed in 0..224 {
        let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        o.variant &= !REFRIGERATED;
        o.size.y = o.size.y.max(1.3);
        assert!(gondola_stock(&o, VERSION - 1).is_none());
        let spec = gondola_stock(&o, VERSION).unwrap();
        let fixture = rack(&o);
        fixture.place(|p| {
            let front = fixture.spine * 0.5 + fixture.decks[p.row / 2].projection;
            has_recess |= p.facing == stock::Facing::Front
                && front - p.base.z.abs() - fixture.products[p.sku].1.z * 0.5 > 0.10;
        });
        seen_forms.extend(spec.package_forms);
        for row in &spec.rows {
            for (i, slot) in row.slots.iter().enumerate() {
                let region = if i == 0 {
                    0
                } else if i + 1 == row.slots.len() {
                    2
                } else {
                    1
                };
                totals[region] += 1;
                occupied[region] += u32::from(slot.is_some());
            }
            has_rear |= row.rear_copies > 0;
            has_paired |= row.side_by_side_copies > 0;
            assert!(
                row.rear_copies + row.side_by_side_copies <= row.slots.iter().flatten().count()
            );
        }
        if let Some(sku) = spec.rows[0].slots[0] {
            first_lots[sku] += 1;
        }
        o.position = Vec3::new(2., 3., 4.);
        o.yaw = 0.7;
        assert_eq!(spec, gondola_stock(&o, VERSION).unwrap());
        o.variant |= REFRIGERATED;
        assert!(gondola_stock(&o, VERSION).is_none());
    }
    let fill: [f32; 3] = std::array::from_fn(|i| occupied[i] as f32 / totals[i] as f32);
    assert!(fill.iter().all(|v| (0.50..0.93).contains(v)));
    assert!(
        (fill[0] - fill[1]).abs() < 0.07 && (fill[2] - fill[1]).abs() < 0.07,
        "{fill:?}"
    );
    assert!(first_lots.into_iter().all(|n| n >= 12));
    assert!(has_rear && has_paired && has_recess);
    assert_eq!(seen_forms.len(), 4);
}

#[test]
fn archived_dry_shelves_replay_and_other_grocery_fixtures_keep_their_meshes() {
    use crate::scene::procedural_indoor::objects::build_object_at_version;
    let mut controls = [false; 2];
    for seed in 0..12 {
        let o = audit::fixture(ObjectKind::GroceryDisplay, seed);
        let old = build_object_at_version(&o, None, VERSION - 1);
        if let Installation::Gondola { .. } = parameters(&o) {
            if seed == 0 {
                // Triangle and batch counts measured on the original generator.
                let stats = audit::check(&o, &old).unwrap();
                assert_eq!((stats.triangles, stats.batches), (12_336, 7));
            }
            continue;
        }
        controls[usize::from(is_refrigerated(&o))] = true;
        let current = build_object_at_version(&o, None, VERSION);
        assert_eq!(old.parts.len(), current.parts.len());
        for (key, g) in &old.parts {
            let other = &current.parts[key];
            assert_eq!(g.positions, other.positions);
            assert_eq!(g.normals, other.normals);
            assert_eq!(g.uvs, other.uvs);
            assert_eq!(g.indices, other.indices);
        }
    }
    assert!(controls.into_iter().all(|v| v));
}

#[test]
fn printed_gusset_bag_is_one_closed_surface_with_consistent_triangle_winding() {
    let mut a = Assembly::default();
    let size = Vec3::new(0.139, 0.218, 0.078);
    Pack::PaperBag.build(&mut a, Vec3::ZERO, size, 0., "food#finish1#uv_local");
    let mut edges = BTreeMap::new();
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for ((surface, _), g) in &a.parts {
        assert_eq!(*surface, Surface::Paper);
        assert!(g.uvs.iter().flatten().all(|v| v.is_finite()));
        for tri in g.indices.as_chunks::<3>().0 {
            let v: [Vec3; 3] =
                std::array::from_fn(|i| Vec3::from_array(g.positions[tri[i] as usize]));
            let normal = (v[1] - v[0]).cross(v[2] - v[0]);
            for (i, p) in v.into_iter().enumerate() {
                assert!(normal.dot(Vec3::from_array(g.normals[tri[i] as usize])) > 0.);
                low = low.min(p);
                high = high.max(p);
                let key = |p: Vec3| p.to_array().map(|x| (x * 1e6).round() as i32);
                let (a, b) = (key(p), key(v[(i + 1) % 3]));
                let dir = if a < b { 1 } else { -1 };
                let (count, balance) = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                *count += 1;
                *balance += dir;
            }
        }
    }
    assert!(edges.values().all(|&v| v == (2, 0)));
    assert_eq!(low.y, 0.);
    assert!((high - low - size).abs().max_element() < 1e-6);
}
