use super::*;
use crate::{
    render::semantic::SemanticLabel,
    scene::procedural_indoor::objects::{audit, build_object, part_label},
};
use std::collections::BTreeSet;

// Intersect upward-facing triangles in the actual local mesh. This also catches
// a seat or full deck inadvertently bridging the stair aisle.
fn highest(a: &Assembly, x: f32, z: f32) -> f32 {
    let mut y = f32::NEG_INFINITY;
    for g in a.parts.values() {
        for tri in g.indices.as_chunks::<3>().0 {
            let [p, q, r] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let a = q - p;
            let b = r - p;
            if a.cross(b).y < 1e-7 {
                continue;
            }
            let v = Vec2::new(x, z) - p.xz();
            let det = a.xz().perp_dot(b.xz());
            let s = v.perp_dot(b.xz()) / det;
            let t = a.xz().perp_dot(v) / det;
            if s >= 0. && t >= 0. && s + t <= 1. {
                y = y.max(p.y + s * a.y + t * b.y);
            }
        }
    }
    y
}

#[test]
fn bleacher_mesh_has_walkable_half_steps_open_guards_and_supported_seat_rows() {
    let mut tiers = BTreeSet::new();
    let mut stocks = BTreeSet::new();
    for seed in 810..850 {
        let mut o = audit::fixture(ObjectKind::BleacherBank, seed);
        o.finish_seed = Some(seed);
        let b = BleacherBank::for_object(&o);
        let a = build_object(&o);
        let stat = audit::check(&o, &a).unwrap();
        assert!(
            stat.triangles < 6500 && stat.batches <= 4,
            "{seed} {stat:?}"
        );
        assert!(a.parts.keys().all(|(_, s)| part_label(s) == CLASS));
        assert_eq!(
            SemanticLabel::from_label(CLASS),
            Some(SemanticLabel::Bleacher)
        );
        tiers.insert(b.rows);
        stocks.insert((
            b.seat_stock == SeatStock::Timber,
            b.deck_stock == Surface::Wood,
        ));
        let spans = b.spans();
        assert!(spans.iter().all(|&(lo, hi)| b.modules(lo, hi) > 0));
        assert!((0.90..1.131).contains(&(spans[1].0 - spans[0].1)));
        for i in 0..b.rows {
            let y = b.deck_height(i);
            let front = b.front(i);
            let x = b.aisle_center + 0.047;
            let tread = front - b.pitch * 0.22;
            assert!((highest(&a, x, tread) - (y - b.rise * 0.5)).abs() < 1e-4);
            assert!((highest(&a, x, front - b.pitch * 0.72) - y).abs() < 1e-4);
            let (lo, hi) = spans[0];
            let seat_x = lo + 0.50 * (hi - lo) / b.modules(lo, hi) as f32;
            let sat = highest(&a, seat_x, b.seat_z(i));
            assert!(
                (y + b.seat_height - 0.024..=y + b.seat_height + 0.002).contains(&sat),
                "{seed} {sat}"
            );
            let mut falsely_solid = false;
            b.view_boxes(|center, half| {
                falsely_solid |= (Vec3::new(x, y, tread) - center).abs().cmplt(half).all()
                    || (Vec3::new(x, y + 0.55, tread) - center)
                        .abs()
                        .cmplt(half)
                        .all();
            });
            assert!(!falsely_solid);
        }
    }
    assert_eq!(tiers, BTreeSet::from([2, 3, 4, 5]));
    assert_eq!(stocks.len(), 4);
}

#[test]
fn bleacher_dimensions_recover_every_fitted_tier_count_at_stock_extremes() {
    for rows in 2..=5 {
        for pitch in [0.605, 0.65, 0.70] {
            let mut o = audit::fixture(ObjectKind::BleacherBank, 5);
            o.size = dimensions(6.2, rows, pitch, 0.245);
            let b = BleacherBank::for_object(&o);
            assert_eq!(b.rows, rows);
            assert!((b.pitch - pitch).abs() < 1e-6 && (b.rise - 0.245).abs() < 1e-6);
        }
    }
}
