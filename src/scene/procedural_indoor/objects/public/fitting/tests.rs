use super::*;
use crate::scene::procedural_indoor::{
    materials::{finish_slot, registered_finish},
    objects::{audit, build_object},
};
use std::collections::BTreeSet;

/// Shoot straight through the actual triangles, halfway up the textile.
fn cloth_at(g: &Geometry, x: f32, y: f32) -> bool {
    let point = Vec2::new(x, y);
    g.indices.as_chunks::<3>().0.iter().any(|t| {
        let p = |i: usize| Vec3::from_array(g.positions[t[i] as usize]).truncate();
        let (a, b, c) = (p(0), p(1), p(2));
        let d = (b - a).perp_dot(c - a);
        if d <= 1e-8 {
            return false;
        }
        let u = (point - a).perp_dot(c - a) / d;
        let v = (b - a).perp_dot(point - a) / d;
        u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001
    })
}

#[test]
fn fitting_room_apertures_match_visible_cloth_and_metric_receipts() {
    let mut stock = BTreeSet::new();
    let mut shape = BTreeSet::new();
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::FittingRoom, seed);
        let spec = fitting_room(&o).unwrap();
        let a = build_object(&o);
        let mesh = audit::check(&o, &a).unwrap();
        assert!(mesh.min_y_m.abs() < 1e-6 && mesh.max_envelope_overrun_m < 0.001);
        assert!(mesh.batches <= 7 && mesh.triangles < 13_000, "{mesh:?}");
        let label = format!("curtain#finish{}", spec.dye_slot);
        let cloth = &a.parts[&(spec.curtain_stock, label.clone())];
        assert_eq!(finish_slot(spec.curtain_stock, &label), Some(spec.dye_slot));
        assert!(registered_finish(spec.curtain_stock, spec.dye_slot));
        for semantic in ["retail_fixture", "bench", "mirror"] {
            assert!(a.parts.keys().any(|(_, key)| part_label(key) == semantic));
        }
        stock.insert((spec.curtain_stock, spec.dye_slot, spec.enclosure_finish));
        for b in &spec.bays {
            let w = b.right - b.left;
            let opening = b.clear_opening / w;
            shape.insert((spec.bays.len(), b.gathered_left, opening > 0.65));
            assert!(b.mirror_bottom + b.mirror_height < spec.rail_height);
            assert!(w - b.bench_depth > 0.75);
            let (closed, clear) = if b.gathered_left {
                (
                    b.left + (w - b.clear_opening) * 0.42,
                    b.right - b.clear_opening * 0.42,
                )
            } else {
                (
                    b.right - (w - b.clear_opening) * 0.42,
                    b.left + b.clear_opening * 0.42,
                )
            };
            assert!(cloth_at(cloth, closed, 1.21));
            assert!(!cloth_at(cloth, clear, 1.21));
            assert!(!cloth_at(cloth, closed, spec.hem_height * 0.5));
        }
        if seed < 4 {
            let persisted: IndoorObject =
                serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
            assert_eq!(spec, fitting_room(&persisted).unwrap());
            for (key, g) in build_object(&persisted).parts {
                assert_eq!(g.positions, a.parts[&key].positions);
                assert_eq!(g.indices, a.parts[&key].indices);
                assert_eq!(g.uvs, a.parts[&key].uvs);
            }
        }
    }
    assert_eq!(shape.len(), 8);
    assert!(stock.len() > 12);
}
