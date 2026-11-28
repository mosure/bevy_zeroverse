use super::*;
use crate::scene::procedural_indoor::{
    envelope::ray_triangle,
    objects::{audit, build_object},
};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn practice_prop_meshes_have_closed_seams_metric_charts_and_floor_contact() {
    let mut finishes = BTreeSet::new();
    let mut forms = BTreeSet::new();
    let mut laminates = BTreeSet::new();
    for kind in [ObjectKind::YogaBlock, ObjectKind::YogaBolster] {
        for seed in 0..96 {
            let o = audit::fixture(kind, seed);
            let a = build_object(&o);
            let check = audit::check(&o, &a).unwrap();
            assert!(
                check.triangles < 530
                    && check.batches <= 2
                    && check.min_y_m.abs() < 0.00001
                    && check.max_envelope_overrun_m < 0.00001,
                "{kind:?}/{seed}: {check:?}"
            );
            let mut edges = BTreeMap::<_, (u8, i8)>::new();
            for ((_, label), mesh) in &a.parts {
                assert_eq!(part_label(label), "fitness_equipment");
                for tri in mesh.indices.as_chunks::<3>().0 {
                    let uv = tri.map(|i| Vec2::from(mesh.uvs[i as usize]));
                    assert!((uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-10);
                    let points =
                        tri.map(|i| mesh.positions[i as usize].map(|x| (x * 1e5).round() as i32));
                    for [p, q] in [
                        [points[0], points[1]],
                        [points[1], points[2]],
                        [points[2], points[0]],
                    ] {
                        let edge = edges.entry((p.min(q), p.max(q))).or_default();
                        edge.0 += 1;
                        edge.1 += if p < q { 1 } else { -1 };
                    }
                }
            }
            assert!(edges.values().all(|&v| v == (2, 0)), "{kind:?}/{seed}");
            if kind == ObjectKind::YogaBlock {
                let b = Block::for_object(&o);
                finishes.insert((b.facing.key().1, b.inlay.is_some()));
                if let Some(stock) = b.inlay {
                    let (surface, class) = stock.equipment_key();
                    let stripe = &a.parts[&(surface, class.to_owned())];
                    for p in &stripe.positions {
                        let (along, thickness) = if b.on_edge {
                            (p[2], o.size.z)
                        } else {
                            (p[1] - o.size.y * 0.5, o.size.y)
                        };
                        assert!((along.abs() - thickness * b.stripe_fraction * 0.5).abs() < 1e-6);
                    }
                    laminates.insert(b.on_edge);
                }
            } else {
                let b = Bolster::for_object(&o);
                forms.insert((b.section_exponent > 3., b.handle_m > 0.));
                if b.handle_m > 0. {
                    // A ray through the grip sees daylight; a ray nearer the
                    // stitched end sees the upholstered body.
                    let end = o.size.z * 0.5 - b.handle_m;
                    for (z, hits) in [(end + b.handle_m * 0.27, false), (end - 0.012, true)] {
                        let origin = Vec3::new(0.00093, o.size.y + 0.1, z);
                        assert_eq!(
                            a.parts
                                .values()
                                .any(|g| g.indices.as_chunks::<3>().0.iter().any(|tri| {
                                    ray_triangle(
                                        origin,
                                        Vec3::NEG_Y,
                                        tri.map(|i| Vec3::from(g.positions[i as usize])),
                                    )
                                    .is_some()
                                })),
                            hits,
                            "bolster {seed} loses its grip opening"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(finishes.len(), 4);
    assert_eq!(laminates.len(), 2);
    assert_eq!(forms.len(), 4);
}
