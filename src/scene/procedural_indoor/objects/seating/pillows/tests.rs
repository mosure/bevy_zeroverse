use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, geometry::Geometry, objects::audit};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn sewn_panels_close_at_every_edge_and_finish_assignment_preserves_shape() {
    let mut styles = BTreeSet::new();
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::Sofa, seed);
        let p = parameters(&o);
        let first = Stock::sample(&o, &p, 0);
        let second = Stock::sample(&o, &p, 1);
        assert_ne!(first.face_origin_m, second.face_origin_m);
        assert_ne!(first.face_origin_m, first.reverse_origin_m);
        let mut rng = stream(seed, 817);
        let cut = cut::Cut::sample(Vec3::new(0.43, 0.12, 0.32), &mut rng);
        styles.insert(cut.edge as usize);
        let form = cut::Form::new(cut);
        let mut a = Assembly::default();
        form.sew(&mut a, Stock::sample(&o, &p, 0), Transform::IDENTITY);
        let mut edges = BTreeMap::new();
        for ((s, label), g) in &a.parts {
            assert!(label.starts_with("pillow#finish") && label.ends_with("#uv_local"));
            assert!(matches!(
                s,
                Surface::Fabric | Surface::FabricAlt | Surface::Leather
            ));
            for t in g.indices.as_chunks::<3>().0 {
                let points = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                let n = (points[1] - points[0]).cross(points[2] - points[0]);
                let uv = t.map(|i| Vec2::from_array(g.uvs[i as usize]));
                assert!(
                    (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs() > 1e-12,
                    "{seed} {:?} {uv:?}",
                    cut.edge
                );
                assert!(
                    n.dot(
                        t.iter()
                            .map(|&i| Vec3::from_array(g.normals[i as usize]))
                            .sum()
                    ) > 0.
                );
                for j in 0..3 {
                    let mut edge = [
                        points[j].to_array().map(f32::to_bits),
                        points[(j + 1) % 3].to_array().map(f32::to_bits),
                    ];
                    edge.sort_unstable();
                    *edges.entry(edge).or_insert(0) += 1;
                }
            }
        }
        assert!(edges.values().all(|&n| n == 2));
        let widths = module_widths_for(&o, &p);
        let geometry = |o: &IndoorObject| {
            metadata(o, &p, &widths)
                .iter()
                .map(|t| (t.center_m, t.rotation, t.cut.size, t.cut.edge))
                .collect::<Vec<_>>()
        };
        let original = geometry(&o);
        o.finish_seed = Some(seed + 100);
        assert_eq!(geometry(&o), original);
    }
    assert_eq!(styles.len(), 4);
}

fn actual_support(g: &Geometry, point: Vec3, axis: Vec3) -> Option<f32> {
    // Rays start outside a furniture-sized fixture on either the seat or back side.
    let origin = point - axis * 3.;
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter_map(|t| {
            ray_triangle(
                origin,
                axis,
                t.map(|i| Vec3::from_array(g.positions[i as usize])),
            )
        })
        .min_by(f32::total_cmp)
        .map(|t| t - 3.)
}

#[test]
fn installed_throws_bear_on_the_rendered_seats_and_backs() {
    let (mut leaning, mut flat) = (0, 0);
    let mut max_seat_gap = 0_f32;
    let mut min_seat_gap = 0_f32;
    let mut max_back_gap = 0_f32;
    let mut min_back_gap = 0_f32;
    for seed in 0..40 {
        let o = audit::fixture(ObjectKind::Sofa, seed);
        let p = parameters(&o);
        if !p.pillows {
            continue;
        }
        let widths = module_widths_for(&o, &p);
        let mut seat = Assembly::default();
        let mut back = Assembly::default();
        let mut start = -o.size.x * 0.5 + p.arm_width;
        for (i, w) in widths.iter().enumerate() {
            let depth = module_depth(&o, &p, i);
            let (x, width) = if p.seat_style == 1 {
                (0., o.size.x - 2. * p.arm_width)
            } else {
                (start + w * 0.5, *w)
            };
            upholstery::seat(
                &mut seat,
                &o,
                &p,
                Vec3::new(
                    x,
                    p.seat_height - p.cushion * 0.5,
                    (o.size.z - depth) * 0.5 - 0.085,
                ),
                Vec3::new(width - p.cushion_gap_m, p.cushion, depth - 0.19),
            );
            if p.seat_style == 1 {
                break;
            }
            start += w;
        }
        let width = (o.size.x - 2. * p.arm_width) / p.back_modules as f32;
        for i in 0..p.back_modules {
            upholstery::back(
                &mut back,
                &o,
                &p,
                (i as f32 - (p.back_modules - 1) as f32 * 0.5) * width,
                width,
                p.seat_height - p.cushion,
            );
        }
        placement::install(&o, &p, &widths, |item, form| {
            let mut gaps = [f32::INFINITY; 2];
            let mut bounds = Vec::new();
            for &pt in &form.points {
                let v = item.rotation * pt + item.center_m;
                bounds.push(v);
            }
            // Include triangle interiors, so the oracle can detect two crossing
            // surfaces even when every vertex lands clear of a narrow pad.
            form.triangles(|t| {
                bounds.push(
                    (t.map(|i| item.rotation * form.points[i] + item.center_m)
                        .into_iter()
                        .sum::<Vec3>())
                        / 3.,
                )
            });
            for v in bounds {
                for (i, (support, axis)) in [(&seat, Vec3::NEG_Y), (&back, Vec3::Z)]
                    .into_iter()
                    .enumerate()
                {
                    if (i == 0 && v.y > p.seat_height + 0.04)
                        || (i == 1 && v.z < o.size.z * 0.5 - 0.34)
                    {
                        continue;
                    }
                    for g in support.parts.values() {
                        if let Some(d) = actual_support(g, v, axis) {
                            gaps[i] = gaps[i].min(d);
                        }
                    }
                }
            }
            let [s, b] = gaps;
            max_seat_gap = max_seat_gap.max(s);
            min_seat_gap = min_seat_gap.min(s);
            assert!((-0.008..=0.004).contains(&s), "seat {seed} {s}");
            if b.is_finite() {
                min_back_gap = min_back_gap.min(b);
                assert!(b > -0.012, "back {seed} {b}");
            }
            if item.propped {
                max_back_gap = max_back_gap.max(b);
                leaning += 1;
                assert!(b < 0.006, "propped {seed} {b}");
            } else {
                flat += 1;
            }
        });
    }
    assert!(leaning >= 4 && flat >= 4, "{leaning} leaning, {flat} flat");
    println!("Independent seat gaps [{min_seat_gap}, {max_seat_gap}] m; back [{min_back_gap}, {max_back_gap}] m across {leaning} supported lean and {flat} resting throws");
}
