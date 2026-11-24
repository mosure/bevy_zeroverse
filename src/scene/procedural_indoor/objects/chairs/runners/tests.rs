use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeSet;

fn ordinates(g: &Geometry, x: f32, z: f32) -> Vec<f32> {
    super::super::tests::vertical_intersections([g], Vec2::new(x, z))
}

#[test]
fn glides_contact_both_runner_and_floor_and_bends_reach_the_seat() {
    let mut combinations = BTreeSet::new();
    for seed in 0..240 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 2 + (seed % 3) as u32;
        let p = parameters(&o);
        let Some(stock) = p.runner_stock(&o, INTRODUCED) else {
            continue;
        };
        combinations.insert((p.base_style, stock.section as u8, p.formed_seat.is_some()));
        let width = (o.size.x * p.seat_width_fraction).min(0.56);
        for depth in [0.29, 0.49, 0.55] {
            let mounting = if p.formed_seat.is_some() {
                formed::carrier_underside(p.base_style)
            } else {
                0.387
            };
            let side = stock.fit(p.base_style, depth, mounting);
            let mut a = Assembly::default();
            stock.build(&mut a, &p, width, depth, mounting, o.seed);
            let rail = &a.parts[&(stock.rail_surface, "chair".into())];
            let pads = &a.parts[&(stock.glide_surface, "chair".into())];
            for sign in [-1., 1.] {
                let x = sign * stock.side_span_fraction * width + 0.0003;
                for center in side.glides_z {
                    for dz in [-0.29, 0., 0.29] {
                        let z = center + dz * side.glide_length;
                        let pad = ordinates(pads, x, z);
                        let frame = ordinates(rail, x, z);
                        assert_eq!(pad.len(), 2);
                        assert!(frame.len() >= 2);
                        assert!(
                            pad[0].abs() < 1e-6 && pad[1] >= frame[0] && pad[1] < frame[1],
                            "detached glide {seed}/{depth}: {pad:?} {frame:?}"
                        );
                    }
                }
                let anchors = if p.base_style == 6 {
                    [-0.25 * depth, 0.25 * depth]
                } else {
                    [side.path[0].z, side.path[PATH_NODES - 1].z]
                };
                for z in anchors {
                    let mut hits = ordinates(rail, x, z);
                    if stock.section == RunnerSection::BentPlywood {
                        hits.extend(ordinates(
                            &a.parts[&(Surface::WoodEdge, "chair".into())],
                            x,
                            z,
                        ));
                    }
                    let top = hits.into_iter().reduce(f32::max).unwrap();
                    assert!(
                        (mounting..mounting + 0.009).contains(&top),
                        "rail misses bearing {seed}/{depth} z={z} top={top} mounting={mounting}"
                    );
                }
            }
            for ((_, label), g) in &a.parts {
                assert_eq!(label, "chair");
                for t in g.indices.as_chunks::<3>().0 {
                    let idx = [t[0], t[1], t[2]].map(|i| i as usize);
                    let [a, b, c] = idx.map(|i| Vec3::from(g.positions[i]));
                    let n: Vec3 = idx.iter().map(|&i| Vec3::from(g.normals[i])).sum();
                    assert!(
                        (b - a).cross(c - a).dot(n) > 0.,
                        "inverted or collapsed frame {seed} {depth}"
                    );
                    assert!([a, b, c].iter().all(|p| p.y >= -1e-6));
                }
            }
        }
    }
    assert_eq!(combinations.len(), 8, "{combinations:?}");
}

#[test]
fn frame_orders_keep_tooling_within_a_table_and_allow_replacements() {
    let mut orders = std::collections::BTreeMap::new();
    let mut individual = BTreeSet::new();
    for seed in 0..512 {
        let mut o = audit::fixture(ObjectKind::Chair, seed);
        o.variant = 3;
        o.interaction_target = Some(4);
        o.finish_seed = Some(2806);
        let p = parameters(&o);
        let Some(stock) = p.runner_stock(&o, INTRODUCED) else {
            continue;
        };
        assert!(p.runner_stock(&o, INTRODUCED - 1).is_none());
        assert_eq!(p.runner_stock(&o, INTRODUCED), Some(stock));
        if stock.table_order {
            let old = orders.entry(p.formed_seat.is_some()).or_insert(stock);
            assert_eq!(*old, stock);
            o.interaction_target = Some(5);
            assert_ne!(p.runner_stock(&o, INTRODUCED), Some(stock));
        } else {
            individual.insert(stock.bend_radius_m.to_bits());
        }
    }
    assert_eq!(orders.len(), 2);
    assert!(individual.len() > 12);
}

#[test]
fn wood_veneer_follows_the_bend_at_metric_scale() {
    let mut a = Assembly::default();
    let path = rounded_side(
        [
            Vec3::new(0., 0.41, -0.16),
            Vec3::new(0., 0.02, -0.21),
            Vec3::new(0., 0.02, 0.23),
            Vec3::new(0., 0.41, 0.155),
        ],
        0.075,
        f32::INFINITY,
    );
    laminate::build(&mut a, &path, 0.042, 0.021, Vec2::new(0.3, 1.2));
    let g = &a.parts[&(Surface::Wood, "chair".into())];
    for face in g.uvs.as_chunks::<{ 2 * PATH_NODES }>().0 {
        for (i, rings) in face.as_chunks::<2>().0.windows(2).enumerate() {
            assert!((rings[1][0][1] - rings[0][0][1] - path[i].distance(path[i + 1])).abs() < 3e-7);
            assert_eq!(rings[0][0][0], rings[1][0][0]);
        }
    }
}
