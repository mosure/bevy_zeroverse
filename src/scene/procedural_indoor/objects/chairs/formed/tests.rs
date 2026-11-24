use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn back_stays_pass_behind_the_sitting_surface() {
    for depth in [0.29, 0.42, 0.55] {
        for socket_z in [-0.03, 0.11, 0.20] {
            let mut g = Geometry::default();
            back_stays(
                &mut g,
                0.48,
                depth,
                [-1., 1.].map(|side| Vec3::new(side * 0.15, 0.57, socket_z)),
            );
            for tri in g.indices.as_chunks::<3>().0 {
                let [a, b, c] =
                    [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
                let normal: Vec3 = tri
                    .iter()
                    .map(|&i| Vec3::from_array(g.normals[i as usize]))
                    .sum();
                assert!(
                    (b - a).cross(c - a).dot(normal) > 0.,
                    "inverted back carrier at depth {depth}, socket {socket_z}"
                );
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    let a = Vec3::from_array(g.positions[tri[i] as usize]);
                    let b = Vec3::from_array(g.positions[tri[j] as usize]);
                    for y in [0.442, 0.46, 0.47, 0.49, 0.50] {
                        if a.y.min(b.y) < y && y < a.y.max(b.y) {
                            assert!(a.lerp(b, (y - a.y) / (b.y - a.y)).z > depth * 0.5);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn one_table_can_share_tooling_with_individual_wood_cuts_and_replacements() {
    let mut order = None;
    let mut replacements = BTreeSet::new();
    let mut phases = BTreeSet::new();
    for seed in 0..128 {
        let mut o =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::Chair, seed);
        o.variant = 3;
        o.finish_seed = Some(4017);
        o.interaction_target = Some(2);
        let Some(seat) = parameters(&o).formed_seat else {
            continue;
        };
        if seat.table_order {
            assert_eq!(*order.get_or_insert(seat), seat);
            let a = crate::scene::procedural_indoor::objects::build_object(&o);
            let (_, face) = a
                .parts
                .iter()
                .find(|((s, _), _)| *s == Surface::Wood)
                .unwrap();
            // The difference between the face's own metric chart and its
            // final UV is the individual object's cut from the stock.
            let v = Vec3::from_array(face.positions[0]).xz();
            let (s, c) = seat.chart_angle_radians.sin_cos();
            let origin = Vec2::from_array(face.uvs[0])
                - Vec2::new(v.dot(Vec2::new(c, -s)), v.dot(Vec2::new(s, c)));
            assert!(phases.insert(origin.to_array().map(f32::to_bits)));
            o.interaction_target = Some(7);
            let other_table = parameters(&o).formed_seat.unwrap();
            assert!(other_table.table_order && other_table != seat);
        } else {
            replacements.insert(seat.gauge_m.to_bits());
        }
    }
    assert!(phases.len() > 32 && replacements.len() > 5);
}

fn ordinates(a: &Assembly, xz: Vec2) -> Vec<f32> {
    super::super::tests::vertical_intersections(a.parts.values(), xz)
}

#[test]
fn finished_shells_are_closed_metric_surfaces_at_the_pelvis_datum() {
    let mut styles = BTreeSet::new();
    for seed in 0..48 {
        let mut o =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::Chair, seed);
        o.variant = 3 + (seed % 2) as u32;
        let p = parameters(&o);
        let Some(seat) = p.formed_seat else { continue };
        styles.insert(p.seat_style);
        let half = Vec2::new(0.245, 0.21);
        let mut a = Assembly::default();
        seat.shell(&mut a, p.seat_surface, half);
        crate::scene::procedural_indoor::objects::audit::check(&o, &a).unwrap();
        let mut edges = BTreeMap::new();
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    let point = |idx: usize| {
                        g.positions[tri[idx] as usize]
                            .map(|v| if v == 0. { 0 } else { v.to_bits() })
                    };
                    let (a, b) = (point(i), point(j));
                    let (count, winding) = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                    *count += 1;
                    *winding += if a < b { 1 } else { -1 };
                }
            }
        }
        assert!(
            edges.values().all(|e| *e == (2, 0)),
            "open or duplicated edge, seed {seed}"
        );
        let center = ordinates(&a, Vec2::ZERO);
        assert_eq!(center.len(), 2);
        assert!((center[1] - SEAT_Y).abs() < 1e-5);
        assert!((center[1] - center[0] - seat.gauge_m).abs() < 1e-5);
        let nose = ordinates(&a, Vec2::new(0., -half.y * 0.94));
        let wing = ordinates(&a, Vec2::new(half.x * 0.82, half.y * 0.11));
        assert_eq!((nose.len(), wing.len()), (2, 2));
        assert!(nose[1] < SEAT_Y - 0.003 && wing[1] > SEAT_Y + 0.002);
        assert!(wing[1] < 0.50 && nose[0] < nose[1]);
        let wing_gauge = wing[1] - wing[0];
        if seat.stock == SeatStock::CarvedTimber {
            assert!(wing_gauge > seat.gauge_m + 0.002);
        } else {
            assert!((wing_gauge - seat.gauge_m).abs() < 1e-5);
        }
        let face = &a.parts[&(p.seat_surface, "chair".into())];
        for (ix, jx) in [(13, 31), (52, 61), (40, 44)] {
            let xz = |i: usize| Vec3::from_array(face.positions[i]).xz();
            let uv = |i: usize| Vec2::from_array(face.uvs[i]);
            assert!((uv(ix).distance(uv(jx)) - xz(ix).distance(xz(jx))).abs() < 1e-6);
        }
    }
    assert_eq!(styles, BTreeSet::from([3, 4, 5]));
}

#[test]
fn all_frames_carry_the_shell_at_four_bearings_without_a_full_width_pan() {
    let mut combinations = BTreeSet::new();
    for seed in 0..96 {
        let mut o =
            crate::scene::procedural_indoor::objects::audit::fixture(ObjectKind::Chair, seed);
        o.variant = 3 + (seed % 2) as u32;
        let p = parameters(&o);
        let Some(seat) = p.formed_seat else { continue };
        combinations.insert((p.seat_style, p.base_style));
        let (width, depth) = (0.49, 0.43);
        let mut skins = Assembly::default();
        seat.shell(&mut skins, p.seat_surface, Vec2::new(width, depth) * 0.5);
        let mut a = Assembly::default();
        seat.build(&mut a, &p, width, depth);
        let center = ordinates(&a, Vec2::ZERO);
        assert_eq!(center.len(), 2, "frame fills the centre of the seat");
        let version = crate::scene::procedural_indoor::layout::GENERATOR_VERSION;
        base::build(&mut a, &o, &p, width, depth, 0.39, version);
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                let bearing = Vec2::new(side * width * 0.29, end * depth * 0.25);
                let hits = ordinates(&a, bearing);
                let shell = ordinates(&skins, bearing);
                // Lower and upper crossbar faces, bearing, then the two seat
                // skins. Their ranges overlap at each interface.
                assert_eq!(hits.len(), 6, "{seed} {hits:?}");
                assert!((hits[1] - 0.431).abs() < 1e-5 && (hits[2] - 0.432).abs() < 1e-5);
                assert_eq!(shell.len(), 2);
                assert!((shell[0] - hits[3]).abs() < 1e-5 && (shell[1] - hits[5]).abs() < 1e-5);
                assert!(
                    hits[4] > shell[0] && hits[4] < shell[1] - 0.0005,
                    "{seed} {hits:?}"
                );
                let rail = ordinates(&a, Vec2::new(side * width * 0.429, end * depth * 0.25));
                assert!(rail.iter().any(|y| (0.39..0.417).contains(y)));
                if p.base_style == 6 {
                    let stock = p.runner_stock(&o, version).unwrap();
                    // The horizontal runner enters the underside of the crossbar
                    // but stays below the seat shell.
                    assert!(rail.iter().any(|y| (0.4145..0.419).contains(y)));
                    assert!(rail.iter().any(|y| (0.417 - stock.thickness_m - 0.002
                        ..0.417 - stock.thickness_m + 0.002)
                        .contains(y)));
                } else {
                    assert!(rail.iter().any(|y| (0.435..0.45).contains(y)));
                }
            }
        }
    }
    assert_eq!(combinations.len(), 10);
}
