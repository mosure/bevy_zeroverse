use super::*;
use crate::scene::procedural_indoor::{layout::stream, program::court::room_size};
use rand::Rng;

fn triangles(g: &Geometry) -> impl Iterator<Item = [Vec2; 3]> + '_ {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|tri| tri.map(|i| Vec3::from_array(g.positions[i as usize]).xz()))
}

fn covers([a, b, c]: [Vec2; 3], p: Vec2) -> bool {
    let [a, b, c, p] = [a, b, c, p].map(|p| p.as_dvec2());
    (b - a).perp_dot(p - a) <= 0. && (c - b).perp_dot(p - b) <= 0. && (a - c).perp_dot(p - c) <= 0.
}

#[test]
fn practice_paint_partitions_timber_and_preserves_the_volleyball_lines() {
    let mut rng = stream(90123, 9427);
    for (i, forward) in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y]
        .into_iter()
        .enumerate()
    {
        for semicircle in [false, true] {
            let seed = 72100 + i as u64;
            let size = room_size(seed);
            let court = VolleyballCourt::fit(seed, size).unwrap();
            let mut old = Assembly::default();
            top(&mut old, court, size, None);
            let mark = FreeThrowMark {
                goal_id: 1,
                center: court.world(Vec2::new(0.33, court.attack_distance + 0.026)),
                forward,
                line_length_m: if semicircle { 3.6 } else { 1.39 },
                backboard_distance_m: 4.60,
                semicircle,
            };
            let mut new = Assembly::default();
            top(&mut new, court, size, Some(&[mark]));
            assert_eq!(new.parts.len(), 3);
            let key = (Surface::Paint, sport::LINE.to_owned());
            // Primary ink keeps its area; a junction may split its boundary edges.
            let painted_area = |g: &Geometry| -> f64 {
                triangles(g)
                    .map(|t| {
                        let [a, b, c] = t.map(|v| v.as_dvec2());
                        (c - a).perp_dot(b - a) * 0.5
                    })
                    .sum()
            };
            assert!((painted_area(&old.parts[&key]) - painted_area(&new.parts[&key])).abs() < 2e-5);
            assert_closed_top(&new, size);
            let mut total = 0.;
            for g in new.parts.values() {
                for ((&pos, &n), &uv) in g.positions.iter().zip(&g.normals).zip(&g.uvs) {
                    assert_eq!(pos[1], 0.);
                    assert!((Vec3::from_array(n) - Vec3::Y).length() < 0.00001);
                    assert!(Vec2::from_array(uv).abs_diff_eq(Vec2::new(pos[0], -pos[2]), 0.00001));
                }
                for t in triangles(g) {
                    let [a, b, c] = t.map(|p| p.as_dvec2());
                    let area = (c - a).perp_dot(b - a) * 0.5;
                    assert!(area > 0.);
                    total += area;
                }
            }
            assert!((total - f64::from(size.x) * f64::from(size.z)).abs() < 2e-4);
            let primary: Vec<_> = triangles(&old.parts[&key]).collect();
            let faces: Vec<_> = new
                .parts
                .iter()
                .flat_map(|((_, label), g)| triangles(g).map(move |t| (label.as_str(), t)))
                .collect();
            let mut checked = [0; 3];
            for _ in 0..4000 {
                let across = rng.random_range(-2.1..2.1);
                let along = rng.random_range(-0.34..2.05);
                let p = mark.world(across, along);
                let radius = mark.line_length_m * 0.5;
                let d = Vec2::new(across, along).length();
                // The arcs are inscribed regular chords. Check the analytic circle
                // off its 4 mm chord error, and the surface partition everywhere.
                let mut hit = faces.iter().filter(|(_, t)| covers(*t, p));
                let label = hit.next().expect("continuous walking surface").0;
                assert!(hit.next().is_none(), "overlapping floor at {p}");
                if primary.iter().any(|t| covers(*t, p)) {
                    assert_eq!(label, sport::LINE);
                    checked[0] += 1;
                } else if !(semicircle
                    && along > 0.
                    && ((d - radius).abs() < 0.004 || (d - radius + LINE).abs() < 0.004))
                {
                    let stripe = across.abs() <= radius && along.abs() <= LINE * 0.5;
                    let arc = semicircle && along > 0. && (radius - LINE..=radius).contains(&d);
                    assert_eq!(
                        label,
                        if stripe || arc {
                            sport::PRACTICE
                        } else {
                            "floor"
                        }
                    );
                    checked[if stripe || arc { 1 } else { 2 }] += 1;
                }
            }
            assert!(checked.into_iter().all(|n| n > 0));
        }
    }
}

/// Every interior edge has the exact same two world-space endpoints on both
/// sides. Equal total area alone cannot catch raster cracks at T-junctions.
fn assert_closed_top(a: &Assembly, size: Vec3) {
    use std::collections::BTreeMap;
    let mut edges = BTreeMap::<_, u32>::new();
    let key = |v: Vec2| v.to_array().map(|x| if x == 0. { 0 } else { x.to_bits() });
    for tri in a.parts.values().flat_map(triangles) {
        for i in 0..3 {
            let mut edge = [key(tri[i]), key(tri[(i + 1) % 3])];
            edge.sort_unstable();
            *edges.entry(edge).or_default() += 1;
        }
    }
    for (edge, count) in edges {
        let [a, b] = edge.map(|v| Vec2::from_array(v.map(f32::from_bits)));
        let exterior = (0..2).any(|axis| {
            a[axis] == b[axis] && (a[axis].abs() - size.xz()[axis] * 0.5).abs() < 0.00001
        });
        assert_eq!(count, if exterior { 1 } else { 2 }, "floor edge {a} to {b}");
    }
}

#[test]
fn court_semicircle_has_no_unmatched_raster_edges_in_oblique_camera_fixture() {
    let size = room_size(72104);
    let court = VolleyballCourt::fit(72104, size).unwrap();
    let mark = FreeThrowMark {
        goal_id: 4,
        center: Vec2::new(9.632776, 1.6547842),
        forward: Vec2::NEG_X,
        line_length_m: 3.6,
        backboard_distance_m: 3.5340176,
        semicircle: true,
    };
    let mut a = Assembly::default();
    top(&mut a, court, size, Some(&[mark]));
    assert_closed_top(&a, size);
    let mut bare = Assembly::default();
    top(&mut bare, court, size, Some(&[]));
    assert_closed_top(&bare, size);
}
