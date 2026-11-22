use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn sampled_vessels_have_closed_shells_contact_feet_and_open_mouths() {
    let mut finishes = BTreeSet::new();
    let mut flutes = BTreeSet::new();
    let mut feet = BTreeSet::new();
    let mut triangles = Vec::new();
    for kind in [ObjectKind::Bowl, ObjectKind::Vase] {
        for seed in 0..128 {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed);
            let p = parameters(&o);
            let mut a = Assembly::default();
            build(&mut a, &o);
            let check = audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert_eq!(check.batches, 1);
            assert_eq!(check.min_y_m, 0.);
            assert!((a.bounds().1.y - o.size.y).abs() < 1e-6);
            triangles.push(check.triangles);
            let g = a.parts.values().next().unwrap();
            let mut foot = 0;
            let mut center_top = 0.0_f32;
            for p in &g.positions {
                if p[1] == 0. && p[0].hypot(p[2]) > 0.001 {
                    foot += 1;
                }
                if p[0].hypot(p[2]) < 1e-7 {
                    center_top = center_top.max(p[1]);
                }
            }
            assert!(foot >= 3 && center_top > p.foot_recess_m && center_top < o.size.y * 0.25);
            assert!(
                g.indices.as_chunks::<3>().0.iter().all(|t| {
                    let [a, b, c] = t.map(|i| Vec2::from_array(g.uvs[i as usize]));
                    (b - a).perp_dot(c - a).abs() > 1e-12
                }),
                "{kind:?}/{seed}: zero-area PBR chart"
            );
            if seed < 8 {
                // Weld only coincident spatial vertices. UV wraps and shading
                // creases may split a vertex but every oriented edge needs a mate.
                let key =
                    |i: u32| g.positions[i as usize].map(|x| if x == 0. { 0 } else { x.to_bits() });
                let mut edges = BTreeMap::new();
                for t in g.indices.as_chunks::<3>().0 {
                    for j in 0..3 {
                        let (a, b) = (key(t[j]), key(t[(j + 1) % 3]));
                        let (pair, sign) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                        let entry = edges.entry(pair).or_insert((0, 0));
                        entry.0 += 1;
                        entry.1 += sign;
                    }
                }
                assert!(
                    edges.values().all(|e| *e == (2, 0)),
                    "{kind:?}/{seed}: open shell"
                );
            }
            finishes.insert(p.finish);
            flutes.insert(p.relief > 0.);
            feet.insert(p.foot_recess_m > 0.);
        }
    }
    assert_eq!((finishes.len(), flutes.len(), feet.len()), (4, 2, 2));
    triangles.sort_unstable();
    assert!(triangles.last().copied().unwrap() <= 8_000);
}

#[test]
fn turned_shoulders_stay_inside_control_radii_and_pass_through_every_station() {
    let knots = [
        (0., 0.36),
        (0.1, 0.41),
        (0.38, 0.97),
        (0.59, 0.78),
        (0.80, 0.32),
        (1., 0.43),
    ];
    for pair in knots.windows(2) {
        let [(a, ra), (b, rb)] = [pair[0], pair[1]];
        for j in 0..33 {
            let t = a + (b - a) * (j as f32 / 32.);
            let r = profile_radius(&knots, t);
            assert!(r >= ra.min(rb) - 1e-6 && r <= ra.max(rb) + 1e-6);
        }
    }
    for (t, r) in knots {
        assert!((profile_radius(&knots, t) - r).abs() < 1e-6);
    }
}
