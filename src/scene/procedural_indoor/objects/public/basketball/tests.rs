use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object, part_label};
use std::collections::BTreeSet;

#[test]
fn basketball_mesh_has_an_open_bore_and_one_marked_front() {
    let mut stocks = BTreeSet::new();
    for seed in 0..64 {
        let o = audit::fixture(ObjectKind::BasketballGoal, seed);
        let goal = BasketballGoal::for_object(&o);
        stocks.insert((goal.arched, goal.timber, goal.upper_braces));
        let a = build_object(&o);
        let stats = audit::check(&o, &a).unwrap();
        assert!(stats.triangles < 2500 && stats.batches <= 6);
        assert!(stats.max_envelope_overrun_m < 0.00002);
        let (center, _) = goal.rim(o.size);
        let face = goal.face_z(o.size);
        let mut front = [Vec::new(), Vec::new()];
        for ((surface, key), g) in &a.parts {
            assert_eq!(part_label(key), CLASS);
            for tri in g.indices.as_chunks::<3>().0 {
                let ps = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                // A vertical ray through the middle must miss the ring, cord
                // diamonds, brackets and board. Check triangle interiors.
                let inside = |[a, b, c]: [Vec2; 3], p: Vec2| {
                    let ab = b - a;
                    let ac = c - a;
                    let ap = p - a;
                    let det = ab.perp_dot(ac);
                    if det.abs() < 1e-10 {
                        return false;
                    }
                    let u = ap.perp_dot(ac) / det;
                    let v = ab.perp_dot(ap) / det;
                    u >= 0. && v >= 0. && u + v <= 1.
                };
                for probe in [Vec2::ZERO, Vec2::new(0.06, -0.036)] {
                    assert!(!inside(ps.map(|p| p.xz()), center.xz() + probe));
                }
                if ps.iter().all(|v| (v.z - face).abs() < 1e-6) {
                    for (index, x) in [0.271, 0.024].into_iter().enumerate() {
                        if inside(
                            ps.map(|p| p.truncate()),
                            Vec2::new(x, goal.rim_height() + 0.239),
                        ) {
                            front[index].push((*surface, key == sport::ENAMEL));
                        }
                    }
                }
            }
        }
        // Generic timber and ink receive seeded finish slots during binding.
        let stock = (Surface::Paint, true);
        if goal.timber {
            assert_eq!(front, [vec![stock], vec![(Surface::Wood, false)]]);
        } else {
            assert_eq!(front, [vec![(Surface::Ink, false)], vec![stock]]);
        }
    }
    assert_eq!(stocks.len(), 8);
}
