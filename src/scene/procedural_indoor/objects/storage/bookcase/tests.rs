use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};

#[test]
fn upright_book_runs_use_both_ends_and_the_middle_of_shelves() {
    let mut bins = [0; 4];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Bookcase, seed);
        let p = parameters(&o);
        for (index, bay) in p.bays.iter().enumerate() {
            for level in 0..bay.levels.len() - 1 {
                let v = contents::placements(&o, &p, index, level);
                if v.is_empty() || !v.iter().all(|v| v.upright) {
                    continue;
                }
                let left = v
                    .iter()
                    .map(|v| v.position.x - rotated_half(v.size.xz(), v.yaw).x)
                    .fold(f32::INFINITY, f32::min);
                let right = v
                    .iter()
                    .map(|v| v.position.x + rotated_half(v.size.xz(), v.yaw).x)
                    .fold(f32::NEG_INFINITY, f32::max);
                let slack = bay.right - bay.left - (right - left) - 0.008;
                if slack < 0.06 {
                    continue;
                }
                let u = (left - bay.left - 0.004) / slack;
                assert!((-1e-4..=1.0001).contains(&u));
                bins[((u * 4.) as usize).min(3)] += 1;
            }
        }
    }
    assert!(
        bins.iter().all(|&v| v > 20),
        "shelf row alignment collapsed: {bins:?}"
    );
}

#[test]
fn bookcase_contents_fit_real_shelf_compartments_and_replay() {
    for seed in 0..96 {
        for size in [Vec3::new(0.7, 1.35, 0.28), Vec3::new(1.6, 2.1, 0.52)] {
            let mut o = audit::fixture(ObjectKind::Bookcase, seed);
            o.size = size;
            let p = parameters(&o);
            let whole = build_object(&o);
            audit::check(&o, &whole).unwrap();
            for (index, b) in p.bays.iter().enumerate() {
                for level in 0..b.levels.len() - 1 {
                    let base = b.levels[level] + p.panel_m * 0.5;
                    let ceiling = b.levels[level + 1] - p.panel_m * 0.5;
                    let rear = -size.z * 0.5 + if p.closed_back { p.panel_m } else { 0. };
                    let children = contents::placements(&o, &p, index, level);
                    let mut occupied = Vec::new();
                    for child in children {
                        let q = Footprint {
                            center: child.position.xz(),
                            half: rotated_half(child.size.xz(), child.yaw),
                            yaw: child.yaw,
                        };
                        assert!(q.inside(
                            Vec2::new(b.left, rear),
                            Vec2::new(b.right, size.z * 0.5),
                            0.0
                        ));
                        assert!(!occupied.iter().any(|a| q.overlaps(*a, 0.)));
                        occupied.push(q);
                        let mut mesh = Assembly::default();
                        contents::build(&mut mesh, &o, &child);
                        let (lo, hi) = mesh.bounds();
                        assert!(
                            lo.y >= base - 1e-5 && hi.y <= ceiling,
                            "{seed}: child clips shelf"
                        );
                        assert!(
                            lo.x >= b.left
                                && hi.x <= b.right
                                && lo.z >= rear
                                && hi.z <= size.z * 0.5,
                            "{seed}: child escapes bay: {lo:?}/{hi:?}"
                        );
                        assert!((lo.y - base).abs() < 0.001, "{seed}: floating child");
                    }
                }
            }
            let replay = build_object(&o);
            for (key, mesh) in whole.parts {
                assert_eq!(mesh.positions, replay.parts[&key].positions);
                assert_eq!(mesh.indices, replay.parts[&key].indices);
            }
        }
    }
}
