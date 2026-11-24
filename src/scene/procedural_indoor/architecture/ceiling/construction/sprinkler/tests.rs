use super::*;
use crate::scene::procedural_indoor::architecture::ceiling::sprinkler::FITTED_HEADS_VERSION as VERSION;
use std::collections::{BTreeMap, BTreeSet};

fn module(seed: u64, length: f32, width: f32, height: f32, upright: bool) -> CeilingModule {
    CeilingModule {
        system: CeilingSystem::SprinklerMain,
        center: Vec3::ZERO,
        size: Vec3::new(length, height, width),
        support_top_m: 0.8,
        pitch_m: 0.4,
        thickness_m: 0.021,
        seed,
        suspension: None,
        detail: Some(CeilingDetail {
            finish: 0,
            shape: CeilingShape::Sprinkler {
                radius_m: 0.024,
                branch_reach: 0.87,
                head_pitch_m: 1.4,
                upright,
            },
        }),
    }
}

#[test]
fn routed_heads_fit_short_shallow_and_long_runs_in_both_orientations() {
    let mut handed = [0; 2];
    let mut products = BTreeSet::new();
    for seed in 0..192 {
        for (length, width, height) in [
            (0.4, 0.08, 0.10),
            (2.9, 0.32, 0.14),
            (7.0, 0.60, 0.22),
            (36., 0.85, 0.34),
        ] {
            for upright in [false, true] {
                let m = module(seed, length, width, height, upright);
                assert!(m.sprinkler_run(VERSION - 1).is_none());
                let run = m.sprinkler_run(VERSION).unwrap();
                let y = run.outlet_y(height, upright);
                let sign = if upright { 1. } else { -1. };
                let end = y + sign
                    * (run.span + (run.crown.max(0.) + DEFLECTOR_GAUGE_M * 0.5).max(BOSS_TIP_M));
                assert!(
                    end >= 0.00199 && end <= height - 0.00199 && run.span > 0.018,
                    "{height} m, upright={upright}, end={end}, span={}",
                    run.span
                );
                let pipe_radius = 0.024_f32.min(height * 0.16) * 0.57;
                assert!((y - height * 0.64) * sign >= pipe_radius - 1e-6);
                let mut prev = None;
                for i in 0..run.count {
                    let p = run.site(&m, i);
                    assert!((p.x.abs() - length * 0.35).abs() > 0.0389);
                    assert!(p.x.abs() + 0.020 < length * 0.47);
                    assert!(p.z.abs() + run.radius < width * 0.5);
                    assert_eq!(p.x, run.site(&m, i).x);
                    if let Some(last) = prev {
                        assert!(p.x - last > length * 0.30 / f32::from(run.count));
                    }
                    prev = Some(p.x);
                }
                handed[usize::from(run.site(&m, 0).z > 0.)] += 1;
                products.insert((run.slots, run.frame as u8, run.bulb as u8));
            }
        }
    }
    assert!(handed[0] > 600 && handed[1] > 600);
    assert!(products.len() > 32);
}

fn hits(g: &Geometry, origin: Vec3, d: Vec3) -> usize {
    g.indices
        .chunks_exact(3)
        .filter(|t| {
            let p = |j: usize| Vec3::from(g.positions[t[j] as usize]);
            let e1 = p(1) - p(0);
            let e2 = p(2) - p(0);
            let n = d.cross(e2);
            let det = e1.dot(n);
            if det.abs() < 1e-11 {
                return false;
            }
            let v = origin - p(0);
            let u = v.dot(n) / det;
            let q = v.cross(e1);
            let w = d.dot(q) / det;
            u >= 0. && w >= 0. && u + w <= 1. && e2.dot(q) / det > 0.
        })
        .count()
}

#[test]
fn stamped_teeth_have_open_slots_a_closed_edge_and_seated_bulbs() {
    for seed in 0..40 {
        let m = module(seed, 3., 0.42, 0.2, true);
        let stock = m.sprinkler_run(VERSION).unwrap();
        let mut g = Geometry::default();
        deflector(&mut g, stock);
        // Every geometric edge has one incident triangle in each winding.
        let mut edges = BTreeMap::new();
        for tri in g.indices.chunks_exact(3) {
            let key = |j: usize| g.positions[tri[j] as usize].map(f32::to_bits);
            for i in 0..3 {
                let a = key(i);
                let b = key((i + 1) % 3);
                let e = edges.entry((a.min(b), a.max(b))).or_insert([0, 0]);
                e[usize::from(a > b)] += 1;
            }
        }
        assert!(edges.values().all(|e| *e == [1, 1]));
        let r = stock.radius * (stock.slot_root + 1.) * 0.50;
        for (phase, expected) in [(0.012, 0), (0.44, 2)] {
            let (s, c) = (TAU / f32::from(stock.slots) * phase).sin_cos();
            for sign in [-1., 1.] {
                assert_eq!(
                    hits(&g, Vec3::new(c * r, sign * 0.1, s * r), Vec3::NEG_Y * sign),
                    expected
                );
            }
        }
        let head = HeadMesh::new(stock);
        // A transverse ray at each end meets both the liquid column and its
        // retaining seat. Across the middle it sees the bulb in open space.
        for (fraction, seated) in [(0.125, true), (0.48, false), (0.785, true)] {
            let origin = Vec3::new(0.00013, fraction * stock.span, 0.1);
            assert!(hits(&head.bulb, origin, Vec3::NEG_Z) >= 2);
            assert_eq!(hits(&head.frame, origin, Vec3::NEG_Z) >= 2, seated);
        }
    }
}
