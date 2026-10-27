use super::*;
use crate::scene::procedural_indoor::motion::IndoorMotionConfig;
mod ottoman;

#[test]
fn benches_keep_ergonomic_seats_and_grounded_fitted_construction() {
    let mut bases = std::collections::BTreeSet::new();
    let mut backless = 0;
    for seed in 0..96 {
        for size in [
            Vec3::new(0.85, 0.46, 0.33),
            Vec3::new(1.4, 0.84, 0.5),
            Vec3::new(2.1, 1.05, 0.58),
        ] {
            let mut o = audit::fixture(ObjectKind::Bench, seed);
            o.size = size;
            let p = bench::parameters(&o);
            bases.insert(p.base);
            backless += usize::from(!p.backed && size.y > 0.8);
            assert!((0.40..=0.52).contains(&p.seat_height));
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            let (lo, _) = a.bounds();
            assert!(lo.y.abs() < 0.0001);
            let seat_vertices = a
                .parts
                .iter()
                .filter(|((s, _), _)| *s == p.seat)
                .flat_map(|(_, g)| &g.positions)
                .filter(|v| (v[1] - p.seat_height).abs() < 0.003)
                .count();
            assert!(
                seat_vertices >= 8,
                "seat geometry disagrees with ergonomics {seed}"
            );
        }
    }
    assert_eq!(bases.len(), 4);
    assert!(backless > 25);
}

#[test]
fn drawer_hardware_moves_with_fronts_and_top_remains_a_level_support() {
    let policy = IndoorMotionConfig {
        cabinet_fraction: 1.,
        ..Default::default()
    };
    for seed in 0..64 {
        for size in [Vec3::new(0.31, 0.43, 0.34), Vec3::new(0.65, 0.81, 0.61)] {
            let mut o = audit::fixture(ObjectKind::DrawerUnit, seed);
            o.size = size;
            let p = drawers::parameters(&o);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 0.0001 && (hi.y - size.y).abs() < 0.0001);
            assert!(a.joints.is_empty());
            let moving = super::super::build_object_with_motion(&o, Some(&policy));
            assert_eq!(moving.joints.len(), p.weights.len());
            for joint in &moving.joints {
                assert!(!joint.parts.is_empty());
                assert!(joint.parts.iter().all(|k| moving.parts.contains_key(k)));
                assert!(joint
                    .parts
                    .iter()
                    .any(|k| k.0 == p.hardware_finish || k.0 == Surface::Ink));
            }
            // Segmentation may change batching; it must not change the rest pose.
            let vertices = |assembly: &Assembly| {
                let mut v: Vec<_> = assembly
                    .parts
                    .values()
                    .flat_map(|g| &g.positions)
                    .map(|v| v.map(f32::to_bits))
                    .collect();
                v.sort();
                v
            };
            assert_eq!(vertices(&a), vertices(&moving));
        }
    }
}
