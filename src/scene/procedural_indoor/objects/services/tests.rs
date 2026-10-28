use super::*;
use crate::scene::procedural_indoor::motion::IndoorMotionConfig;
mod ottoman;

#[test]
fn service_render_inputs_record_actual_metric_shapes_and_independent_finishes() {
    let Some(path) = std::env::var_os("ZEROVERSE_AMENITY_REPORT") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    std::fs::create_dir_all(&path).unwrap();
    let mut categories = Vec::new();
    for kind in [
        ObjectKind::DrawerUnit,
        ObjectKind::Bench,
        ObjectKind::MobileCart,
        ObjectKind::RoomDivider,
    ] {
        let mut maps = Vec::new();
        let mut samples = Vec::new();
        for seed in 0..64 {
            let mut o = audit::fixture(kind, seed);
            let mut r = stream(seed, 0x534552564d455452);
            o.size = match kind {
                ObjectKind::DrawerUnit => drawers::dimensions(&mut r),
                ObjectKind::Bench => Vec3::new(
                    r.random_range(0.95..1.70),
                    r.random_range(0.46..0.87),
                    r.random_range(0.39..0.54),
                ),
                ObjectKind::MobileCart => Vec3::new(
                    r.random_range(0.60..0.91),
                    r.random_range(0.74..1.02),
                    r.random_range(0.42..0.59),
                ),
                ObjectKind::RoomDivider => Vec3::new(
                    r.random_range(0.90..1.60),
                    r.random_range(1.05..1.70),
                    r.random_range(0.35..0.49),
                ),
                _ => unreachable!(),
            };
            o.finish_seed = Some(seed ^ 0x415050454152);
            let a = build_object(&o);
            let mesh = audit::check(&o, &a).unwrap();
            maps.push(audit::descriptor(&o, &a));
            samples.push(serde_json::json!({"object":o,"program":audit::program(&o),"mesh":mesh}));
        }
        categories.push(serde_json::json!({"kind":kind,"samples":samples,"depth_descriptors":maps,"failures":[]}));
    }
    std::fs::write(path.join("service-metric-audit.json"), serde_json::to_vec(&serde_json::json!({
        "build_provenance":crate::provenance::capture_provenance(),"categories":categories,
        "method":"metric service furniture samples; geometry and finish identities independent; selected renders are illustrative, not frequencies"
    })).unwrap()).unwrap();
}

#[test]
fn service_finishes_change_without_altering_shape_or_motion() {
    use ObjectKind::*;
    for kind in [DrawerUnit, MobileCart, Bench, RoomDivider] {
        let mut finishes = std::collections::BTreeSet::new();
        let mut changed = 0;
        for seed in 0..64 {
            let o = audit::fixture(kind, seed);
            let mut other = o.clone();
            other.finish_seed = Some(seed ^ 0x46534f424a454354);
            let a = build_object(&o);
            let b = build_object(&other);
            let vertices = |a: &Assembly, metric_uvs: bool| {
                let mut v: Vec<_> = a
                    .parts
                    .values()
                    .flat_map(|g| {
                        g.indices.iter().map(|&i| {
                            let i = i as usize;
                            (
                                g.positions[i].map(f32::to_bits),
                                g.normals[i].map(f32::to_bits),
                                if metric_uvs {
                                    g.uvs[i].map(f32::to_bits)
                                } else {
                                    [0; 2]
                                },
                            )
                        })
                    })
                    .collect();
                v.sort_unstable();
                v
            };
            assert!(
                vertices(&a, false) == vertices(&b, false),
                "finish changed {kind:?}/{seed} geometry"
            );
            // Substrate-specific UV phase is part of material assignment. The
            // constructor's metric coordinates must remain shape-independent.
            let raw = |o: &IndoorObject| {
                let mut a = Assembly::default();
                super::build(&mut a, o);
                a
            };
            assert!(
                vertices(&raw(&o), true) == vertices(&raw(&other), true),
                "finish changed {kind:?}/{seed} metric UVs"
            );
            changed += usize::from(
                a.parts.keys().collect::<Vec<_>>() != b.parts.keys().collect::<Vec<_>>(),
            );
            finishes.insert(construction(&other));
            audit::check(&other, &b).unwrap();
        }
        assert!(
            changed > 24 && finishes.len() > 20,
            "{kind:?}: changed={changed} constructions={}",
            finishes.len()
        );
    }
}

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
