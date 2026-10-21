use super::*;
use crate::scene::procedural_indoor::materials::program;
use std::collections::BTreeSet;

fn pure(family: usize, scale: f32, variation: f32) -> ConstructionSample {
    let mut weights = [0.; 4];
    weights[family] = 1.;
    ConstructionSample {
        weights,
        scale,
        variation,
    }
}

#[test]
fn construction_priors_are_seeded_balanced_and_continuous() {
    let mut counts = [0; 4];
    let mut samples = BTreeSet::new();
    let mut intermediate = 0;
    for seed in 0..2048 {
        let p = ConstructionSample::sample(seed);
        assert_eq!(p, ConstructionSample::sample(seed));
        p.validate().unwrap();
        counts[p.dominant()] += 1;
        intermediate += usize::from(p.weights[p.dominant()] < 0.60);
        assert!(p.weights.iter().all(|v| *v > 0.));
        samples.insert(serde_json::to_vec(&p).unwrap());
    }
    assert_eq!(samples.len(), 2048);
    assert!(counts.iter().all(|n| (400..650).contains(n)), "{counts:?}");
    assert!(
        (400..1200).contains(&intermediate),
        "intermediate finishes were lost: {intermediate}"
    );
}

#[test]
fn construction_categories_validate_at_sampled_and_authored_extremes() {
    for floor in 0..3 {
        for seed in 0..128 {
            let recipes = program::sample_with_floor(seed, floor).unwrap();
            for r in recipes {
                r.validate()
                    .unwrap_or_else(|e| panic!("{seed}/{floor}/{:?}: {e}", r.surface));
                for slot in 1..super::super::variants::structure_count(r.surface) {
                    r.variant(slot)
                        .validate()
                        .unwrap_or_else(|e| panic!("variant {seed}/{:?}/{slot}: {e}", r.surface));
                }
                if seed == 0 && labels(r.surface, floor).is_some() {
                    for family in 0..4 {
                        for scale in [0.5, 2.] {
                            for variation in [0., 1.] {
                                let mut candidate = r.clone();
                                apply(&mut candidate, pure(family, scale, variation), floor)
                                    .unwrap_or_else(|e| {
                                        panic!("{:?}/{floor}/{family}/{scale}: {e}", r.surface)
                                    });
                                assert_eq!(candidate.map_size(floor), r.map_size(floor));
                                assert_eq!(candidate.surface, r.surface);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn resolved_archives_do_not_reapply_construction_or_require_new_metadata() {
    for r in program::sample_with_floor(73, 1).unwrap() {
        let value = serde_json::to_value(&r).unwrap();
        let saved: MaterialRecipe = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(r, saved);
        let mut legacy = value;
        legacy.as_object_mut().unwrap().remove("construction");
        let legacy: MaterialRecipe = serde_json::from_value(legacy).unwrap();
        for (u, v) in [(0.13, 0.27), (-0.05, 1.02), (0.99, 0.51)] {
            let a = r.texel(u, v, 1);
            let b = legacy.texel(u, v, 1);
            assert_eq!(
                (a.color, a.height, a.roughness, a.occlusion),
                (b.color, b.height, b.roughness, b.occlusion)
            );
        }
    }
    let mut r = program::sample(3)[Surface::Wood as usize].clone();
    let before = r.clone();
    let mut bad = pure(0, 1., 0.5);
    bad.weights[0] = f32::NAN;
    assert!(apply(&mut r, bad, 0).is_err());
    assert_eq!(r, before);
    r.wood = None;
    let missing = r.clone();
    assert!(apply(&mut r, pure(1, 1., 0.5), 0).is_err());
    assert_eq!(
        r, missing,
        "missing substrate must not silently record a no-op construction"
    );
}

#[test]
fn every_supported_category_has_four_distinct_conditioned_constructions() {
    for floor in 0..3 {
        for r in program::sample_with_floor(4, floor).unwrap() {
            if labels(r.surface, floor).is_none() || r.surface != Surface::Floor && floor != 0 {
                continue;
            }
            let f: Vec<_> = (0..4)
                .map(|family| {
                    let mut r = r.clone();
                    apply(&mut r, pure(family, 1., 0.6), floor).unwrap();
                    super::super::qualification::descriptor(&r, floor)
                })
                .collect();
            for i in 0..4 {
                for j in 0..i {
                    let d: f64 = f[i].iter().zip(&f[j]).map(|(a, b)| (a - b).powi(2)).sum();
                    assert!(
                        d > 1e-4,
                        "{:?}/{floor}: families {i}/{j} collapsed: {d}",
                        r.surface
                    );
                }
            }
        }
    }
}

#[test]
fn construction_preserves_bounded_atlas_counts_and_physical_surface_contracts() {
    for seed in 0..32 {
        let base = program::sample_base(seed);
        let new = program::sample(seed);
        for (a, b) in base.iter().zip(&new) {
            assert_eq!(a.map_size(0), b.map_size(0));
            assert_eq!(a.surface, b.surface);
            if let Some(l) = &b.leaf {
                assert!(l.transmission <= 0.09 && b.period_m == 1.);
            }
            if let Some(l) = &b.leather {
                assert!(l.coat * l.nap < 0.20);
            }
            assert_eq!(
                a.coating.as_ref().and_then(|c| c.wall.as_ref()).is_some(),
                b.coating.as_ref().and_then(|c| c.wall.as_ref()).is_some(),
                "wall binding changes geometry/material partitioning"
            );
        }
        assert_eq!(
            new[Surface::Wood as usize].wood,
            new[Surface::WoodEdge as usize].wood
        );
    }
    assert_eq!(super::super::variants::COUNT, 12);
    assert_eq!(super::super::rugs::COUNT, 4);
}

#[test]
fn authored_construction_preserves_detail_and_seeded_weave_support() {
    let mut topologies: [BTreeSet<_>; 4] = std::array::from_fn(|_| BTreeSet::new());
    let mut fiber_finishes = BTreeSet::new();
    for seed in 0..256 {
        let recipes = program::sample_with_floor(seed, 0).unwrap();
        for (family, topologies) in topologies.iter_mut().enumerate() {
            let mut fabric = recipes[Surface::Fabric as usize].clone();
            apply(&mut fabric, pure(family, 1., 0.5), 0).unwrap();
            let t = fabric.textile.as_ref().unwrap();
            if family == 0 {
                fiber_finishes.insert((t.fuzz.to_bits(), t.slub.to_bits(), t.lustre.to_bits()));
            }
            topologies.insert((t.repeat, t.bundle, t.float_length, t.advance, t.herringbone));
        }
        if seed >= 4 {
            continue;
        }
        for original in recipes {
            if labels(original.surface, 0).is_none() {
                continue;
            }
            for detail in [0., 0.25, 1.] {
                let mut r = original.clone();
                r.detail_scale = Some(detail);
                apply(&mut r, pure(1, 1., 0.5), 0).unwrap();
                let once = r.clone();
                apply(&mut r, pure(1, 1., 0.5), 0).unwrap();
                assert_eq!(r, once, "authored application must be idempotent");
                if detail == 0. {
                    assert_eq!(r.texel(0.21, 0.73, 0).height, 0., "{:?}", r.surface);
                }
            }
        }
    }
    for (family, ts) in topologies.iter().enumerate() {
        assert!(
            ts.len() > 100,
            "construction {family} collapsed weave support: {}",
            ts.len()
        );
        assert_eq!(ts.iter().map(|t| t.0).collect::<BTreeSet<_>>().len(), 6);
        assert_eq!(ts.iter().map(|t| t.1).collect::<BTreeSet<_>>().len(), 3);
    }
    assert_eq!(
        fiber_finishes.len(),
        256,
        "a process prior must retain independent microstructure variation"
    );
}

#[test]
#[ignore = "bounded actual-map construction study; writes private out/material_expansion/study"]
fn construction_map_qualification() {
    super::super::qualification::construction_study();
}
