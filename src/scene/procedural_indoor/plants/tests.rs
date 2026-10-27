use super::*;
use crate::scene::procedural_indoor::layout::ObjectKind;
mod pots;
mod quality;
#[test]
fn habit_selection_does_not_stretch_compact_plants_into_tall_floor_envelopes() {
    let mut tall = std::collections::BTreeSet::new();
    let mut compact = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut rng = stream(seed, 6280);
        tall.insert(sample_species(Vec3::new(0.75, 1.80, 0.75), &mut rng));
        compact.insert(sample_species(Vec3::new(0.26, 0.22, 0.26), &mut rng));
    }
    assert!(!tall.contains(&7) && !tall.contains(&4));
    assert!(tall.len() >= 8 && compact.contains(&7) && compact.contains(&4));
    assert!(!compact.contains(&2) && !compact.contains(&5));
}
#[test]
fn all_botanical_forms_have_supported_pots_valid_leaf_frames_and_bounded_canopies() {
    for variant in 0..SPECIES {
        let mut topology_counts = std::collections::BTreeSet::new();
        for seed in 0..12 {
            let o = IndoorObject {
                finish_seed: None,
                id: 0,
                kind: ObjectKind::Plant,
                position: Vec3::ZERO,
                size: Vec3::new(0.9, 1.7, 0.9),
                yaw: 0.0,
                variant,
                seed,
                solid: true,
                support: None,
                neighbor: false,
                interaction_target: None,
                cable_path: None,
                cable_target: None,
            };
            let mut a = Assembly::default();
            build(&mut a, &o);
            let (lo, hi) = a.bounds();
            assert!(
                lo.y >= -0.001
                    && hi.y <= o.size.y
                    && lo.x >= -0.45
                    && hi.x <= 0.45
                    && lo.z >= -0.45
                    && hi.z <= 0.45,
                "species {variant}: {lo:?} {hi:?}"
            );
            let mut triangles = 0;
            for g in a.parts.values() {
                assert!(g
                    .normals
                    .iter()
                    .all(|n| (Vec3::from_array(*n).length() - 1.0).abs() < 0.001));
                for t in g.indices.as_chunks::<3>().0 {
                    let p = t
                        .iter()
                        .map(|i| Vec3::from_array(g.positions[*i as usize]))
                        .collect::<Vec<_>>();
                    assert!((p[1] - p[0]).cross(p[2] - p[0]).length_squared() > 1e-18);
                }
                triangles += g.indices.len() / 3;
            }
            assert!(
                (1000..20000).contains(&triangles),
                "species {variant}: {triangles} triangles"
            );
            topology_counts.insert(triangles);
        }
        assert!(
            topology_counts.len() >= 4,
            "plant growth did not diversify topology: {variant}"
        );
    }
}

#[test]
fn compact_and_wide_botanical_canopies_keep_actual_vertices_in_reserved_bounds() {
    for dimensions in [
        Vec3::new(0.18, 0.23, 0.16),
        Vec3::new(0.95, 2.12, 1.064),
        Vec3::new(0.45, 2.30, 0.45),
        Vec3::new(0.75, 0.40, 0.70),
    ] {
        for variant in 0..SPECIES {
            for seed in 0..12 {
                let o = IndoorObject {
                    finish_seed: None,
                    id: 0,
                    kind: ObjectKind::Plant,
                    position: Vec3::ZERO,
                    size: dimensions,
                    yaw: 0.,
                    variant,
                    seed,
                    solid: true,
                    support: None,
                    neighbor: false,
                    interaction_target: None,
                    cable_path: None,
                    cable_target: None,
                };
                let mut a = Assembly::default();
                let growth = Growth::sample(o.seed);
                let (radius, height) = pot_dimensions(&o, &growth);
                assert_eq!((radius, height), pot_dimensions(&o, &growth));
                assert!(height / (2. * radius) <= 1.65);
                assert!(radius * 1.10 <= dimensions.x.min(dimensions.z) * 0.5);
                if dimensions.y > 2.2 && dimensions.x < 0.5 {
                    assert!(radius * 2. >= 0.25, "insufficient tall-plant root volume");
                }
                build(&mut a, &o);
                let (lo, hi) = a.bounds();
                assert!(
                    lo.y >= -0.0001
                        && hi.y <= dimensions.y + 0.0001
                        && lo.x >= -dimensions.x * 0.5
                        && hi.x <= dimensions.x * 0.5
                        && lo.z >= -dimensions.z * 0.5
                        && hi.z <= dimensions.z * 0.5,
                    "{} seed {seed} size {dimensions:?}: {lo:?} {hi:?}",
                    species_name(variant)
                );
                for g in a.parts.values() {
                    assert!(g.normals.iter().all(|n| n.iter().all(|v| v.is_finite())
                        && (Vec3::from_array(*n).length() - 1.).abs() < 0.001));
                }
            }
        }
    }
}
