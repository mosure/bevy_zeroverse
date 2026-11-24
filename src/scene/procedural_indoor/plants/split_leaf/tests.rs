use super::*;
use crate::scene::procedural_indoor::layout::ObjectKind;
use crate::scene::procedural_indoor::{geometry::Geometry, objects::audit};
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn perforations_are_real_manifold_boundaries_with_a_continuous_two_sided_lamina() {
    let mut total = [0; 3];
    for seed in 0..48 {
        let growth = Growth::sample(seed);
        let stats = canopy_stats(seed, &growth);
        let mut actual_windows = 0;
        for leaf in leaves(seed, &growth) {
            let mut g = Geometry::default();
            lamina::append(&mut g, Vec3::ZERO, Vec3::Z, 0.35, &growth, &leaf);
            let mut edges = BTreeMap::new();
            let mut vertices = BTreeSet::new();
            for tri in g.indices.as_chunks::<3>().0 {
                let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let normal = (p[1] - p[0]).cross(p[2] - p[0]);
                assert!(normal.y > 0., "folded blade {seed}: {tri:?}");
                for i in 0..3 {
                    let a = tri[i];
                    let b = tri[(i + 1) % 3];
                    vertices.insert(a);
                    let edge = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
                    edge.0 += 1;
                    edge.1 += if a < b { 1 } else { -1 };
                }
            }
            assert_eq!(vertices.len(), g.positions.len());
            let mut boundary = BTreeMap::<u32, Vec<u32>>::new();
            for ((a, b), &(count, winding)) in &edges {
                match count {
                    1 => {
                        boundary.entry(*a).or_default().push(*b);
                        boundary.entry(*b).or_default().push(*a);
                    }
                    2 => assert_eq!(winding, 0),
                    _ => panic!("nonmanifold edge {a}-{b}"),
                }
            }
            assert!(boundary.values().all(|next| next.len() == 2));
            let mut visited = BTreeSet::new();
            let mut loops = 0;
            for &start in boundary.keys() {
                if visited.contains(&start) {
                    continue;
                }
                loops += 1;
                let mut current = start;
                let mut rim = Vec::new();
                while visited.insert(current) {
                    rim.push(Vec3::from_array(g.positions[current as usize]).xz());
                    current = *boundary[&current]
                        .iter()
                        .find(|n| !visited.contains(n))
                        .unwrap_or(&start);
                }
                // The outer loop contains the shared petiole vertex (index 0).
                // A ray through each inner loop must miss the entire blade.
                if start != 0 {
                    let center = rim.iter().sum::<Vec2>() / rim.len() as f32;
                    for tri in g.indices.as_chunks::<3>().0 {
                        let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]).xz());
                        assert!(
                            (0..3)
                                .any(|i| { (p[(i + 1) % 3] - p[i]).perp_dot(center - p[i]) > 0. }),
                            "covered window in leaf from seed {seed}"
                        );
                    }
                }
            }
            let holes = leaf.bays[..leaf.count]
                .iter()
                .flatten()
                .filter(|opening| matches!(opening, Opening::Window { .. }))
                .count();
            assert_eq!(loops, 1 + holes);
            assert_eq!(
                vertices.len() as isize - edges.len() as isize + (g.indices.len() / 3) as isize,
                1 - holes as isize
            );
            actual_windows += holes as u32;
        }
        assert_eq!(actual_windows, stats.closed_windows);
        total[0] += stats.intact_leaves;
        total[1] += stats.closed_windows;
        total[2] += stats.marginal_clefts;
    }
    assert!(total.iter().all(|n| *n >= 48), "{total:?}");
}

#[test]
fn floor_and_tabletop_canopies_respect_reserves_and_historical_versions() {
    for seed in 0..96 {
        let mut object = audit::fixture(ObjectKind::Plant, seed);
        object.variant = 3;
        object.size = sample_size(3, seed % 2 == 0, &mut stream(seed, 6470));
        let mut current = Assembly::default();
        build_at_version(&mut current, &object, 207);
        let report = audit::check(&object, &current).unwrap();
        assert!(report.triangles < 20_000);
        let stalk_vertices: BTreeSet<_> = current.parts[&(Surface::Succulent, "other_prop".into())]
            .positions
            .iter()
            .map(|p| p.map(f32::to_bits))
            .collect();
        let attachments: BTreeSet<_> = current
            .parts
            .iter()
            .filter(|((surface, _), _)| matches!(surface, Surface::Leaf | Surface::LeafLight))
            .flat_map(|(_, g)| g.positions.iter().map(|p| p.map(f32::to_bits)))
            .filter(|point| stalk_vertices.contains(point))
            .collect();
        assert_eq!(attachments.len(), Growth::sample(seed).count(11));
        let mut old = Assembly::default();
        build_at_version(&mut old, &object, 206);
        for (key, geometry) in &old.parts {
            if !matches!(key.0, Surface::Bark | Surface::Leaf | Surface::LeafLight) {
                let kept = &current.parts[key];
                assert_eq!(geometry.positions, kept.positions);
                assert_eq!(geometry.indices, kept.indices);
            }
        }
        assert_ne!(
            old.parts[&(Surface::Leaf, "other_prop".into())]
                .positions
                .len(),
            current.parts[&(Surface::Leaf, "other_prop".into())]
                .positions
                .len()
        );
    }
}
