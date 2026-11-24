use super::*;
use crate::scene::procedural_indoor::{geometry::Geometry, layout::ObjectKind, objects::audit};

#[test]
fn grains_fit_the_actual_polygon_are_separated_and_emerge_from_below_the_fill() {
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::Plant, seed);
        let pot = PotProfile::sample(seed);
        for radius in [0.019, 0.11, 0.36] {
            let h = radius * 1.43;
            let fill = Fill::new(radius, h, &pot);
            let step = TAU / pot.sides as f32;
            let apothem = fill.radius * (step * 0.5).cos();
            for medium in [
                Medium::Loam,
                Medium::PerliteMix,
                Medium::BarkChips,
                Medium::ExpandedClay,
                Medium::MineralGravel,
            ] {
                let d = Dressing::sample(o.seed, medium, fill);
                assert!(d.grains.len() <= 32);
                for (i, g) in d.grains.iter().enumerate() {
                    for other in &d.grains[..i] {
                        assert!(g.center.distance(other.center) + 1e-6 >= g.half.x + other.half.x);
                    }
                    let mut m = Geometry::default();
                    mesh::append(&mut m, *g, medium, fill.level);
                    let mut low = f32::INFINITY;
                    let mut high = 0_f32;
                    for p in m.positions.iter().map(|v| Vec3::from_array(*v)) {
                        low = low.min(p.y);
                        high = high.max(p.y);
                        for k in 0..pot.sides {
                            assert!(
                                p.xz().dot(Vec2::from_angle((k as f32 + 0.5) * step))
                                    <= apothem + 1e-6
                            );
                        }
                    }
                    assert!(low < fill.level && high > fill.level && high < h);
                    for tri in m.indices.as_chunks::<3>().0 {
                        let [a, b, c] = tri.map(|i| Vec3::from_array(m.positions[i as usize]));
                        let normal = (b - a).cross(c - a);
                        for &i in tri {
                            let n = Vec3::from_array(m.normals[i as usize]);
                            assert!(n.is_normalized() && normal.dot(n) > 0.);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn a_larger_mouth_samples_more_exposed_aggregate_without_inflating_the_grade() {
    use std::collections::BTreeSet;
    let mut classes = BTreeSet::new();
    let mut small_counts = BTreeSet::new();
    let mut large_counts = [0_u16; 33];
    let mut larger = 0;
    for seed in 0..256 {
        let o = audit::fixture(ObjectKind::Plant, seed);
        let p = PotProfile::sample(seed);
        let m = Medium::sample(&o);
        let small = Dressing::sample(seed, m, Fill::new(0.035, 0.05, &p));
        let large = Dressing::sample(seed, m, Fill::new(0.24, 0.31, &p));
        classes.insert(m as usize);
        small_counts.insert(small.grains.len());
        large_counts[large.grains.len()] += 1;
        larger += usize::from(large.grains.len() > small.grains.len());
        assert!(large.diameter <= 0.039);
        let replay = Dressing::sample(seed, m, Fill::new(0.035, 0.05, &p));
        assert_eq!(small.grains.len(), replay.grains.len());
        for (a, b) in small.grains.iter().zip(replay.grains) {
            assert_eq!(
                (a.center, a.half, a.shape_seed),
                (b.center, b.half, b.shape_seed)
            );
        }
    }
    assert_eq!(classes.len(), 5);
    assert!(
        small_counts.contains(&0) && small_counts.len() >= 10,
        "{small_counts:?}"
    );
    assert!(large_counts.iter().filter(|&&n| n > 0).count() >= 18);
    assert!(larger > 180, "{larger}");
    assert!(large_counts.into_iter().max().unwrap() < 40);
}

#[test]
fn changed_medium_does_not_resample_stems_leaves_or_planters() {
    for seed in 0..4 {
        for variant in 0..SPECIES {
            let mut o = audit::fixture(ObjectKind::Plant, seed);
            o.variant = variant;
            let make = |version| {
                let mut a = Assembly::default();
                build_at_version(&mut a, &o, version);
                a.parts.retain(|(s, _), _| *s != Surface::Soil);
                a.parts
            };
            let previous = make(INTRODUCED - 1);
            let now = make(INTRODUCED);
            assert_eq!(previous.len(), now.len());
            for (key, g) in previous {
                let q = &now[&key];
                assert_eq!(
                    (&g.positions, &g.normals, &g.uvs, &g.indices),
                    (&q.positions, &q.normals, &q.uvs, &q.indices)
                );
            }
        }
    }
}
