use super::*;
use crate::scene::procedural_indoor::objects::audit;
use std::collections::BTreeSet;

#[test]
fn pot_proportions_vary_independently_of_a_fixed_canopy_with_real_support() {
    let mut radii = Vec::new();
    let mut heights = Vec::new();
    let mut aspects = Vec::new();
    let mut saucers = BTreeSet::new();
    let mut sides = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Plant, seed);
        o.size = Vec3::new(0.9, 1.7, 0.9);
        o.variant = 0;
        let p = PotProfile::sample(seed);
        assert!(p.sides < 16 || p.sides >= p.flute_count * 4);
        let growth = Growth::sample(seed);
        let (r, h) = pot_dimensions(&o, &growth);
        radii.push(r);
        heights.push(h);
        aspects.push(h / (2. * r));
        saucers.insert(p.saucer);
        sides.insert(p.sides);
        let mut a = Assembly::default();
        let (soil, soil_radius) = crate::scene::procedural_indoor::plants::pots::build(
            &mut a,
            Surface::Ceramic,
            r,
            h,
            growth.pot_taper,
            &p,
        );
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-6 && hi.y <= h + 1e-6);
        assert!(lo.x >= -0.45 && hi.x <= 0.45 && lo.z >= -0.45 && hi.z <= 0.45);
        assert!(soil > 0. && soil < h && soil_radius > r * 0.25);
        assert!(h <= o.size.y * 0.48);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("pot {seed}: {e}"));
    }
    for values in [&mut radii, &mut heights, &mut aspects] {
        values.sort_by(f32::total_cmp);
    }
    assert!(
        radii[230] / radii[25] > 1.7,
        "narrow radius distribution: {radii:?}"
    );
    assert!(
        heights[230] / heights[25] > 2.3,
        "narrow height distribution: {heights:?}"
    );
    assert!(
        aspects[25] < 0.50 && aspects[230] > 1.10,
        "missing bowls or tall planters: {aspects:?}"
    );
    assert_eq!(saucers.len(), 2);
    assert_eq!(sides.len(), 7);
}

fn crossings(
    g: &crate::scene::procedural_indoor::geometry::Geometry,
    at: Vec3,
    dir: Vec3,
) -> Vec<f32> {
    let mut hits = Vec::new();
    for tri in g.indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
        let (ab, ac) = (b - a, c - a);
        let h = dir.cross(ac);
        let det = ab.dot(h);
        if det.abs() < 1e-10 {
            continue;
        }
        let delta = at - a;
        let u = delta.dot(h) / det;
        let q = delta.cross(ab);
        let v = dir.dot(q) / det;
        let t = ac.dot(q) / det;
        if t > 0. && u >= 0. && v >= 0. && u + v <= 1. {
            hits.push(t);
        }
    }
    hits.sort_unstable_by(f32::total_cmp);
    hits
}

#[test]
fn potting_mix_meets_the_actual_inner_wall_without_leaking_outside() {
    for seed in 0..64 {
        let p = PotProfile::sample(seed);
        let mut a = Assembly::default();
        let (r, h) = (0.18, 0.24);
        let (level, roots) = crate::scene::procedural_indoor::plants::pots::build(
            &mut a,
            Surface::Ceramic,
            r,
            h,
            0.7,
            &p,
        );
        let part = |s| &a.parts[&(s, "other_prop".into())];
        for drop in [0.024, 0.0002] {
            let y = level - h * drop;
            let at = Vec3::Y * y;
            for i in 0..37 {
                let angle = (i as f32 + 0.37) * TAU / 37.;
                let dir = radial(angle);
                let wall = crossings(part(Surface::Ceramic), at, dir);
                let soil = crossings(part(Surface::Soil), at, dir);
                assert_eq!(wall.len(), 2, "seed {seed}, sector {i}");
                assert_eq!(soil.len(), 1);
                if drop < 0.001 {
                    assert!(roots < wall[0]);
                }
                assert!(
                    wall[0] < soil[0] && soil[0] < wall[1],
                    "seed {seed}, sector {i}: {roots} {wall:?} {soil:?}"
                );
            }
        }
    }
}

#[test]
fn grounds_cross_the_fill_at_every_scale_and_preserve_the_canopy_random_stream() {
    use crate::scene::procedural_indoor::geometry::Geometry;
    for seed in 0..64 {
        for radius in [0.01, 0.06, 0.30] {
            let mut g = Geometry::default();
            let mut rng = stream(seed, 5);
            let mut original = rng.clone();
            for _ in 0..18 {
                let _ = original.random_range(0.0_f32..TAU);
                let _ = original.random_range(0.12_f32..0.92);
                let _ = original.random_range(0.0_f32..TAU);
            }
            let level = 0.15;
            soil::dressing(&mut g, seed, level, radius, &mut rng);
            assert_eq!(original.random::<u64>(), rng.random::<u64>());
            assert_eq!(g.indices.len() / 3, 18 * 12);
            for clod in g.positions.as_chunks::<36>().0 {
                assert!(clod.iter().any(|p| p[1] < level) && clod.iter().any(|p| p[1] > level));
                for p in clod {
                    assert!(Vec2::new(p[0], p[2]).length() < radius + 1e-6);
                }
            }
        }
    }
}
