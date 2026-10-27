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
