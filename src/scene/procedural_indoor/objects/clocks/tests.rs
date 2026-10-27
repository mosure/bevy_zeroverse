use super::*;
use std::collections::BTreeSet;

#[test]
fn clock_cases_and_relief_fit_varied_aspects_and_shallow_depths() {
    let mut depths = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for seed in 0..96 {
        for size in [
            Vec3::new(0.20, 0.20, 0.032),
            Vec3::new(0.72, 0.32, 0.040),
            Vec3::new(0.25, 0.40, 0.080),
            Vec3::new(0.46, 0.46, 0.032),
        ] {
            let mut o = audit::fixture(ObjectKind::Clock, seed);
            o.size = size;
            let p = parameters(&o);
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{size:?}: {e}"));
            let (lo, hi) = a.bounds();
            assert!(lo
                .cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(1e-5))
                .all());
            assert!(hi
                .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(1e-5))
                .all());
            assert!((lo.z + case_depth(&o) * 0.5).abs() < 1e-6);
            let dial_half = size.xy() * 0.5 * (1. - p.bezel_fraction).min(0.98);
            let front = case_depth(&o) * 0.5;
            // Every dial mark must have a physical dial behind it, including
            // digital strokes at the corners of a rounded housing.
            for g in a.parts.values() {
                for v in &g.positions {
                    let v = Vec3::from_array(*v);
                    if v.z < front + 0.0014 {
                        continue;
                    }
                    let uv = (v.xy() - Vec2::Y * size.y * 0.5).abs() / dial_half;
                    assert!(
                        uv.x.powf(p.roundness) + uv.y.powf(p.roundness) < 1.001,
                        "unsupported mark {seed}/{size:?}: {v:?}"
                    );
                }
            }
            depths.insert((case_depth(&o) * 10000.).round() as i32);
            shapes.insert((
                (p.roundness * 100.).round() as i32,
                (p.case_taper * 100.).round() as i32,
            ));
        }
    }
    assert!(depths.len() > 100 && shapes.len() > 60);
}

#[test]
fn clock_metric_sampling_and_finish_identity_are_independent() {
    let mut ratios = BTreeSet::new();
    let mut depths = BTreeSet::new();
    for seed in 0..128 {
        let mut r = stream(seed, 99881);
        let size = dimensions(&mut r);
        assert!((0.20..0.46).contains(&size.y) && (0.032..0.080).contains(&size.z));
        ratios.insert((size.x / size.y * 100.).round() as i32);
        depths.insert((size.z * 10000.).round() as i32);
        let mut o = audit::fixture(ObjectKind::Clock, seed);
        o.size = size;
        let a = audit::descriptor(&o, &build_object(&o));
        o.finish_seed = Some(10000 + seed);
        assert_eq!(a, audit::descriptor(&o, &build_object(&o)));
        let angles = hand_angles(parameters(&o).seconds);
        assert!(angles
            .iter()
            .all(|a| a.is_finite() && (0.0..TAU).contains(a)));
    }
    assert!(ratios.len() > 35 && depths.len() > 90);
}
