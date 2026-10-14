use super::*;

fn object(seed: u64) -> IndoorObject {
    IndoorObject {
        finish_seed: None,
        id: 0,
        kind: ObjectKind::Sofa,
        position: Vec3::ZERO,
        size: Vec3::new(2.8, 0.92, 1.85),
        yaw: 0.,
        variant: 0,
        seed,
        solid: true,
        support: None,
        neighbor: false,
        interaction_target: None,
        cable_path: None,
        cable_target: None,
    }
}
#[test]
fn asymmetric_upholstery_modules_reserve_arms_and_replay_exactly() {
    let mut signatures = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let o = object(seed);
        let p = parameters(&o);
        let widths = module_widths(&o);
        assert_eq!(widths.len(), p.modules);
        assert_eq!(widths, module_widths(&o));
        assert!((widths.iter().sum::<f32>() + p.arm_width * 2. - o.size.x).abs() < 1e-6);
        assert!(widths.iter().all(|&w| w > p.cushion_gap_m + 0.20));
        let smallest = widths.iter().copied().fold(f32::INFINITY, f32::min);
        let largest = widths.iter().copied().fold(0., f32::max);
        assert!(
            largest / smallest < 1.64,
            "realistic module ratio: {widths:?}"
        );
        signatures.insert(widths.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(signatures.len(), 256);
}
#[test]
fn upholstery_profiles_are_closed_and_bounded_for_short_loveseats_and_deep_chaise() {
    let mut scene = super::super::super::layout::IndoorManifest::generate_with_humans(
        17,
        super::super::super::layout::IndoorLayout::Lounge,
        0.6,
        0,
        0.,
    )
    .unwrap();
    scene.objects.clear();
    for seed in 0..64 {
        for (kind, size, variant) in [
            (ObjectKind::Sofa, Vec3::new(1.30, 0.76, 0.82), 0),
            (ObjectKind::Sofa, Vec3::new(2.8, 0.92, 1.85), 0),
            (ObjectKind::Chair, Vec3::new(0.68, 0.95, 0.68), 8),
        ] {
            let mut o = object(seed);
            o.kind = kind;
            o.size = size;
            o.variant = variant;
            o.id = scene.objects.len();
            let assembly = super::super::build_object(&o);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, -0.003, -size.z * 0.5) - Vec3::splat(1e-5))
                    .all(),
                "{kind:?}/{seed}: {lo:?}"
            );
            assert!(
                hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.003))
                    .all(),
                "{kind:?}/{seed}: {hi:?}"
            );
            scene.objects.push(o);
        }
    }
    super::super::super::validation::validate_geometry(&scene).unwrap();
}
