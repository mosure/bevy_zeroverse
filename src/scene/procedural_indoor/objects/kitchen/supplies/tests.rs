use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};

#[test]
fn supplies_fit_metric_extremes_and_keep_semantics() {
    for kind in [
        ObjectKind::Bread,
        ObjectKind::KitchenTowel,
        ObjectKind::CuttingBoard,
        ObjectKind::SoapDispenser,
        ObjectKind::KnifeBlock,
    ] {
        for seed in 0..64 {
            for scale in [
                Vec3::ONE,
                Vec3::new(0.70, 0.60, 1.2),
                Vec3::new(1.3, 1.5, 0.65),
            ] {
                let mut o = audit::fixture(kind, seed);
                o.size *= scale;
                let a = build_object(&o);
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}/{scale:?}: {e}"));
                let (lo, _) = a.bounds();
                assert!(lo.y.abs() < 1e-5, "unsupported {kind:?}/{seed}: {lo:?}");
                assert!(a
                    .parts
                    .keys()
                    .all(|(_, l)| l.split('#').next() == Some(kind.class_name())));
            }
        }
    }
}

#[test]
fn knife_handles_are_not_forced_to_identical_rows_or_occupied_slots() {
    let mut empty = 0;
    let mut rotated = 0;
    let mut counts = std::collections::BTreeSet::new();
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::KnifeBlock, seed);
        let p = knives::parameters(&o);
        let items = knives::contents(&o, &p);
        counts.insert(items.iter().filter(|v| v.occupied).count());
        for (i, v) in items.iter().enumerate() {
            empty += usize::from(!v.occupied);
            rotated += usize::from(v.footprint.yaw.sin().abs() > 0.10);
            assert!(items[..i]
                .iter()
                .all(|u| !u.footprint.overlaps(v.footprint, 0.002)));
            assert!(p.body_fraction * o.size.y + v.blade + v.handle.y < o.size.y + 1e-6);
        }
    }
    assert!(empty > 30 && rotated > 150 && counts.len() >= 5);
}
