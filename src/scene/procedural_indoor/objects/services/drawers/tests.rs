use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
use std::collections::BTreeSet;

#[test]
fn drawer_banks_fit_real_carcasses_with_independent_heights_and_metric_sizes() {
    let mut banks = [0; 2];
    let mut counts = BTreeSet::new();
    let mut widths = BTreeSet::new();
    let mut unequal_rows = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::DrawerUnit, seed);
        o.size = dimensions(&mut stream(seed, 0x44524d4554524943));
        let p = parameters(&o);
        banks[p.banks.len() - 1] += 1;
        counts.insert(p.weights.len());
        widths.insert((o.size.x * 100.) as u32);
        let fronts: Vec<_> = p.fronts(o.size.y).collect();
        assert_eq!(fronts.len(), p.weights.len());
        for front in &fronts {
            assert!(front.top - front.bottom > 0.065);
            assert!(front.right - front.left > 0.19);
            assert!(front.bottom > p.foot + p.panel);
            assert!(front.top < o.size.y - p.top - p.panel);
            // Tray sides are inset 30 mm from the front's edges and fit beside
            // an 18 mm partition, even when neighboring drawers fully open.
            for bank in p.banks.iter().skip(1) {
                let divider = bank.left - 0.004;
                assert!(
                    front.right - 0.03 < divider - p.panel * 0.5
                        || front.left + 0.03 > divider + p.panel * 0.5
                );
            }
        }
        for (i, a) in fronts.iter().enumerate() {
            for b in fronts.iter().skip(i + 1) {
                assert!(
                    a.right <= b.left || b.right <= a.left || a.top < b.bottom || b.top < a.bottom
                );
            }
        }
        if p.banks.len() == 2 {
            unequal_rows +=
                usize::from((fronts[0].top - fronts[p.banks[1].first].top).abs() > 0.01);
        }
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{:?}: {e}", o.size));
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-5 && (hi.y - o.size.y).abs() < 1e-5);
        assert!(a
            .parts
            .keys()
            .all(|(_, label)| label.split('#').next() == Some(o.kind.class_name())));
    }
    assert!(banks.iter().all(|v| *v > 45), "banks={banks:?}");
    assert!(counts.len() >= 6 && widths.len() > 45 && unequal_rows > 35);
    let compact_counts: BTreeSet<_> = (0..128)
        .map(|seed| {
            parameters(&audit::fixture(ObjectKind::DrawerUnit, seed))
                .weights
                .len()
        })
        .collect();
    assert!(
        compact_counts.contains(&5),
        "compact five-drawer units disappeared"
    );
    println!("drawer banks={banks:?}, drawer counts={counts:?}, width bins={}, unequal rows={unequal_rows}", widths.len());
}
