use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
#[test]
fn dispenser_taps_trays_and_head_proportions_are_bounded_and_replayable() {
    let mut taps = std::collections::BTreeSet::new();
    let mut grilles = std::collections::BTreeSet::new();
    let mut heads = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::DrinkDispenser, seed);
        o.size = super::super::dimensions(o.kind, seed);
        let p = parameters(&o);
        grilles.insert((p.grille_count, p.grille_crosswise));
        if p.construction == 3 {
            taps.insert(p.tap_count);
            heads.insert((
                (p.front * 100.) as u32,
                (p.bay_top * 100.) as u32,
                (p.body_width * 100.) as u32,
            ));
            assert!(p.label_center - p.label_height * 0.5 > 0.);
            assert!(p.label_center + p.label_height * 0.5 < 1.);
            assert!(p.bay_top < p.top_fraction + 0.04);
        }
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-6);
        assert!(hi.y <= o.size.y + 1e-6);
        assert!(a.parts.keys().all(|(_, l)| matches!(
            crate::scene::procedural_indoor::objects::part_label(l),
            "appliance" | "faucet"
        )));
        let replay = build_object(&o);
        for (key, g) in &a.parts {
            assert_eq!(g.positions, replay.parts[key].positions);
            assert_eq!(g.indices, replay.parts[key].indices);
        }
        o.finish_seed = Some(913);
        let alt = parameters(&o);
        assert_eq!(p.construction, alt.construction);
        assert_eq!(p.tap_count, alt.tap_count);
        assert_eq!(p.body_bottom, alt.body_bottom);
        assert_eq!(p.fill, alt.fill);
    }
    assert_eq!(taps.len(), 4);
    assert_eq!(grilles.len(), 16);
    assert!(heads.len() > 45);
}
