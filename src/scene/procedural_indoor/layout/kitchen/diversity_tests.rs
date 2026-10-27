use super::*;
use crate::scene::procedural_indoor::validation;
use std::collections::BTreeSet;
mod qualification;

#[test]
fn kitchen_placement_breaks_fixed_order_and_centering_without_losing_clearances() {
    let mut first = BTreeSet::new();
    let mut gaps = BTreeSet::new();
    let mut shifts = BTreeSet::new();
    let mut rooms = 0;
    for layout in [
        IndoorLayout::Kitchen,
        IndoorLayout::Kitchenette,
        IndoorLayout::Cafe,
        IndoorLayout::Mixed,
    ] {
        for seed in 0..12 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 3, 0.15)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            validation::validate_layout(&scene).unwrap();
            if seed < 4 {
                validation::validate_geometry(&scene).unwrap();
            }
            for group in &scene.assembly_diagnostics.kitchens {
                let p = group.sampling.as_ref().unwrap();
                assert!(p.valid());
                first.insert(format!("{:?}", scene.objects[group.units[0]].kind));
                gaps.insert((p.unit_gap_m * 10000.).round() as i32);
                shifts.insert((p.run_offset_m * 100.).round() as i32);
                rooms += 1;
            }
        }
    }
    assert!(rooms >= 24);
    assert!(first.len() >= 3, "required order: {first:?}");
    // The four profiles intentionally reuse twelve seeds: this independent
    // installation stream should produce twelve, not forty-eight, products.
    assert!(gaps.len() >= 10, "gap bins: {}", gaps.len());
    assert!(shifts.len() > 16);
}

#[test]
fn continuous_kitchen_proposals_are_bounded_replayable_and_have_no_center_atom() {
    let mut positions = BTreeSet::new();
    for seed in 0..256 {
        let mut r = stream(seed, 0x4b49544348504c41);
        let a = sampling::run_shifts(8., 3., &mut r);
        let b = sampling::island_shifts(&mut r);
        assert!(a.iter().all(|&x| x.abs() <= 2.075 && x != 0.));
        assert!(b.iter().all(|x| x.length() < 1.051 && x.length() > 0.059));
        positions.insert((a[0] * 1000.).round() as i32);
        let mut replay = stream(seed, 0x4b49544348504c41);
        assert_eq!(a, sampling::run_shifts(8., 3., &mut replay));
        assert_eq!(b, sampling::island_shifts(&mut replay));
    }
    assert!(positions.len() > 220);
}
