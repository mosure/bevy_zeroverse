use super::super::audit;
use super::*;
use std::collections::BTreeSet;

#[test]
fn audience_seats_fit_short_and_deep_metric_envelopes_including_folded_states() {
    let mut folded = 0;
    for seed in 0..128 {
        for variant in [10, 11] {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = variant;
            o.size = Vec3::new(
                0.65 + (seed % 7) as f32 * 0.02,
                0.84 + (seed % 11) as f32 * 0.025,
                0.56 + (seed % 13) as f32 * 0.022,
            );
            folded += usize::from(is_folded(&o));
            let a = super::super::build_object(&o);
            audit::check(&o, &a)
                .unwrap_or_else(|e| panic!("{e}; variant={variant}; size={:?}", o.size));
            let boxes = clearance_boxes(&o);
            for p in a.parts.values().flat_map(|g| &g.positions) {
                let p = Vec3::from_array(*p);
                assert!(
                    boxes
                        .iter()
                        .any(|(lo, hi)| p.cmpge(*lo - Vec3::splat(0.003)).all()
                            && p.cmple(*hi + Vec3::splat(0.003)).all()),
                    "{variant}/{seed}: {p}"
                );
            }
        }
    }
    assert!(folded > 12);
}

#[test]
fn chair_and_stool_programs_fit_clearances_replay_and_keep_seating_datum() {
    let mut backs = BTreeSet::new();
    let mut bases = BTreeSet::new();
    for variant in [0, 1, 2, 3, 4, 5, 6, 7, 9] {
        for seed in 0..96 {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = variant;
            o.size.y = if is_backless(&o) {
                0.48
            } else if variant == 7 {
                0.79
            } else if variant <= 4 {
                0.84 + seed as f32 / 96. * 0.28
            } else {
                1.35
            };
            let a = super::super::build_object(&o);
            let check = audit::check(&o, &a)
                .unwrap_or_else(|e| panic!("{e}; variant={variant}; {:?}", parameters(&o)));
            assert!(
                check.min_y_m <= 0.003,
                "floor contact {variant}/{seed}: {check:?}"
            );
            let boxes = clearance_boxes(&o);
            for point in a.parts.values().flat_map(|g| &g.positions) {
                let point = Vec3::from_array(*point);
                assert!(
                    boxes
                        .iter()
                        .any(|(lo, hi)| point.cmpge(*lo - Vec3::splat(0.003)).all()
                            && point.cmple(*hi + Vec3::splat(0.003)).all()),
                    "{variant}/{seed}: {point}"
                );
            }
            if variant == 6 || variant == 7 {
                bases.insert(stool::parameters(&o).base);
                // Top centre remains at the human pelvis datum (upholstery can
                // have a shallow dish); it must not vary with the base program.
                assert!(a
                    .parts
                    .values()
                    .flat_map(|g| &g.positions)
                    .any(|p| p[0].abs() < 0.02
                        && p[2].abs() < 0.02
                        && (p[1] - 0.47).abs() < 0.006));
            } else {
                backs.insert(parameters(&o).back_construction);
            }
            if seed < 8 {
                let b = super::super::build_object(&o);
                for (key, g) in &a.parts {
                    assert_eq!(g.positions, b.parts[key].positions);
                    assert_eq!(g.indices, b.parts[key].indices);
                    assert_eq!(g.normals, b.parts[key].normals);
                }
            }
        }
    }
    assert_eq!(backs, (0..10).collect());
    assert_eq!(bases, (0..3).collect());
}

#[test]
fn base_and_seat_choices_change_actual_surfaces() {
    let o = audit::fixture(ObjectKind::Chair, 17);
    let p = parameters(&o);
    let mut bases = Vec::new();
    for style in 0..8 {
        let mut p = p.clone();
        p.base_style = style;
        p.casters = false;
        let mut a = Assembly::default();
        base::build(&mut a, &o, &p, 0.48, 0.46);
        audit::check(&o, &a).unwrap();
        bases.push(audit::descriptor(&o, &a));
    }
    assert_eq!(audit::representatives(&bases, 0.01).len(), 8);
    let mut seats = Vec::new();
    for style in 0..3 {
        let mut p = p.clone();
        p.seat_style = style;
        let mut a = Assembly::default();
        upper::seat(&mut a, &p, 0.48, 0.46, 0.06);
        audit::check(&o, &a).unwrap();
        seats.push(audit::descriptor(&o, &a));
    }
    assert_eq!(audit::representatives(&seats, 0.002).len(), 3);
}

#[test]
fn structural_controls_are_realized_and_mechanically_compatible() {
    let mut bases = BTreeSet::new();
    let mut arms = BTreeSet::new();
    let mut seats = BTreeSet::new();
    let mut headrests = BTreeSet::new();
    for variant in [0, 1, 2, 3, 4, 5, 9] {
        for seed in 0..128 {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = variant;
            let p = parameters(&o);
            assert!(!p.casters || p.base_style < 3);
            assert_eq!(p.base_style < 4, matches!(variant, 0 | 1 | 5 | 9));
            assert!(p.back_construction != 1 || variant == 3);
            bases.insert(p.base_style);
            if p.armrests {
                arms.insert(p.arm_support);
            }
            seats.insert(p.seat_style);
            if p.headrest {
                headrests.insert(p.headrest_style);
            }
        }
    }
    assert_eq!(bases, (0..8).collect());
    assert_eq!(arms, (0..6).collect());
    assert_eq!(seats, (0..3).collect());
    assert_eq!(headrests, (0..3).collect());
}
