use super::super::audit;
use super::*;
use std::collections::BTreeSet;

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
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}; variant={variant}"));
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
    assert_eq!(backs, (0..6).collect());
    assert_eq!(bases, (0..3).collect());
}
