use super::super::audit;
use super::*;
use std::collections::BTreeSet;

/// Vertical intersections with the actual triangles in a set of material batches.
pub(super) fn vertical_intersections<'a>(
    geometry: impl IntoIterator<Item = &'a Geometry>,
    xz: Vec2,
) -> Vec<f32> {
    let mut hits = Vec::new();
    for g in geometry {
        for tri in g.indices.as_chunks::<3>().0 {
            let [a, b, c] =
                [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
            let (e, f, q) = (b.xz() - a.xz(), c.xz() - a.xz(), xz - a.xz());
            let det = e.perp_dot(f);
            if det.abs() < 1e-11 {
                continue;
            }
            let (u, v) = (q.perp_dot(f) / det, e.perp_dot(q) / det);
            if u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001 {
                hits.push(a.y + u * (b.y - a.y) + v * (c.y - a.y));
            }
        }
    }
    hits.sort_by(f32::total_cmp);
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
    hits
}

#[test]
fn compact_upholstered_shells_keep_the_rear_skin_outside_the_lumbar_bend() {
    // The fully built adjoining-room chair in mixed seed 31248 inverted its
    // rear skin. Exercise the actual instance and nearby valid product sizes.
    let mut o = audit::fixture(ObjectKind::Chair, 4_589_592_075_449_161_286);
    o.variant = 2;
    for height in [0.77, 0.78641623, 0.81, 0.87] {
        for width in [0.62, 0.69471407, 0.74] {
            o.size = Vec3::new(width, height, 0.66443765);
            let p = parameters(&o);
            assert_eq!(p.back_construction, 2);
            assert!(p.shell_thickness > 0.04 && p.lumbar > 0.048);
            let assembly = super::super::build_object(&o);
            audit::check(&o, &assembly).unwrap();
        }
    }
}

#[test]
fn purchasing_a_set_preserves_matching_seats_and_allows_replacements() {
    for family in [0, 2, 6] {
        let mut counts = std::collections::BTreeMap::new();
        let mut independent = BTreeSet::new();
        for instance in 0..256 {
            let size = footprint(family, 70, instance);
            *counts.entry(size.to_array().map(f32::to_bits)).or_insert(0) += 1;
            independent.insert(
                footprint(family, instance, instance)
                    .to_array()
                    .map(f32::to_bits),
            );
        }
        let most_matched = *counts.values().max().unwrap();
        assert!((180..245).contains(&most_matched));
        assert!(independent.len() > 250);
    }
}

#[test]
fn rounded_arm_carriers_keep_valid_bend_radii_at_short_pad_extremes() {
    let o = audit::fixture(ObjectKind::Chair, 69_430_991);
    for style in [1, 4] {
        for height in [0.61, 0.655, 0.70] {
            for length in [0.16, 0.20, 0.33] {
                for width in [0.40, 0.48, 0.52] {
                    let mut p = parameters(&o);
                    p.armrests = true;
                    p.arm_support = style;
                    p.arm_height = height;
                    p.arm_pad_length = length;
                    let mut a = Assembly::default();
                    upper::arms(&mut a, &p, width, 2);
                    audit::check(&o, &a).unwrap_or_else(|e| {
                        panic!(
                            "style={style}, height={height}, length={length}, width={width}: {e}"
                        )
                    });
                }
            }
        }
    }
}

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
            let size = footprint(variant, seed / 4, seed);
            o.size.x = size.x;
            o.size.z = size.y;
            o.size.y = if matches!(variant, 6 | 7) {
                stool::fitted_envelope(&o, None)
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
                        && (p[1] - stool::parameters(&o).seat_height_m).abs() < 0.006));
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
        base::build(
            &mut a,
            &o,
            &p,
            0.48,
            0.46,
            0.39,
            super::super::super::layout::GENERATOR_VERSION,
        );
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
    assert_eq!(seats, (0..6).collect());
    assert_eq!(headrests, (0..3).collect());
}

#[test]
fn swivel_controls_move_with_the_pan_while_casters_stay_on_the_base() {
    use crate::scene::procedural_indoor::{motion::IndoorMotionConfig, objects};
    let o = (0..128)
        .find_map(|seed| {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = 0;
            parameters(&o).casters.then_some(o)
        })
        .unwrap();
    let policy = IndoorMotionConfig {
        chair_fraction: 1.,
        ..default()
    };
    let a = objects::build_object_with_motion(&o, Some(&policy));
    let upper = a.joints.iter().find(|j| j.angle != 0.).unwrap();
    let base = a.joints.iter().find(|j| j.angle == 0.).unwrap();
    let p = parameters(&o);
    let width = (o.size.x * p.seat_width_fraction).min(0.56);
    let depth = (o.size.z * p.seat_depth_fraction).min(0.55);
    let mut r = stream(o.seed, 70);
    let _ = r.random_range(0.43..0.52_f32);
    let _ = r.random_range(0.42..0.49_f32);
    let padding = r.random_range(0.044..0.075);
    let mut lever = Assembly::default();
    base::tilt_control(&mut lever, width, depth, 0.47 - padding - 0.023);
    for ((surface, _), geometry) in lever.parts {
        assert!(upper
            .parts
            .iter()
            .filter(|(s, _)| *s == surface)
            .any(|key| {
                a.parts[key]
                    .positions
                    .windows(geometry.positions.len())
                    .any(|v| v == geometry.positions)
            }));
    }
    assert!(base.parts.iter().any(|(s, _)| *s == Surface::Rubber));
    assert!(!upper.parts.iter().any(|(s, _)| *s == Surface::Rubber));
}
