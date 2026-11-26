use super::*;
use crate::scene::procedural_indoor::envelope::ray_triangle;

fn ray(a: &Assembly, origin: Vec3, dir: Vec3) -> Option<(Surface, f32)> {
    a.parts
        .iter()
        .flat_map(|((surface, label), g)| {
            assert_eq!(label, "other_structure");
            g.indices.as_chunks::<3>().0.iter().filter_map(move |t| {
                ray_triangle(
                    origin,
                    dir,
                    t.map(|i| Vec3::from_array(g.positions[i as usize])),
                )
                .map(|distance| (*surface, distance))
            })
        })
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
}

#[test]
fn both_gaskets_and_jaws_bear_on_real_panel_faces_at_each_post() {
    let mut types = [0; 4];
    for seed in 0..96 {
        let section = GuardSection::sample(seed);
        let Some(stock) = section.panel_mount else {
            continue;
        };
        let clips = matches!(stock.retainer, Retainer::EdgeClips { .. });
        types[2 * usize::from(section.infill == Infill::Solid) + usize::from(clips)] += 1;
        let pan = if section.infill == Infill::Glass {
            Surface::GlassInterior
        } else {
            section.panel
        };
        let axis = Quat::from_rotation_y(seed as f32 * 0.7) * Vec3::X;
        for slope in [-0.65, 0., 0.61] {
            // Include short residual bays at the deck/flight junction.
            for length in [0.13, 1.03] {
                let start = Vec3::new(-2., 0.8, 1.2);
                let end = start + (axis + Vec3::Y * slope) * length;
                let height = 1.12;
                let rise = if slope == 0. { 0. } else { 0.16 };
                let bottom = section.bottom_gap + rise;
                let depth = if section.round_handrail {
                    section.handrail_width
                } else {
                    section.handrail_depth
                };
                let top = height - depth * 0.5 * (1. + slope * slope).sqrt() - 0.015;
                let inset = section.post_width * 0.5 + stock.edge_clearance_m;
                let bite = stock.bite_m.min((length - 2. * inset) * 0.24);
                let half = section.panel_thickness * 0.5;
                let mut a = Assembly::default();
                stock.bay(&mut a, &section, start, end, rise, height);
                if let Retainer::PostChannel { end_setback_m } = stock.retainer {
                    let pads = &a.parts[&(Surface::Rubber, "other_structure".into())];
                    for point in &pads.positions {
                        let offset = Vec3::from(*point) - start;
                        let above_rake = offset.y - slope * offset.dot(axis);
                        let error = (above_rake - bottom - end_setback_m)
                            .abs()
                            .min((above_rake - top + end_setback_m).abs());
                        assert!(error < 2e-5, "channel end cut crosses the panel edge");
                    }
                }
                let normal = axis.cross(Vec3::Y);
                for (anchor, direction) in [(start, 1.), (end, -1.)] {
                    let edge = anchor + direction * inset * (axis + Vec3::Y * slope);
                    let hit = |x, y, side: f32, expected, distance: f32| {
                        let point = edge + axis * (direction * x) + Vec3::Y * y;
                        let (material, d) =
                            ray(&a, point + normal * (side * 0.08), -normal * side).unwrap();
                        assert_eq!(material, expected, "seed {seed} x={x} y={y}");
                        assert!(
                            (0.08 - d - distance).abs() < 2e-5,
                            "seed {seed}, depth={d}, x={x}"
                        );
                    };
                    let levels = match stock.retainer {
                        Retainer::EdgeClips {
                            lower_fraction,
                            upper_fraction,
                            ..
                        } => [lower_fraction, upper_fraction],
                        Retainer::PostChannel { .. } => [0.15, 0.84],
                    };
                    for side in [-1., 1.] {
                        for f in levels {
                            let y = bottom + (top - bottom) * f;
                            hit(
                                bite * 0.35,
                                y,
                                side,
                                stock.finish,
                                half + stock.gasket_m + stock.jaw_wall_m,
                            );
                            hit(
                                bite + (bite * 0.18).min(0.0018) * 0.5,
                                y,
                                side,
                                Surface::Rubber,
                                half + stock.gasket_m,
                            );
                        }
                        if clips {
                            hit(bite * 0.35, (top + bottom) * 0.5, side, pan, half);
                        }
                        hit(
                            (length - 2. * inset) * 0.48,
                            (top + bottom) * 0.5,
                            side,
                            pan,
                            half,
                        );
                    }
                }
            }
        }
    }
    assert!(types.iter().all(|&n| n >= 3), "{types:?}");
}

#[test]
fn retention_is_recorded_validated_and_omitted_from_archived_and_open_guards() {
    for seed in 0..64 {
        let mut s = GuardSection::sample(seed);
        s.validate().unwrap();
        assert_eq!(
            s.panel_mount.is_some(),
            matches!(s.infill, Infill::Glass | Infill::Solid)
        );
        let mut value = serde_json::to_value(&s).unwrap();
        assert_eq!(s, serde_json::from_value(value.clone()).unwrap());
        value.as_object_mut().unwrap().remove("panel_mount");
        let archived: GuardSection = serde_json::from_value(value).unwrap();
        archived.validate().unwrap();
        assert!(archived.panel_mount.is_none());
        if let Some(m) = s.panel_mount.as_mut() {
            m.bite_m = f32::NAN;
            assert!(s.validate().is_err());
        }
    }
}
