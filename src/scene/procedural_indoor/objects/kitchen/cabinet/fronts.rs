//! Construction details within the unchanged 28 mm front and motion envelope.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct FrontProgram {
    pub frame_fraction: Vec2,
    pub field_thickness_m: f32,
    pub raised_thickness_m: f32,
    pub flute_pitch_m: f32,
    pub flute_fill: f32,
    pub horizontal: bool,
    pub glazing_rows: usize,
    pub glazing_columns: usize,
    pub mullion_fraction: f32,
}
impl FrontProgram {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 0x46524f4e544a4f49);
        Self {
            frame_fraction: Vec2::new(r.random_range(0.055..0.19), r.random_range(0.055..0.20)),
            field_thickness_m: r.random_range(0.006..0.012),
            raised_thickness_m: r.random_range(0.024..0.028),
            flute_pitch_m: r.random_range(0.028..0.095),
            flute_fill: r.random_range(0.26..0.72),
            horizontal: r.random_bool(0.35),
            glazing_rows: r.random_range(1..=3),
            glazing_columns: r.random_range(1..=2),
            mullion_fraction: r.random_range(0.18..0.38),
        }
    }
    pub fn frame_widths(&self, size: Vec2) -> Vec2 {
        (self.frame_fraction * size.min_element())
            .clamp(Vec2::splat(0.018), Vec2::splat(0.065))
            .min(size * 0.22)
    }
}

pub(in super::super) fn front_panel(
    a: &mut Assembly,
    size: Vec2,
    tf: Transform,
    p: &KitchenProgram,
    label: &str,
) {
    let glass = p.door_style == 1;
    if p.front_profile == 0 && !glass {
        sheet(a, p.finish, label, Vec3::ZERO, size.extend(0.022), tf);
        return;
    }
    let edge = p.front.frame_widths(size);
    for side in [-1., 1.] {
        sheet(
            a,
            p.finish,
            label,
            Vec3::X * side * (size.x - edge.x) * 0.5,
            Vec3::new(edge.x, size.y, 0.022),
            tf,
        );
        sheet(
            a,
            p.finish,
            label,
            Vec3::Y * side * (size.y - edge.y) * 0.5,
            Vec3::new(size.x - edge.x * 2., edge.y, 0.022),
            tf,
        );
    }
    let inner = size - edge * 2.;
    if glass {
        let columns = p
            .front
            .glazing_columns
            .min((inner.x / 0.12).floor().max(1.) as usize);
        let rows = p
            .front
            .glazing_rows
            .min((inner.y / 0.15).floor().max(1.) as usize);
        let mullion = edge.min_element() * p.front.mullion_fraction;
        for i in 1..columns {
            sheet(
                a,
                p.finish,
                label,
                Vec3::X * inner.x * (i as f32 / columns as f32 - 0.5),
                Vec3::new(mullion, inner.y, 0.016),
                tf,
            );
        }
        for row in 0..rows {
            let cell_h = inner.y / rows as f32;
            for col in 0..columns {
                let cell_w = inner.x / columns as f32;
                let lo = Vec2::new(
                    -inner.x * 0.5
                        + col as f32 * cell_w
                        + if col > 0 {
                            mullion * 0.5 + 0.0005
                        } else {
                            0.001
                        },
                    -inner.y * 0.5
                        + row as f32 * cell_h
                        + if row > 0 {
                            mullion * 0.5 + 0.0005
                        } else {
                            0.001
                        },
                );
                let hi = Vec2::new(
                    -inner.x * 0.5 + (col + 1) as f32 * cell_w
                        - if col + 1 < columns {
                            mullion * 0.5 + 0.0005
                        } else {
                            0.001
                        },
                    -inner.y * 0.5 + (row + 1) as f32 * cell_h
                        - if row + 1 < rows {
                            mullion * 0.5 + 0.0005
                        } else {
                            0.001
                        },
                );
                sheet(
                    a,
                    Surface::GlassInterior,
                    label,
                    ((lo + hi) * 0.5).extend(0.),
                    (hi - lo).extend(0.004),
                    tf,
                );
            }
            if row > 0 {
                for col in 0..columns {
                    sheet(
                        a,
                        p.finish,
                        label,
                        Vec3::new(
                            inner.x * ((col as f32 + 0.5) / columns as f32 - 0.5),
                            inner.y * (row as f32 / rows as f32 - 0.5),
                            0.,
                        ),
                        Vec3::new(
                            inner.x / columns as f32 - if columns > 1 { mullion } else { 0. },
                            mullion,
                            0.016,
                        ),
                        tf,
                    );
                }
            }
        }
        return;
    }
    let inner = inner - Vec2::splat(0.002);
    let thickness = if p.front_profile == 2 {
        p.front.raised_thickness_m
    } else {
        p.front.field_thickness_m
    };
    sheet(a, p.finish, label, Vec3::ZERO, inner.extend(thickness), tf);
    if p.front_profile == 3 {
        let horizontal = p.front.horizontal;
        let span = if horizontal { inner.y } else { inner.x };
        let n = (span / p.front.flute_pitch_m).round().clamp(2., 32.) as usize;
        let pitch = span / n as f32;
        for i in 0..n {
            let offset = span * ((i as f32 + 0.5) / n as f32 - 0.5);
            let center = if horizontal {
                Vec3::new(0., offset, thickness * 0.5 + 0.002)
            } else {
                Vec3::new(offset, 0., thickness * 0.5 + 0.002)
            };
            let extent = if horizontal {
                Vec3::new(inner.x, pitch * p.front.flute_fill, 0.004)
            } else {
                Vec3::new(pitch * p.front.flute_fill, inner.y, 0.004)
            };
            sheet(a, p.finish, label, center, extent, tf);
        }
    }
}
fn sheet(a: &mut Assembly, surface: Surface, label: &str, center: Vec3, size: Vec3, tf: Transform) {
    let h = size.xy() * 0.5;
    a.part(surface, label).profile_slab(
        &[
            Vec2::new(-h.x, -h.y),
            Vec2::new(h.x, -h.y),
            Vec2::new(h.x, h.y),
            Vec2::new(-h.x, h.y),
        ],
        size.z,
        size.min_element().min(0.008) * 0.20,
        tf.mul_transform(
            Transform::from_translation(center).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        ),
    );
}

/// Hardware anchors on a real rail/stile, including narrow-framed glazing.
pub(in super::super) fn edge_pull(size: Vec2, p: &KitchenProgram, side: f32, bottom: bool) -> Vec2 {
    let framed = p.door_style == 1 || p.front_profile != 0;
    let edge = p.front.frame_widths(size);
    if bottom {
        Vec2::new(
            0.,
            -size.y * 0.5 + if framed { edge.y * 0.5 } else { 0.0525 },
        )
    } else {
        Vec2::new(
            -side * (size.x * 0.5 - if framed { edge.x * 0.5 } else { 0.0525 }),
            0.,
        )
    }
}
pub(in super::super) fn top_pull(size: Vec2, p: &KitchenProgram) -> Vec2 {
    if p.front_profile != 0 || p.door_style == 1 {
        Vec2::new(0., (size.y - p.front.frame_widths(size).y) * 0.5)
    } else {
        Vec2::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;
    #[test]
    fn front_variations_keep_closed_motion_envelope_and_valid_panes() {
        for seed in 0..64 {
            for size in [
                Vec2::new(0.30, 0.10),
                Vec2::new(0.44, 1.80),
                Vec2::new(1.10, 0.70),
            ] {
                let mut o = audit::fixture(ObjectKind::Pantry, seed);
                o.size = Vec3::new(size.x, size.y, 0.028);
                for profile in 0..4 {
                    for style in [0, 1] {
                        let mut p = parameters(&o);
                        p.front_profile = profile;
                        p.door_style = style;
                        let mut a = Assembly::default();
                        front_panel(
                            &mut a,
                            size,
                            Transform::from_xyz(0., size.y * 0.5, 0.),
                            &p,
                            "cabinet",
                        );
                        audit::check(&o, &a).unwrap();
                        let (lo, hi) = a.bounds();
                        assert!(
                            lo.z >= -0.01401 && hi.z <= 0.01401,
                            "{seed}/{style}/{profile}: {lo:?}..{hi:?}"
                        );
                        if style == 1 {
                            assert!(a
                                .parts
                                .contains_key(&(Surface::GlassInterior, "cabinet".into())));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn hardware_anchors_touch_the_actual_front_triangles() {
        for seed in 0..64 {
            let o = audit::fixture(ObjectKind::Pantry, seed);
            for size in [
                Vec2::new(0.28, 0.12),
                Vec2::new(0.44, 1.8),
                Vec2::new(1.10, 0.70),
            ] {
                for profile in 0..4 {
                    for style in [0, 1] {
                        let mut p = parameters(&o);
                        p.front_profile = profile;
                        p.door_style = style;
                        let mut a = Assembly::default();
                        front_panel(&mut a, size, Transform::IDENTITY, &p, "cabinet");
                        let mut points = vec![top_pull(size, &p)];
                        if size.y > 0.2 {
                            points.extend([
                                edge_pull(size, &p, -1., false),
                                edge_pull(size, &p, 1., false),
                                edge_pull(size, &p, 1., true),
                            ]);
                        }
                        for point in points {
                            let attached = a
                                .parts
                                .iter()
                                .filter(|((surface, _), _)| *surface != Surface::GlassInterior)
                                .any(|(_, g)| {
                                    g.indices.as_chunks::<3>().0.iter().any(|t| {
                                        let v =
                                            t.map(|i| Vec3::from_array(g.positions[i as usize]));
                                        if v.iter().any(|p| (p.z - 0.011).abs() > 1e-5) {
                                            return false;
                                        }
                                        let [a, b, c] = v.map(|v| v.xy());
                                        let det = (b - a).perp_dot(c - a);
                                        if det.abs() < 1e-9 {
                                            return false;
                                        }
                                        let u = (point - a).perp_dot(c - a) / det;
                                        let v = (b - a).perp_dot(point - a) / det;
                                        u >= -1e-5 && v >= -1e-5 && u + v <= 1. + 1e-5
                                    })
                                });
                            assert!(attached,"unsupported pull {seed}/{profile}/{style} at {point:?} for {size:?}");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn profiles_cover_independent_rail_widths_reed_pitches_and_axes() {
        use std::collections::BTreeSet;
        let mut widths = BTreeSet::new();
        let mut pitches = BTreeSet::new();
        let mut axes = [0; 2];
        let mut panes = BTreeSet::new();
        for seed in 0..256 {
            let p = FrontProgram::sample(seed);
            widths.insert((
                (p.frame_fraction.x * 100.) as u32,
                (p.frame_fraction.y * 100.) as u32,
            ));
            pitches.insert((p.flute_pitch_m * 1000.) as u32);
            axes[usize::from(p.horizontal)] += 1;
            panes.insert((p.glazing_rows, p.glazing_columns));
        }
        assert!(widths.len() > 100 && pitches.len() > 55);
        assert!(axes.into_iter().all(|n| n > 55));
        assert_eq!(panes.len(), 6);
    }
}
