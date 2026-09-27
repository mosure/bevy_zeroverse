//! Continuous convex top outlines and independent structural support programs.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableProgram {
    pub top_thickness: f32,
    pub edge_bevel: f32,
    pub leg_radius: f32,
    pub leg_inset: f32,
    pub leg_rake: f32,
    /// 2 = ellipse/circle; high exponents approach rounded rectangles/trapezoids.
    pub outline_exponent: f32,
    pub taper: f32,
    pub support: u8,
    pub top_surface: Surface,
    pub pedestal_radius: f32,
    pub pedestal_base_radius: f32,
    pub pedestal_base_aspect: f32,
}
pub fn parameters(o: &IndoorObject) -> TableProgram {
    let mut rng = stream(o.seed, 172);
    let exponent = if o.kind == ObjectKind::Desk {
        rng.random_range(5.0..24.0)
    } else if rng.random_bool(0.4) {
        2.0
    } else {
        rng.random_range(2.2..24.0)
    };
    TableProgram {
        top_thickness: rng.random_range(0.026..0.052),
        edge_bevel: rng.random_range(0.003..0.012),
        leg_radius: rng.random_range(0.018..0.036),
        leg_inset: rng.random_range(0.11..0.18),
        leg_rake: rng.random_range(0.0..0.075),
        outline_exponent: exponent,
        taper: if exponent > 5.0 && rng.random_bool(0.45) {
            rng.random_range(-0.42..0.42)
        } else {
            0.0
        },
        support: rng.random_range(0..6),
        top_surface: match rng.random_range(0..10) {
            0 => Surface::Ceramic,
            1 => Surface::Plastic,
            2 => Surface::Concrete,
            _ => Surface::Wood,
        },
        pedestal_radius: rng.random_range(0.045..0.085),
        pedestal_base_radius: rng.random_range(0.24..0.36),
        pedestal_base_aspect: rng.random_range(0.72..1.0),
    }
}
pub fn outline(o: &IndoorObject) -> Vec<Vec2> {
    let p = parameters(o);
    (0..64)
        .map(|i| {
            let a = i as f32 * TAU / 64.0;
            let f = |x: f32| {
                if x.abs() < 1e-6 {
                    0.0
                } else {
                    x.signum() * x.abs().powf(2.0 / p.outline_exponent)
                }
            };
            let x = f(a.cos());
            let z = f(a.sin());
            Vec2::new(
                x * o.size.x * 0.5 * (1.0 + p.taper * z) / (1.0 + p.taper.abs()),
                z * o.size.z * 0.5,
            )
        })
        .collect()
}
/// Require every prop corner to sit on the actual surface, not its rectangular bounds.
pub fn supports(o: &IndoorObject, lo: Vec2, hi: Vec2, margin: f32) -> bool {
    let outline = outline(o);
    [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
        .iter()
        .all(|p| {
            (0..outline.len()).all(|i| {
                let a = outline[i];
                let edge = outline[(i + 1) % outline.len()] - a;
                edge.perp_dot(*p - a) >= (margin + 0.012) * edge.length()
            })
        })
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = parameters(o);
    let label = o.kind.class_name();
    let outline = outline(o);
    let underside = s.y - p.top_thickness;
    a.part(p.top_surface, label).profile_slab(
        &outline,
        p.top_thickness,
        p.edge_bevel,
        Transform::from_xyz(0.0, s.y - p.top_thickness * 0.5, 0.0),
    );
    if s.x > s.z && p.top_surface == Surface::Wood {
        for uv in &mut a.part(Surface::Wood, label).uvs {
            *uv = [uv[1], -uv[0]];
        }
    }
    let edge: Vec<_> = outline.iter().map(|p| *p * 0.985).collect();
    a.part(Surface::WoodEdge, label).profile_slab(
        &edge,
        0.012,
        0.003,
        Transform::from_xyz(0.0, underside - 0.006, 0.0),
    );
    let underside = underside - 0.012;
    let frame = if o.variant == 1 {
        Surface::WoodEdge
    } else {
        Surface::Metal
    };
    let half = Vec2::new(s.x, s.z) * 0.5;
    let anchor = Vec2::new(
        (half.x - p.leg_inset).max(0.12),
        (half.y - p.leg_inset).max(0.12),
    ) * 0.68;
    match p.support {
        0 | 1 => {
            for x in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let top = Vec3::new(
                        x * anchor.x,
                        underside,
                        z * anchor.y * if p.support == 1 { 0.38 } else { 1.0 },
                    );
                    let foot = Vec3::new(
                        x * (anchor.x + p.leg_rake),
                        p.leg_radius + 0.008,
                        z * (anchor.y + p.leg_rake),
                    );
                    a.part(frame, label).rod(foot, top, p.leg_radius);
                    a.part(Surface::Rubber, label).cylinder(
                        p.leg_radius + 0.003,
                        p.leg_radius + 0.01,
                        Transform::from_translation(foot.with_y((p.leg_radius + 0.01) * 0.5)),
                    );
                }
            }
            for x in [-1.0, 1.0] {
                a.part(frame, label).rod(
                    Vec3::new(x * anchor.x, underside - 0.03, -anchor.y),
                    Vec3::new(x * anchor.x, underside - 0.03, anchor.y),
                    0.024,
                );
                if p.support == 1 {
                    a.part(frame, label).rod(
                        Vec3::new(x * anchor.x, 0.22, -anchor.y * 0.7),
                        Vec3::new(x * anchor.x, 0.22, anchor.y * 0.7),
                        0.018,
                    );
                }
            }
        }
        2 => {
            let long_x = s.x > s.z;
            let ratio = s.x.max(s.z) / s.x.min(s.z);
            let offsets = if ratio > 1.8 {
                vec![-0.25, 0.25]
            } else {
                vec![0.0]
            };
            for offset in offsets {
                let center = if long_x {
                    Vec3::X * s.x * offset
                } else {
                    Vec3::Z * s.z * offset
                };
                a.part(frame, label).cylinder(
                    s.x.min(s.z) * p.pedestal_radius,
                    underside - 0.04,
                    Transform::from_translation(center + Vec3::Y * (underside + 0.04) * 0.5),
                );
                a.part(frame, label).cylinder(
                    s.x.min(s.z) * p.pedestal_base_radius,
                    0.035,
                    Transform::from_translation(center + Vec3::Y * 0.0175).with_scale(Vec3::new(
                        1.0,
                        1.0,
                        p.pedestal_base_aspect,
                    )),
                );
            }
        }
        3 => {
            for x in [-anchor.x, anchor.x] {
                for z in [-anchor.y, anchor.y] {
                    a.part(frame, label).rod(
                        Vec3::new(x, p.leg_radius + 0.007, z),
                        Vec3::new(x, underside, z),
                        p.leg_radius,
                    );
                }
                for y in [p.leg_radius + 0.007, underside] {
                    a.part(frame, label).rod(
                        Vec3::new(x, y, -anchor.y),
                        Vec3::new(x, y, anchor.y),
                        p.leg_radius,
                    );
                }
            }
        }
        4 => {
            for z in [-anchor.y, anchor.y] {
                for sign in [-1.0, 1.0] {
                    a.part(frame, label).rod(
                        Vec3::new(sign * anchor.x, p.leg_radius + 0.01, z),
                        Vec3::new(-sign * anchor.x, underside, z),
                        p.leg_radius,
                    );
                }
            }
        }
        _ => {
            // End panels leave an open knee bay; width, rake and thickness
            // remain independent of the top outline and finish.
            for x in [-anchor.x, anchor.x] {
                a.part(frame, label).cuboid(
                    Vec3::new(p.leg_radius * 2.0, underside - 0.016, anchor.y * 2.0),
                    0.004,
                    Transform::from_xyz(x, (underside + 0.016) * 0.5, 0.0),
                );
            }
        }
    }
    if p.support >= 3 {
        for x in [-anchor.x, anchor.x] {
            for z in [-anchor.y, anchor.y] {
                a.part(Surface::Rubber, label).cylinder(
                    p.leg_radius + 0.005,
                    0.016,
                    Transform::from_xyz(x, 0.008, z),
                );
            }
        }
    }
    a.part(frame, label).rod(
        Vec3::new(-anchor.x, underside - 0.035, 0.0),
        Vec3::new(anchor.x, underside - 0.035, 0.0),
        0.024,
    );
    if o.kind == ObjectKind::Desk || (o.kind == ObjectKind::Table && p.outline_exponent > 4.0) {
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0.0, underside - 0.07, 0.0),
            Vec3::new(s.x * 0.35, 0.045, 0.16),
            0.005,
        );
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(0.0, s.y + 0.001, -s.z * 0.20),
            Vec3::new(0.16, 0.002, 0.055),
            0.0,
        );
    }
}
