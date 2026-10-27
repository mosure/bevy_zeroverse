//! Unequal acoustic, glazed and slatted panels with dimensioned stable supports.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DividerProgram {
    pub weights: Vec<f32>,
    pub bottom: f32,
    pub top: f32,
    pub panel_depth: f32,
    pub border: f32,
    pub stand_span: f32,
    pub base: u8,
    pub frame: Surface,
    pub panel: Surface,
    pub slats: bool,
    pub slat_pitch: f32,
    pub slat_fraction: f32,
    pub slat_angle: f32,
    pub lower_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> DividerProgram {
    let mut r = stream(o.seed, 0x4449564944455250);
    let n = r.random_range(1..=3);
    DividerProgram {
        weights: (0..n).map(|_| r.random_range(0.60..1.45)).collect(),
        bottom: o.size.y * r.random_range(0.07..0.31),
        top: o.size.y * r.random_range(0.89..1.0),
        panel_depth: r.random_range(0.018_f32..0.048).min(o.size.z * 0.24),
        border: r.random_range(0.016_f32..0.035).min(o.size.x * 0.024),
        stand_span: r.random_range(0.27..0.41),
        base: r.random_range(0..3),
        frame: [Surface::Wood, Surface::Metal, Surface::Chrome][r.random_range(0..3)],
        panel: [
            Surface::Fabric,
            Surface::FabricAlt,
            Surface::Wood,
            Surface::GlassInterior,
        ][r.random_range(0..4)],
        slats: r.random_bool(0.36),
        slat_pitch: r.random_range(0.045..0.115),
        slat_fraction: r.random_range(0.30..0.73),
        slat_angle: r.random_range(-0.30..0.30),
        lower_fraction: r.random_range(0.24..0.61),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let foot_h = 0.024_f32.min(s.y * 0.04);
    for side in [-1., 1.] {
        let x = side * s.x * p.stand_span;
        if p.base == 0 {
            a.box_part(
                p.frame,
                l,
                Vec3::new(x, foot_h * 0.5, 0.),
                Vec3::new(s.x.min(0.6) * 0.12, foot_h, s.z * 0.98),
                foot_h * 0.2,
            );
        } else if p.base == 1 {
            let r = s.z * 0.43;
            let rx = (r * 0.68).min(s.x * (0.5 - p.stand_span) - 0.002);
            a.part(p.frame, l).housing(
                &[
                    (0., Vec2::new(rx, r)),
                    (foot_h, Vec2::new(rx * 0.86, r * 0.88)),
                ],
                2.4,
                Transform::from_xyz(x, 0., 0.),
            );
        } else {
            for z in [-1., 1.] {
                a.part(p.frame, l).rod(
                    Vec3::new(x, foot_h * 0.5, z * (s.z * 0.5 - foot_h)),
                    Vec3::new(x, p.bottom * 0.7, 0.),
                    foot_h * 0.40,
                );
                a.box_part(
                    Surface::Rubber,
                    l,
                    Vec3::new(x, foot_h * 0.25, z * (s.z * 0.5 - foot_h)),
                    Vec3::new(foot_h, foot_h * 0.5, foot_h),
                    foot_h * 0.10,
                );
            }
        }
        a.part(p.frame, l).rod(
            Vec3::new(x, foot_h, 0.),
            Vec3::new(x, p.top - p.border, 0.),
            p.border * 0.48,
        );
    }
    let width = s.x * 0.97;
    let total: f32 = p.weights.iter().sum();
    let mut left = -width * 0.5;
    for w in &p.weights {
        let w = width * w / total;
        let x = left + w * 0.5;
        let inner_w = w - p.border * 2.;
        let height = p.top - p.bottom;
        for sign in [-1., 1.] {
            a.box_part(
                p.frame,
                l,
                Vec3::new(
                    x + sign * (w - p.border) * 0.5,
                    (p.top + p.bottom) * 0.5,
                    0.,
                ),
                Vec3::new(p.border, height, p.panel_depth + 0.008),
                0.002,
            );
            a.box_part(
                p.frame,
                l,
                Vec3::new(
                    x,
                    if sign < 0. {
                        p.bottom + p.border * 0.5
                    } else {
                        p.top - p.border * 0.5
                    },
                    0.,
                ),
                Vec3::new(w - p.border * 2., p.border, p.panel_depth + 0.008),
                0.002,
            );
        }
        let lo = p.bottom + p.border;
        let hi = p.top - p.border;
        if p.slats {
            let lower = (hi - lo) * p.lower_fraction;
            a.box_part(
                p.panel,
                l,
                Vec3::new(x, lo + lower * 0.5, 0.),
                Vec3::new(inner_w, lower, p.panel_depth),
                0.002,
            );
            let n = (inner_w / p.slat_pitch).round().clamp(2., 24.) as usize;
            let pitch = inner_w / n as f32;
            let slat_w = pitch * p.slat_fraction;
            let slat_d = p.panel_depth * 0.55;
            for i in 0..n {
                a.part(p.frame, l).cuboid(
                    Vec3::new(slat_w, hi - lo - lower, slat_d),
                    0.001,
                    Transform::from_xyz(
                        x - inner_w * 0.5 + pitch * (i as f32 + 0.5),
                        (hi + lo + lower) * 0.5,
                        0.,
                    )
                    .with_rotation(Quat::from_rotation_y(p.slat_angle)),
                );
            }
        } else {
            a.box_part(
                p.panel,
                l,
                Vec3::new(x, (hi + lo) * 0.5, 0.),
                Vec3::new(inner_w, hi - lo, p.panel_depth),
                0.002,
            );
        }
        left += w;
    }
}
