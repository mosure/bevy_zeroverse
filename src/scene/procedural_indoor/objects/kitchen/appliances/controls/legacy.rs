//! Front panels of microwave manifests through generator 226.
use super::*;

pub(in super::super) fn microwave(
    a: &mut Assembly,
    o: &IndoorObject,
    fraction: f32,
    front: f32,
    lower: bool,
    sign: f32,
) {
    let s = o.size;
    let w = s.x - 0.018;
    let label = o.kind.class_name();
    let center = if lower {
        Vec3::new(0., s.y * 0.17, front + 0.016)
    } else {
        Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.51, front + 0.016)
    };
    let panel_size = if lower {
        Vec3::new(w * 0.96, s.y * 0.21, 0.025)
    } else {
        Vec3::new(w * fraction - 0.006, s.y * 0.86, 0.025)
    };
    a.box_part(Surface::Plastic, label, center, panel_size, 0.003);
    if lower {
        knobs(
            a,
            center + Vec3::new(-w * 0.29, 0., 0.021),
            Vec3::X * w * 0.18,
            3,
            0.012,
            label,
            o.seed,
        );
    } else {
        knobs(
            a,
            Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.24, front + 0.037),
            Vec3::Y * s.y * 0.19,
            2,
            0.016,
            label,
            o.seed,
        );
    }
    let screen_center = if lower {
        center + Vec3::new(w * 0.30, 0., 0.014)
    } else {
        Vec3::new(sign * w * (1. - fraction) * 0.5, s.y * 0.77, front + 0.030)
    };
    computers::displays::screen_cover(
        a,
        Surface::Metal,
        Surface::Screen,
        label,
        Vec3::new(w * 0.12, s.y * 0.11, 0.002),
        0.,
        Transform::from_translation(screen_center),
    );
}
