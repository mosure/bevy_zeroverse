//! Dimensioned service furniture. Contact, tray tops and outer bounds are exact.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ServiceProgram {
    pub levels: usize,
    pub drawers: usize,
    pub frame: Surface,
    pub upholstery: Surface,
    pub roundness: f32,
    pub feet_height: f32,
    pub slats: bool,
    pub backed: bool,
    pub panel_fraction: f32,
    pub drawer_bias: f32,
    pub handle_fraction: f32,
}

pub fn parameters(o: &IndoorObject) -> ServiceProgram {
    let mut rng = stream(o.seed, 5200);
    let mut detail = stream(o.seed, 7495);
    ServiceProgram {
        drawer_bias: detail.random_range(-0.55..0.55),
        handle_fraction: detail.random_range(0.13..0.33),
        levels: rng.random_range(2..=3),
        drawers: rng.random_range(2..=5),
        frame: if rng.random_bool(0.45) {
            Surface::Wood
        } else {
            Surface::Metal
        },
        upholstery: if rng.random_bool(0.28) {
            Surface::Leather
        } else {
            Surface::FabricAlt
        },
        roundness: rng.random_range(2.0..8.0),
        feet_height: rng.random_range(0.055..0.12),
        slats: rng.random_bool(0.50),
        backed: rng.random_bool(0.45),
        panel_fraction: rng.random_range(0.58..0.90),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match o.kind {
        ObjectKind::MobileCart => cart(a, o, &p),
        ObjectKind::DrawerUnit => drawers(a, o, &p),
        ObjectKind::Bench => bench(a, o, &p),
        ObjectKind::Ottoman => ottoman(a, o, &p),
        ObjectKind::RoomDivider => divider(a, o, &p),
        _ => unreachable!("non-service furniture"),
    }
}

fn feet(a: &mut Assembly, o: &IndoorObject, height: f32, caster: bool) {
    let s = o.size;
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            let pos = Vec3::new(x * s.x * 0.36, 0., z * s.z * 0.34);
            if caster {
                a.part(Surface::Rubber, o.kind.class_name()).ellipsoid(
                    Vec3::new(height * 0.38, height * 0.38, height * 0.22),
                    Transform::from_translation(pos + Vec3::Y * height * 0.38),
                );
                a.box_part(
                    Surface::Metal,
                    o.kind.class_name(),
                    pos + Vec3::Y * height * 0.78,
                    Vec3::new(height * 0.36, height * 0.44, height * 0.32),
                    0.003,
                );
            } else {
                a.box_part(
                    Surface::WoodEdge,
                    o.kind.class_name(),
                    pos + Vec3::Y * height * 0.5,
                    Vec3::new(0.038, height, 0.038),
                    0.004,
                );
            }
        }
    }
}

fn cart(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let feet_h = p.feet_height.min(s.y * 0.16);
    feet(a, o, feet_h, true);
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            a.part(Surface::Metal, label).rod(
                Vec3::new(x * s.x * 0.45, feet_h, z * s.z * 0.43),
                Vec3::new(x * s.x * 0.45, s.y - 0.014, z * s.z * 0.43),
                0.012,
            );
        }
    }
    for level in 0..p.levels {
        let y = feet_h + 0.07 + (s.y - feet_h - 0.084) * level as f32 / (p.levels - 1) as f32;
        a.box_part(
            p.frame,
            label,
            Vec3::Y * y,
            Vec3::new(s.x, 0.028, s.z),
            0.004,
        );
        // Lower trays have a raised lip. The upper top remains fully usable.
        if level + 1 < p.levels {
            for z in [-1., 1.] {
                a.box_part(
                    Surface::Metal,
                    label,
                    Vec3::new(0., y + 0.03, z * (s.z * 0.5 - 0.009)),
                    Vec3::new(s.x, 0.045, 0.018),
                    0.003,
                );
            }
            let mut rng = stream(o.seed, 5210 + level as u64);
            if rng.random_bool(0.7) {
                a.box_part(
                    Surface::Paper,
                    "other_prop",
                    Vec3::new(0., y + 0.085, 0.),
                    Vec3::new(s.x * rng.random_range(0.40..0.66), 0.14, s.z * 0.70),
                    0.004,
                );
            }
        }
    }
    a.part(Surface::Metal, label).rod(
        Vec3::new(-s.x * 0.44, s.y - 0.075, -s.z * 0.43),
        Vec3::new(s.x * 0.44, s.y - 0.075, -s.z * 0.43),
        0.013,
    );
}

fn drawers(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let foot = p.feet_height.min(s.y * 0.16);
    feet(a, o, foot, o.variant == 0);
    kitchen::case_shell(
        a,
        Vec3::new(s.x * 0.98, s.y - foot - 0.02, s.z - 0.052),
        foot,
        -0.018,
        p.frame,
        label,
    );
    let pitch = (s.y - foot - 0.035) / p.drawers as f32;
    let mut bottom = foot + 0.012;
    for i in 0..p.drawers {
        let pitch = pitch * (1. + p.drawer_bias * (2. * i as f32 / (p.drawers - 1) as f32 - 1.));
        let y = bottom + pitch * 0.5;
        bottom += pitch;
        use super::super::motion::{construction as motion, MotionKind};
        let seed = o.seed ^ i as u64 ^ 0x44524157;
        motion::hinged(
            a,
            MotionKind::Cabinet,
            seed,
            || {
                motion::rigid(
                    MotionKind::Cabinet,
                    seed,
                    Vec3::ZERO,
                    Vec3::Y,
                    0.,
                    Vec3::Z * s.z * 0.60,
                )
            },
            |a| {
                a.box_part(
                    Surface::WoodEdge,
                    label,
                    Vec3::new(0., y, s.z * 0.5 - 0.012),
                    Vec3::new(s.x * 0.97, pitch - 0.007, 0.022),
                    0.003,
                );
                kitchen::drawer_tray(
                    a,
                    s.x * 0.97,
                    pitch - 0.007,
                    s.z - 0.052,
                    Transform::from_xyz(0., y, s.z * 0.5 - 0.012),
                    label,
                );
                a.part(Surface::Metal, label).rod(
                    Vec3::new(
                        -s.x * p.handle_fraction,
                        y + pitch * 0.21,
                        s.z * 0.5 - 0.002,
                    ),
                    Vec3::new(s.x * p.handle_fraction, y + pitch * 0.21, s.z * 0.5 - 0.002),
                    0.0025,
                );
            },
        );
    }
    a.box_part(
        p.frame,
        label,
        Vec3::Y * (s.y - 0.010),
        Vec3::new(s.x, 0.020, s.z),
        0.004,
    );
}

fn bench(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let seat = if p.backed { s.y.min(0.48) } else { s.y };
    let thickness = 0.07_f32.min(seat * 0.18);
    for x in [-1., 1.] {
        let xx = x * s.x * 0.38;
        for z in [-1., 1.] {
            a.box_part(
                p.frame,
                label,
                Vec3::new(xx, (seat - thickness) * 0.5, z * s.z * 0.33),
                Vec3::new(0.038, seat - thickness, 0.038),
                0.004,
            );
        }
    }
    if p.slats {
        let count = (s.z / 0.078).round().clamp(4., 10.) as usize;
        let pitch = s.z / count as f32;
        for i in 0..count {
            a.box_part(
                Surface::Wood,
                label,
                Vec3::new(
                    0.,
                    seat - thickness * 0.5,
                    (i as f32 + 0.5) * pitch - s.z * 0.5,
                ),
                Vec3::new(s.x, thickness, pitch - 0.008),
                0.005,
            );
        }
    } else {
        a.part(p.upholstery, label).cushion(
            Vec3::new(s.x, thickness, s.z),
            super::super::geometry::CushionProfile {
                roundness: p.roundness,
                crown: 3.5,
                dish: 0.003,
            },
            Transform::from_xyz(0., seat - thickness * 0.5, 0.),
        );
    }
    if p.backed && s.y > seat + 0.10 {
        for x in [-1., 1.] {
            a.box_part(
                p.frame,
                label,
                Vec3::new(x * s.x * 0.40, s.y * 0.5, -s.z * 0.43),
                Vec3::new(0.032, s.y, 0.032),
                0.003,
            );
        }
        a.box_part(
            if p.slats { Surface::Wood } else { p.upholstery },
            label,
            Vec3::new(0., (s.y + seat + 0.045) * 0.5, -s.z * 0.5 + 0.034),
            Vec3::new(s.x * 0.95, s.y - seat - 0.045, 0.06),
            0.012,
        );
    }
}

/// Categorical construction signature includes only parameters used by this
/// object's geometry, never inactive draws from another object's program.
pub fn construction(o: &IndoorObject) -> String {
    let p = parameters(o);
    match o.kind {
        ObjectKind::MobileCart => format!("cart/{:?}/{}-levels", p.frame, p.levels),
        ObjectKind::DrawerUnit => format!(
            "drawers/{:?}/{}-drawers/casters={}",
            p.frame,
            p.drawers,
            o.variant == 0
        ),
        ObjectKind::Bench => format!(
            "bench/{:?}/slats={}/back={}",
            p.frame,
            p.slats,
            p.backed && o.size.y > 0.58
        ),
        ObjectKind::Ottoman => format!(
            "ottoman/{:?}/{}",
            p.upholstery,
            if p.roundness < 3. {
                "round"
            } else if p.roundness < 5. {
                "soft-square"
            } else {
                "square"
            }
        ),
        ObjectKind::RoomDivider => format!("divider/slats={}", p.slats),
        _ => unreachable!("non-service construction"),
    }
}

fn ottoman(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    let leg = p.feet_height.min(s.y * 0.24);
    feet(a, o, leg, false);
    let cushion = super::super::geometry::CushionProfile {
        roundness: p.roundness,
        crown: 4.2,
        dish: 0.002,
    };
    let top = (s.y - leg) * 0.32;
    a.part(p.upholstery, label).cushion(
        Vec3::new(s.x * 0.98, s.y - leg - top, s.z * 0.98),
        cushion,
        Transform::from_xyz(0., (s.y - top + leg) * 0.5, 0.),
    );
    a.part(p.upholstery, label).cushion(
        Vec3::new(s.x, top, s.z),
        cushion,
        Transform::from_xyz(0., s.y - top * 0.5, 0.),
    );
}

fn divider(a: &mut Assembly, o: &IndoorObject, p: &ServiceProgram) {
    let s = o.size;
    let label = o.kind.class_name();
    for x in [-1., 1.] {
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x * s.x * 0.34, 0.012, 0.),
            Vec3::new(0.08, 0.024, s.z),
            0.006,
        );
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x * s.x * 0.34, s.y * 0.5, 0.),
            Vec3::new(0.026, s.y * 0.97, 0.026),
            0.003,
        );
    }
    let bottom = s.y * (1. - p.panel_fraction);
    a.box_part(
        Surface::FabricAlt,
        label,
        Vec3::Y * ((s.y + bottom) * 0.5),
        Vec3::new(s.x * 0.98, s.y - bottom, s.z.min(0.055)),
        0.010,
    );
    if p.slats {
        let n = (s.x / 0.07).round().clamp(6., 28.) as usize;
        for i in 0..n {
            a.box_part(
                Surface::WoodEdge,
                label,
                Vec3::new(
                    (i as f32 + 0.5) * s.x * 0.95 / n as f32 - s.x * 0.475,
                    (s.y + bottom) * 0.5,
                    s.z.min(0.055) * 0.5 + 0.012,
                ),
                Vec3::new(0.016, s.y - bottom - 0.02, 0.016),
                0.002,
            );
        }
    }
}
