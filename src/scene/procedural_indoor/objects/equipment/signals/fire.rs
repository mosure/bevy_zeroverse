//! Distinct installed devices, rather than recolors of a single alarm box.
use super::*;

pub const FINISH_BASE: usize = 300;
#[derive(Debug, Clone, serde::Serialize)]
pub struct FireProgram {
    pub style: u32,
    pub finish: usize,
    pub bevel: f32,
    pub slots: usize,
    pub grille_fraction: f32,
    pub lens_fraction: f32,
}

pub fn program(o: &IndoorObject) -> FireProgram {
    let mut r = stream(o.seed, 0x4649524550524f47);
    let style = o.variant % 4;
    let colors: &[usize] = match o.kind {
        ObjectKind::FireDetector => &[1, 1, 1, 2, 3],
        ObjectKind::FirePullStation => &[0, 0, 0, 1, 4],
        _ => &[0, 0, 1, 1, 2, 4],
    };
    FireProgram {
        style,
        finish: FINISH_BASE + colors[r.random_range(0..colors.len())],
        bevel: r.random_range(0.025..0.12),
        slots: r.random_range(5..=12),
        grille_fraction: r.random_range(0.50..0.78),
        lens_fraction: r.random_range(0.38..0.64),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = program(o);
    match o.kind {
        ObjectKind::FireAlarm => alarm(a, o, &p),
        ObjectKind::FireDetector => detector(a, o, &p),
        ObjectKind::FirePullStation => pull(a, o, &p),
        _ => unreachable!(),
    }
}

fn alarm(a: &mut Assembly, o: &IndoorObject, p: &FireProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let body = format!("{l}#finish{}", p.finish);
    if p.style == 3 {
        // Steel gong bell and central fastening screw; its axis faces the room.
        let radius = s.x.min(s.y) * 0.48;
        let tf = Transform::from_xyz(0., s.y * 0.5, -s.z * 0.46)
            .with_rotation(Quat::from_rotation_x(PI * 0.5));
        a.part(Surface::Metal, &body).lathe(
            &[
                (0., 0.),
                (radius, 0.),
                (radius, s.z * 0.20),
                (radius * 0.85, s.z * 0.68),
                (radius * 0.40, s.z * 0.92),
                (0., s.z * 0.94),
            ],
            24,
            tf,
        );
        a.part(Surface::Chrome, l).ellipsoid(
            Vec3::new(radius * 0.06, radius * 0.06, s.z * 0.04),
            Transform::from_xyz(0., s.y * 0.5, s.z * 0.49),
        );
        return;
    }
    a.box_part(
        Surface::Plastic,
        &body,
        Vec3::new(0., s.y * 0.5, -s.z * 0.10),
        s * Vec3::new(1., 1., 0.80),
        s.x.min(s.y) * p.bevel,
    );
    let front = s.z * 0.30;
    if p.style != 2 {
        for i in 0..p.slots {
            let y = if p.style == 0 {
                0.20 + 0.45 * i as f32 / p.slots as f32
            } else {
                0.16 + 0.29 * i as f32 / p.slots as f32
            };
            let width = if i % 2 == 0 { 1. } else { 0.85 };
            a.box_part(
                Surface::Ink,
                l,
                Vec3::new(0., s.y * y, front + 0.0006),
                Vec3::new(s.x * p.grille_fraction * width, s.y * 0.012, 0.0008),
                0.0003,
            );
        }
    }
    if p.style != 0 {
        let h = s.y * if p.style == 2 { 0.38 } else { 0.21 };
        let center = Vec3::new(0., s.y * 0.61, front + s.z * 0.09);
        a.box_part(
            Surface::Paper,
            l,
            center,
            Vec3::new(s.x * p.lens_fraction, h, s.z * 0.18),
            0.003,
        );
        // Molded optical ribs, with positive separation from the lens face.
        for i in 0..5 {
            a.box_part(
                Surface::Plastic,
                &format!("{l}#finish{}", FINISH_BASE + 1),
                center
                    + Vec3::new(
                        (i as f32 - 2.) * s.x * p.lens_fraction * 0.16,
                        0.,
                        s.z * 0.09 + 0.0005,
                    ),
                Vec3::new(s.x * 0.016, h * 0.90, 0.0008),
                0.0002,
            );
        }
    }
    lettering::build(
        a,
        l,
        "FIRE",
        Vec2::new(s.x * 0.66, s.y * 0.12),
        Vec3::new(0., s.y * 0.85, front + 0.0015),
        if p.finish == FINISH_BASE {
            Surface::Paper
        } else {
            Surface::Ink
        },
    );
    a.part(Surface::Light, l).ellipsoid(
        Vec3::splat(s.x * 0.012),
        Transform::from_xyz(s.x * 0.34, s.y * 0.11, front),
    );
}

fn detector(a: &mut Assembly, o: &IndoorObject, p: &FireProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let body = format!("{l}#finish{}", p.finish);
    let radius = s.x.min(s.z) * 0.49;
    let underside = if p.style == 2 { 0.25 } else { 0.02 };
    a.part(Surface::Plastic, &body).lathe(
        &[
            (0., s.y * underside),
            (radius * 0.60, s.y * underside),
            (radius * 0.82, s.y * (underside + 0.16)),
            (radius * 0.96, s.y * 0.70),
            (radius, s.y * 0.84),
            (radius, s.y),
            (0., s.y),
        ],
        24,
        Transform::IDENTITY,
    );
    // Shadowed inlet band with separated louvres on the actual radial surface.
    for i in 0..p.slots * 2 {
        let angle = 2. * PI * i as f32 / (p.slots * 2) as f32;
        let tf = Transform::from_xyz(
            angle.sin() * radius * 0.93,
            s.y * 0.51,
            angle.cos() * radius * 0.93,
        )
        .with_rotation(Quat::from_rotation_y(angle));
        a.part(Surface::Ink, l)
            .cuboid(Vec3::new(radius * 0.09, s.y * 0.15, radius * 0.03), 0., tf);
    }
    if p.style == 2 {
        // Heat-sensing element with three protective struts beneath the dome.
        a.part(Surface::Metal, l).cylinder(
            radius * 0.07,
            s.y * 0.30,
            Transform::from_xyz(0., s.y * 0.15, 0.),
        );
        for i in 0..3 {
            let angle = 2. * PI * i as f32 / 3.;
            a.part(Surface::Metal, l).rod(
                Vec3::new(
                    angle.sin() * radius * 0.32,
                    s.y * 0.28,
                    angle.cos() * radius * 0.32,
                ),
                Vec3::new(
                    angle.sin() * radius * 0.16,
                    s.y * 0.005,
                    angle.cos() * radius * 0.16,
                ),
                s.y * 0.018,
            );
        }
    } else {
        a.part(Surface::Plastic, &format!("{l}#finish{}", FINISH_BASE + 1))
            .cylinder(
                radius * if p.style == 1 { 0.42 } else { 0.21 },
                s.y * 0.018,
                Transform::from_xyz(radius * 0.11, s.y * 0.012, -radius * 0.12),
            );
    }
    a.part(Surface::Light, l).ellipsoid(
        Vec3::new(radius * 0.035, s.y * 0.015, radius * 0.035),
        Transform::from_xyz(radius * 0.43, s.y * (underside - 0.002), radius * 0.18),
    );
}

fn pull(a: &mut Assembly, o: &IndoorObject, p: &FireProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let body = format!("{l}#finish{}", p.finish);
    a.box_part(
        Surface::Plastic,
        &body,
        Vec3::new(0., s.y * 0.5, -s.z * 0.12),
        s * Vec3::new(1., 1., 0.76),
        s.x * p.bevel * 0.6,
    );
    let front = s.z * 0.26;
    let text_surface = if p.finish == FINISH_BASE {
        Surface::Paper
    } else {
        Surface::Ink
    };
    lettering::build(
        a,
        l,
        "FIRE",
        Vec2::new(s.x * 0.70, s.y * 0.14),
        Vec3::new(0., s.y * 0.83, front + 0.0015),
        text_surface,
    );
    if p.style < 2 {
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(0., s.y * 0.43, front + 0.0008),
            Vec3::new(s.x * 0.68, s.y * 0.37, 0.001),
            0.001,
        );
        let hinge = Vec3::new(0., s.y * 0.43, front + s.z * 0.10);
        a.box_part(
            Surface::Plastic,
            &body,
            hinge,
            Vec3::new(s.x * 0.62, s.y * 0.12, s.z * 0.18),
            0.002,
        );
        lettering::build(
            a,
            l,
            "PULL",
            Vec2::new(s.x * 0.43, s.y * 0.075),
            hinge + Vec3::Z * (s.z * 0.09 + 0.0012),
            text_surface,
        );
        if p.style == 1 {
            a.box_part(
                Surface::Metal,
                l,
                Vec3::new(0., s.y * 0.63, front + s.z * 0.05),
                Vec3::new(s.x * 0.64, s.y * 0.055, s.z * 0.10),
                0.001,
            );
        }
    } else {
        // Resettable manual call point / break-glass panel, unactivated.
        a.box_part(
            Surface::Paper,
            l,
            Vec3::new(0., s.y * 0.45, front + 0.001),
            Vec3::new(s.x * 0.71, s.y * 0.41, 0.0015),
            0.001,
        );
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(0., s.y * 0.45, front + 0.0024),
            Vec3::new(s.x * 0.21, s.y * 0.025, 0.0006),
            0.0002,
        );
        if p.style == 3 {
            for x in [-0.40, 0.40] {
                a.part(Surface::Metal, l).rod(
                    Vec3::new(s.x * x, s.y * 0.19, s.z * 0.46),
                    Vec3::new(s.x * x, s.y * 0.72, s.z * 0.46),
                    s.x * 0.018,
                );
            }
            a.part(Surface::Metal, l).rod(
                Vec3::new(-s.x * 0.40, s.y * 0.72, s.z * 0.46),
                Vec3::new(s.x * 0.40, s.y * 0.72, s.z * 0.46),
                s.x * 0.018,
            );
        }
    }
    a.part(Surface::Chrome, l).ellipsoid(
        Vec3::new(s.x * 0.025, s.x * 0.025, 0.001),
        Transform::from_xyz(0., s.y * 0.11, front + 0.0005),
    );
}
