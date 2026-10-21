//! Wall hardware in a local +Z facing frame. Recesses are inset dark geometry.
use super::*;

/// Root-to-wall offset. Geometry rear faces differ for plates and clock cases.
/// A millimetre gap avoids coplanar faces while retaining a plausible mounting.
pub(crate) fn mount_offset(o: &IndoorObject) -> Option<f32> {
    let fraction = match o.kind {
        ObjectKind::Display
        | ObjectKind::Whiteboard
        | ObjectKind::WallArt
        | ObjectKind::ProjectionScreen
        | ObjectKind::WallShelf
        | ObjectKind::NoticeBoard
        | ObjectKind::WallSconce
        | ObjectKind::ExitSign
        | ObjectKind::FirePullStation => 0.5,
        ObjectKind::StringLights if !equipment::signals::string_program(o).suspended => 0.5,
        ObjectKind::FireAlarm => {
            if o.variant % 4 == 3 {
                0.46
            } else {
                0.5
            }
        }
        ObjectKind::Speaker if !o.solid && o.support.is_none() => 0.46,
        ObjectKind::Clock => return Some(clocks::case_depth(o) * 0.5 + 0.001),
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => 0.35,
        _ => return None,
    };
    Some(o.size.z * fraction + 0.001)
}

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct FixtureProgram {
    pub gangs: usize,
    pub style: u8,
    pub edge_radius: f32,
    pub rocker_angle: f32,
    pub finish: Surface,
    pub screws: bool,
}
pub fn parameters(o: &IndoorObject) -> FixtureProgram {
    let mut rng = stream(o.seed, 3070);
    let mut fit = stream(o.seed, 7490);
    let capacity = (o.size.x / 0.075).round().clamp(1., 3.) as usize;
    FixtureProgram {
        gangs: if capacity > 1 && fit.random_bool(0.35) {
            capacity - 1
        } else {
            capacity
        },
        style: rng.random_range(0..6),
        edge_radius: rng.random_range(0.001..0.006),
        rocker_angle: rng.random_range(-0.025..0.025),
        finish: [Surface::Metal, Surface::Plastic, Surface::Paper][rng.random_range(0..3)],
        screws: rng.random_bool(0.6),
    }
}
/// Lower receptacle of the first gang. Cable endpoints use this exact frame.
pub(crate) fn outlet_port_local(o: &IndoorObject) -> Vec3 {
    let gangs = parameters(o).gangs as f32;
    Vec3::new(
        -(gangs - 1.) * 0.5 * o.size.x / gangs,
        o.size.y * 0.28,
        o.size.z * 0.52 + 0.001,
    )
}
fn disk(a: &mut Assembly, surface: Surface, label: &str, p: Vec3, r: f32, depth: f32) {
    a.part(surface, label).cylinder(
        r,
        depth,
        Transform::from_translation(p).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
}
pub(super) fn fixture(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    a.box_part(
        p.finish,
        label,
        Vec3::Y * s.y * 0.5,
        s * Vec3::new(1., 1., 0.7),
        p.edge_radius,
    );
    let pitch = s.x / p.gangs as f32;
    for gang in 0..p.gangs {
        let x = (gang as f32 - (p.gangs - 1) as f32 * 0.5) * pitch;
        let face = s.z * 0.48;
        if o.kind == ObjectKind::LightSwitch {
            // The recess and controls are *in front* of the plate, not hidden
            // inside it. Separation remains positive at every rocker angle.
            let w = pitch * 0.57;
            let h = s.y * 0.65;
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(x, s.y * 0.5, s.z * 0.36),
                Vec3::new(w + 0.002, h + 0.002, 0.0003),
                0.0001,
            );
            match p.style {
                0 => {
                    // Broad rocker with sampled on/off angle.
                    a.part(Surface::Plastic, label).cuboid(
                        Vec3::new(w, h, 0.003),
                        0.001,
                        Transform::from_xyz(x, s.y * 0.5, face + 0.0005)
                            .with_rotation(Quat::from_rotation_x(p.rocker_angle)),
                    );
                }
                1 => {
                    // Narrow toggle on a separate escutcheon.
                    a.box_part(
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        Vec3::new(w, h, 0.002),
                        0.001,
                    );
                    a.part(Surface::Paper, label).cuboid(
                        Vec3::new(0.008, s.y * 0.23, 0.003),
                        0.001,
                        Transform::from_xyz(x, s.y * 0.5, face + 0.001)
                            .with_rotation(Quat::from_rotation_x(p.rocker_angle * 2.)),
                    );
                }
                2 => {
                    // Rotary dimmer with visible pointer.
                    disk(
                        a,
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        w * 0.46,
                        0.004,
                    );
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.5 + w * 0.25, face + 0.002),
                        Vec3::new(0.001, w * 0.19, 0.0005),
                        0.,
                    );
                }
                3 => {
                    // Split rocker.
                    for side in [-1., 1.] {
                        a.box_part(
                            Surface::Plastic,
                            label,
                            Vec3::new(x + side * w * 0.255, s.y * 0.5, face + 0.001),
                            Vec3::new(w * 0.46, h, 0.004),
                            0.001,
                        );
                    }
                }
                4 => {
                    // Sliding dimmer.
                    a.box_part(
                        Surface::Plastic,
                        label,
                        Vec3::new(x, s.y * 0.5, face),
                        Vec3::new(w, h, 0.002),
                        0.001,
                    );
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.5, face + 0.0013),
                        Vec3::new(w * 0.10, h * 0.75, 0.0003),
                        0.,
                    );
                    a.box_part(
                        Surface::Paper,
                        label,
                        Vec3::new(x, s.y * (0.5 + p.rocker_angle * 2.), face + 0.002),
                        Vec3::new(w * 0.62, h * 0.20, 0.002),
                        0.0005,
                    );
                }
                _ => {
                    // Two push buttons, distinct status lens.
                    for y in [0.33, 0.67] {
                        disk(
                            a,
                            Surface::Plastic,
                            label,
                            Vec3::new(x, s.y * y, face),
                            w * 0.39,
                            0.004,
                        );
                    }
                    disk(
                        a,
                        Surface::Ink,
                        label,
                        Vec3::new(x, s.y * 0.84, face),
                        0.0012,
                        0.0008,
                    );
                }
            }
        } else {
            for y in [s.y * 0.28, s.y * 0.72] {
                let center = Vec3::new(x, y, face);
                let w = (pitch * 0.62).min(0.050);
                let h = s.y * 0.32;
                if p.style == 1 || p.style == 4 {
                    disk(a, Surface::Plastic, label, center, w.min(h) * 0.49, 0.003);
                    disk(
                        a,
                        Surface::Ink,
                        label,
                        center + Vec3::Z * 0.0018,
                        w.min(h) * 0.37,
                        0.0004,
                    );
                    disk(
                        a,
                        p.finish,
                        label,
                        center + Vec3::Z * 0.0022,
                        w.min(h) * 0.32,
                        0.0004,
                    );
                } else {
                    a.box_part(
                        Surface::Plastic,
                        label,
                        center,
                        Vec3::new(w, h, 0.003),
                        0.002,
                    );
                }
                let z = s.z * 0.52 + 0.001;
                for sign in [-1., 1.] {
                    if p.style == 1 || p.style == 4 {
                        disk(
                            a,
                            Surface::Ink,
                            label,
                            Vec3::new(x + sign * w * 0.18, y, z + 0.001),
                            0.0024,
                            0.0005,
                        );
                    } else {
                        a.box_part(
                            Surface::Ink,
                            label,
                            Vec3::new(x + sign * w * 0.18, y + 0.002, z),
                            Vec3::new(
                                if p.style == 2 { 0.006 } else { 0.0025 },
                                if p.style == 2 { 0.003 } else { 0.009 },
                                0.0006,
                            ),
                            0.,
                        );
                    }
                }
                if p.style != 1 {
                    a.box_part(
                        Surface::Ink,
                        label,
                        Vec3::new(x, y + if p.style == 2 { 0.010 } else { -0.008 }, z + 0.0002),
                        Vec3::new(0.003, 0.004, 0.0006),
                        0.0005,
                    );
                }
            }
            if p.style == 3 || p.style == 5 {
                // USB-A / USB-C inserts between the mains receptacles.
                a.box_part(
                    Surface::Ink,
                    label,
                    Vec3::new(x, s.y * 0.50, face + 0.0018),
                    Vec3::new(if p.style == 3 { 0.013 } else { 0.008 }, 0.0035, 0.0006),
                    0.0005,
                );
            }
        }
    }
    if p.screws {
        for y in [s.y * 0.075, s.y * 0.925] {
            disk(
                a,
                Surface::Metal,
                label,
                Vec3::new(0., y, s.z * 0.37),
                0.0019,
                0.0006,
            );
            a.box_part(
                Surface::Ink,
                label,
                Vec3::new(0., y, s.z * 0.37 + 0.0005),
                Vec3::new(0.0024, 0.0004, 0.0003),
                0.,
            );
        }
    }
}
