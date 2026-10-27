//! Dimensioned drawer carcasses with fitted fronts, hardware and motion ownership.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DrawerProgram {
    pub weights: Vec<f32>,
    pub base: u8,
    pub foot: f32,
    pub foot_span: Vec2,
    pub top: f32,
    pub front: Surface,
    pub frame: Surface,
    pub profile: u8,
    pub hardware: u8,
    pub hardware_finish: Surface,
    pub handle_depth: f32,
    pub handle_width: f32,
    pub handle_height: f32,
}
pub fn parameters(o: &IndoorObject) -> DrawerProgram {
    let mut r = stream(o.seed, 0x4452415745525052);
    let n = r.random_range(2..=5);
    let frame = [Surface::Wood, Surface::Metal, Surface::Plastic][r.random_range(0..3)];
    DrawerProgram {
        weights: (0..n).map(|_| r.random_range(0.55..1.65)).collect(),
        base: r.random_range(0..3),
        foot: r.random_range(0.035_f32..0.095).min(o.size.y * 0.15),
        foot_span: Vec2::new(r.random_range(0.32..0.41), r.random_range(0.29..0.39)),
        top: r.random_range(0.016_f32..0.033).min(o.size.y * 0.075),
        front: if r.random_bool(0.65) {
            frame
        } else {
            Surface::WoodEdge
        },
        frame,
        profile: r.random_range(0..3),
        hardware: r.random_range(0..3),
        hardware_finish: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Wood,
            Surface::Plastic,
        ][r.random_range(0..4)],
        handle_depth: r.random_range(0.013..0.027),
        handle_width: r.random_range(0.17..0.39),
        handle_height: r.random_range(-0.06..0.31),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    if p.base == 2 {
        a.box_part(
            p.frame,
            label,
            Vec3::Y * p.foot * 0.5,
            Vec3::new(s.x * 0.88, p.foot, s.z * 0.84),
            0.004,
        );
    } else {
        for x in [-1., 1.] {
            for z in [-1., 1.] {
                let pos = Vec3::new(x * p.foot_span.x * s.x, 0., z * p.foot_span.y * s.z);
                if p.base == 0 {
                    a.part(Surface::Rubber, label).ellipsoid(
                        Vec3::new(p.foot * 0.38, p.foot * 0.38, p.foot * 0.22),
                        Transform::from_translation(pos + Vec3::Y * p.foot * 0.38),
                    );
                    a.box_part(
                        p.hardware_finish,
                        label,
                        pos + Vec3::Y * p.foot * 0.78,
                        Vec3::new(p.foot * 0.36, p.foot * 0.44, p.foot * 0.32),
                        0.002,
                    );
                } else {
                    a.box_part(
                        p.frame,
                        label,
                        pos + Vec3::Y * p.foot * 0.5,
                        Vec3::new(0.031, p.foot, 0.031),
                        0.003,
                    );
                }
            }
        }
    }
    kitchen::case_shell(
        a,
        Vec3::new(s.x * 0.98, s.y - p.foot - p.top, s.z - 0.065),
        p.foot,
        -0.025,
        p.frame,
        label,
    );
    let total: f32 = p.weights.iter().sum();
    let available = s.y - p.foot - p.top - 0.008;
    let mut bottom = p.foot + 0.004;
    for (i, w) in p.weights.iter().enumerate() {
        let pitch = available * w / total;
        let y = bottom + pitch * 0.5;
        bottom += pitch;
        let z = s.z * 0.5 - p.handle_depth - 0.013;
        use super::super::super::motion::{construction as motion, MotionKind};
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
                    p.front,
                    label,
                    Vec3::new(0., y, z),
                    Vec3::new(s.x * 0.97, pitch - 0.006, 0.022),
                    0.002,
                );
                kitchen::drawer_tray(
                    a,
                    s.x * 0.97,
                    pitch - 0.006,
                    s.z - p.handle_depth - 0.045,
                    Transform::from_xyz(0., y, z),
                    label,
                );
                if p.profile != 0 {
                    let h = (pitch - 0.012) * if p.profile == 1 { 0.66 } else { 0.86 };
                    a.box_part(
                        p.front,
                        label,
                        Vec3::new(0., y, z + 0.0125),
                        Vec3::new(s.x * if p.profile == 1 { 0.78 } else { 0.91 }, h, 0.003),
                        0.001,
                    );
                }
                let center = Vec3::new(0., y + (pitch - 0.006) * p.handle_height, z + 0.011);
                if p.hardware == 0 {
                    let x = s.x * p.handle_width * 0.5;
                    for side in [-1., 1.] {
                        a.part(p.hardware_finish, label).rod(
                            center + Vec3::X * side * x,
                            center + Vec3::X * side * x + Vec3::Z * (p.handle_depth - 0.004),
                            0.0025,
                        );
                    }
                    a.part(p.hardware_finish, label).rod(
                        center + Vec3::new(-x, 0., p.handle_depth - 0.004),
                        center + Vec3::new(x, 0., p.handle_depth - 0.004),
                        0.003,
                    );
                } else if p.hardware == 1 {
                    let radius = (pitch * 0.11).clamp(0.004, 0.011);
                    a.part(p.hardware_finish, label).rod(
                        center,
                        center + Vec3::Z * (p.handle_depth * 0.62),
                        radius * 0.35,
                    );
                    a.part(p.hardware_finish, label).ellipsoid(
                        Vec3::new(radius, radius, p.handle_depth * 0.33),
                        Transform::from_translation(center + Vec3::Z * p.handle_depth * 0.64),
                    );
                } else {
                    a.box_part(
                        Surface::Ink,
                        label,
                        center + Vec3::Z * 0.0045,
                        Vec3::new(s.x * p.handle_width, (pitch * 0.08).min(0.014), 0.001),
                        0.0003,
                    );
                    a.box_part(
                        p.hardware_finish,
                        label,
                        center + Vec3::new(0., pitch * 0.045, p.handle_depth * 0.45),
                        Vec3::new(s.x * p.handle_width, 0.003, p.handle_depth * 0.82),
                        0.001,
                    );
                }
            },
        );
    }
    // Full, level support surface is shared with clutter and elevated plants.
    a.box_part(
        p.frame,
        label,
        Vec3::Y * (s.y - p.top * 0.5),
        Vec3::new(s.x, p.top, s.z),
        0.003,
    );
}
