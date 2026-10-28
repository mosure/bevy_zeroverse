//! Dimensioned drawer carcasses with fitted fronts, hardware and motion ownership.
use super::*;

mod program;
pub use program::{dimensions, parameters, DrawerBank, DrawerFront, DrawerProgram};
#[cfg(test)]
mod tests;

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
    // Dividers are behind the fronts and fit between the carcass floor/roof.
    for bank in p.banks.iter().skip(1) {
        let center = bank.left - 0.004;
        a.box_part(
            p.frame,
            label,
            Vec3::new(center, (p.foot + s.y - p.top) * 0.5, -0.025 + p.panel * 0.5),
            Vec3::new(
                p.panel,
                s.y - p.foot - p.top - 2. * p.panel,
                s.z - 0.065 - p.panel,
            ),
            0.001,
        );
    }
    for front in p.fronts(s.y) {
        let i = front.index;
        let width = front.right - front.left;
        let x = (front.right + front.left) * 0.5;
        let height = front.top - front.bottom;
        let y = (front.top + front.bottom) * 0.5;
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
                    Vec3::new(x, y, z),
                    Vec3::new(width, height, 0.022),
                    0.002,
                );
                kitchen::drawer_tray(
                    a,
                    width,
                    height,
                    s.z - p.handle_depth - 0.045,
                    Transform::from_xyz(x, y, z),
                    label,
                );
                if p.profile != 0 {
                    let h = (height - 0.006) * if p.profile == 1 { 0.66 } else { 0.86 };
                    a.box_part(
                        p.front,
                        label,
                        Vec3::new(x, y, z + 0.0125),
                        Vec3::new(width * if p.profile == 1 { 0.78 } else { 0.91 }, h, 0.003),
                        0.001,
                    );
                }
                let center = Vec3::new(x, y + height * p.handle_height, z + 0.011);
                if p.hardware == 0 {
                    let x = width * p.handle_width * 0.5;
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
                    let radius = (height * 0.11).clamp(0.004, 0.011);
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
                        Vec3::new(width * p.handle_width, (height * 0.08).min(0.014), 0.001),
                        0.0003,
                    );
                    a.box_part(
                        p.hardware_finish,
                        label,
                        center + Vec3::new(0., height * 0.045, p.handle_depth * 0.45),
                        Vec3::new(width * p.handle_width, 0.003, p.handle_depth * 0.82),
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
