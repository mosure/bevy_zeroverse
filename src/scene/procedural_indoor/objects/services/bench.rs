//! Supported bench constructions with ergonomic seat height independent of the back.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchProgram {
    pub seat_height: f32,
    pub seat_thickness: f32,
    pub base: u8,
    pub span: Vec2,
    pub splay: f32,
    pub rail: f32,
    pub slats: bool,
    pub longitudinal: bool,
    pub divisions: usize,
    pub gap_fraction: f32,
    pub backed: bool,
    pub back_tilt: f32,
    pub arms: bool,
    pub frame: Surface,
    pub seat: Surface,
    pub roundness: f32,
}

pub fn parameters(o: &IndoorObject) -> BenchProgram {
    let mut r = stream(o.seed, 0x42454e4348505247);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4245_4e43_4649_4e49);
    let seat_height = o.size.y.min(r.random_range(0.41..0.52));
    let slats = r.random_bool(0.52);
    let longitudinal = r.random_bool(0.58);
    let wants_back = r.random_bool(0.72);
    let wants_arms = r.random_bool(0.38);
    let backed = o.size.y > seat_height + 0.14 && wants_back;
    BenchProgram {
        seat_height,
        seat_thickness: (seat_height * r.random_range(0.085..0.19)).min(0.085),
        base: r.random_range(0..4),
        span: Vec2::new(r.random_range(0.27..0.40), r.random_range(0.24..0.37)),
        splay: r.random_range(0.01..0.055),
        rail: r.random_range(0.025..0.050),
        slats,
        longitudinal,
        divisions: if slats {
            ((if longitudinal { o.size.z } else { o.size.x }) / r.random_range(0.045..0.095_f32))
                .round()
                .clamp(3., 22.) as usize
        } else {
            r.random_range(1..=3)
        },
        gap_fraction: r.random_range(0.025..0.13),
        backed,
        back_tilt: r.random_range(0.035..0.16),
        arms: o.size.y > seat_height + 0.14 && wants_arms,
        frame: if f.random_bool(0.55) {
            Surface::Wood
        } else {
            Surface::Metal
        },
        seat: if slats {
            Surface::Wood
        } else if f.random_bool(0.28) {
            Surface::Leather
        } else {
            Surface::FabricAlt
        },
        roundness: r.random_range(2.6..7.5),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let underside = p.seat_height - p.seat_thickness;
    let span = p.span * s.xz();
    for side in [-1., 1.] {
        let x = side * span.x;
        match p.base {
            0 => {
                for z in [-1., 1.] {
                    let bottom = Vec3::new(
                        side * (span.x + p.splay).min(s.x * 0.5 - p.rail * 0.51),
                        0.,
                        z * (span.y + p.splay * 0.4).min(s.z * 0.5 - p.rail * 0.51),
                    );
                    let top = Vec3::new(x, underside, z * span.y);
                    // Flat foot caps at y=0; inclined members stop inside those caps.
                    a.box_part(
                        p.frame,
                        label,
                        bottom + Vec3::Y * 0.008,
                        Vec3::new(p.rail, 0.016, p.rail),
                        0.002,
                    );
                    a.part(p.frame, label)
                        .rod(bottom + Vec3::Y * 0.008, top, p.rail * 0.40);
                }
            }
            1 => {
                a.box_part(
                    p.frame,
                    label,
                    Vec3::new(x, p.rail * 0.5, 0.),
                    Vec3::new(p.rail, p.rail, span.y * 2. + p.rail),
                    0.003,
                );
                for z in [-1., 1.] {
                    a.box_part(
                        p.frame,
                        label,
                        Vec3::new(x, (underside + p.rail) * 0.5, z * span.y),
                        Vec3::new(p.rail, underside - p.rail, p.rail),
                        0.003,
                    );
                }
            }
            2 => {
                a.box_part(
                    p.frame,
                    label,
                    Vec3::new(x, p.rail * 0.4, 0.),
                    Vec3::new(s.x * 0.12, p.rail * 0.8, s.z * 0.79),
                    p.rail * 0.15,
                );
                a.box_part(
                    p.frame,
                    label,
                    Vec3::new(x, (underside + p.rail * 0.8) * 0.5, 0.),
                    Vec3::new(s.x * 0.055, underside - p.rail * 0.8, s.z * 0.24),
                    0.006,
                );
            }
            _ => a.box_part(
                p.frame,
                label,
                Vec3::new(x, underside * 0.5, 0.),
                Vec3::new(p.rail * 1.3, underside, s.z * 0.70),
                0.004,
            ),
        }
        a.box_part(
            p.frame,
            label,
            Vec3::new(x, underside - p.rail * 0.5, 0.),
            Vec3::new(p.rail, p.rail, s.z * 0.84),
            0.003,
        );
    }
    a.box_part(
        p.frame,
        label,
        Vec3::Y * (underside - p.rail * 0.5),
        Vec3::new(span.x * 2., p.rail, p.rail),
        0.003,
    );
    for i in 0..p.divisions {
        let across_depth = p.slats && p.longitudinal;
        let extent = if across_depth { s.z } else { s.x };
        let pitch = extent / p.divisions as f32;
        let offset = pitch * (i as f32 + 0.5) - extent * 0.5;
        let width = pitch * (1. - p.gap_fraction);
        let size = if across_depth {
            Vec3::new(s.x, p.seat_thickness, width)
        } else {
            Vec3::new(width, p.seat_thickness, s.z)
        };
        let pos = Vec3::new(
            if across_depth { 0. } else { offset },
            p.seat_height - p.seat_thickness * 0.5,
            if across_depth { offset } else { 0. },
        );
        if p.slats {
            a.box_part(p.seat, label, pos, size, p.seat_thickness.min(width) * 0.12);
        } else {
            a.part(p.seat, label).cushion(
                size,
                super::super::super::geometry::CushionProfile {
                    roundness: p.roundness,
                    crown: 3.5,
                    dish: 0.002,
                },
                Transform::from_translation(pos),
            );
        }
    }
    if p.backed {
        let available = s.y - p.seat_height - 0.045;
        let thickness = (s.z * 0.08).min(0.045);
        let tilt = p.back_tilt.min((s.z * 0.13 / available).atan());
        let height = (available - thickness * tilt.sin()) / tilt.cos();
        let center = Vec3::new(0., p.seat_height + 0.025 + available * 0.5, -s.z * 0.34);
        let tf = Transform::from_translation(center).with_rotation(Quat::from_rotation_x(-tilt));
        for side in [-1., 1.] {
            a.part(p.frame, label).rod(
                Vec3::new(side * s.x * 0.41, underside, -s.z * 0.34),
                tf.transform_point(Vec3::new(
                    side * s.x * 0.41,
                    height * 0.46,
                    -thickness * 0.3,
                )),
                0.012,
            );
        }
        a.part(p.seat, label).cuboid(
            Vec3::new(s.x * 0.92, height, thickness),
            thickness * 0.20,
            tf,
        );
    }
    if p.arms {
        let y = (p.seat_height + 0.15).min(s.y - 0.025);
        for side in [-1., 1.] {
            for z in [-1., 1.] {
                a.part(p.frame, label).rod(
                    Vec3::new(side * s.x * 0.46, p.seat_height - 0.01, z * s.z * 0.31),
                    Vec3::new(side * s.x * 0.46, y, z * s.z * 0.31),
                    0.010,
                );
            }
            a.box_part(
                p.frame,
                label,
                Vec3::new(side * s.x * 0.46, y, 0.),
                Vec3::new(s.x.min(1.) * 0.045, 0.025, s.z * 0.73),
                0.008,
            );
        }
    }
}
