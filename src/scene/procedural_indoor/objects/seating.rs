//! Modular upholstery: armchairs, loveseats, sofas and left/right chaise returns.
use super::*;

#[derive(Debug, serde::Serialize)]
pub struct SofaProgram {
    pub modules: usize,
    pub chaise: bool,
    pub left_return: bool,
    pub arm_width: f32,
    pub seat_height: f32,
    pub cushion: f32,
    pub leg_height: f32,
    pub roundness: f32,
    pub back_tilt: f32,
    pub upholstery: Surface,
    pub exposed_frame: bool,
    pub pillows: bool,
}
pub fn parameters(o: &IndoorObject) -> SofaProgram {
    let mut rng = stream(o.seed, 3031);
    SofaProgram {
        modules: if o.kind == ObjectKind::Chair {
            1
        } else {
            (o.size.x / rng.random_range(0.62..0.93))
                .round()
                .clamp(1., 5.) as usize
        },
        chaise: o.kind == ObjectKind::Sofa && o.size.z > 1.25,
        left_return: rng.random_bool(0.5),
        arm_width: rng.random_range(0.065..0.17_f32).min(o.size.x * 0.17),
        seat_height: if o.kind == ObjectKind::Chair {
            0.47
        } else {
            rng.random_range(0.41..0.48)
        },
        cushion: rng.random_range(0.075..0.145),
        leg_height: rng.random_range(0.07..0.18),
        roundness: rng.random_range(0.025..0.075),
        back_tilt: rng.random_range(-0.06..0.14),
        upholstery: [Surface::Fabric, Surface::FabricAlt, Surface::Leather][rng.random_range(0..3)],
        exposed_frame: rng.random_bool(0.35),
        pillows: rng.random_bool(0.65),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let label = o.kind.class_name();
    let frame = if p.exposed_frame {
        Surface::WoodEdge
    } else {
        p.upholstery
    };
    let base_depth = if p.chaise { s.z.min(0.94) } else { s.z };
    let module_width = (s.x - p.arm_width * 2.) / p.modules as f32;
    let back_z = s.z * 0.5 - 0.11;
    for index in 0..p.modules {
        let is_return = p.chaise && index == if p.left_return { 0 } else { p.modules - 1 };
        let depth = if is_return { s.z } else { base_depth };
        let z = (s.z - depth) * 0.5;
        let x = (index as f32 - (p.modules - 1) as f32 * 0.5) * module_width;
        let base_top = p.seat_height - p.cushion;
        a.box_part(
            frame,
            label,
            Vec3::new(x, (p.leg_height + base_top) * 0.5, z),
            Vec3::new(module_width - 0.004, base_top - p.leg_height, depth - 0.018),
            p.roundness * 0.5,
        );
        a.box_part(
            p.upholstery,
            label,
            Vec3::new(x, p.seat_height - p.cushion * 0.5, z - 0.085),
            Vec3::new(module_width - 0.014, p.cushion, depth - 0.19),
            p.roundness,
        );
        for side in [-1., 1.] {
            for end in [-1., 1.] {
                a.part(
                    if p.exposed_frame {
                        Surface::WoodEdge
                    } else {
                        Surface::Metal
                    },
                    label,
                )
                .rod(
                    Vec3::new(
                        x + side * module_width * 0.38,
                        0.010,
                        z + end * (depth * 0.5 - 0.07),
                    ),
                    Vec3::new(
                        x + side * module_width * 0.35,
                        p.leg_height + 0.018,
                        z + end * (depth * 0.5 - 0.09),
                    ),
                    0.018,
                );
                a.part(Surface::Rubber, label).cylinder(
                    0.020,
                    0.012,
                    Transform::from_xyz(
                        x + side * module_width * 0.38,
                        0.006,
                        z + end * (depth * 0.5 - 0.07),
                    ),
                );
            }
        }
        let bh = (s.y - p.seat_height + 0.015).max(0.18);
        a.part(p.upholstery, label).cuboid(
            Vec3::new(module_width - 0.014, bh, 0.18),
            p.roundness,
            Transform::from_xyz(
                x,
                p.seat_height + bh * 0.5 - 0.04,
                back_z - bh * p.back_tilt.abs() * 0.5,
            )
            .with_rotation(Quat::from_rotation_x(p.back_tilt)),
        );
    }
    for side in [-1., 1.] {
        let is_return = p.chaise && ((side < 0.) == p.left_return);
        let depth = if is_return { s.z } else { base_depth };
        let height = (p.seat_height + 0.16).min(s.y - 0.08);
        a.box_part(
            frame,
            label,
            Vec3::new(
                side * (s.x - p.arm_width) * 0.5,
                (p.leg_height + height) * 0.5,
                (s.z - depth) * 0.5,
            ),
            Vec3::new(p.arm_width, height - p.leg_height, depth),
            p.roundness,
        );
        if p.pillows && s.x > 1.2 {
            a.part(Surface::FabricAlt, "pillow").cuboid(
                Vec3::new(0.28, 0.27, 0.10),
                0.042,
                Transform::from_xyz(
                    side * (s.x * 0.5 - p.arm_width - 0.18),
                    p.seat_height + 0.13,
                    back_z - 0.17,
                )
                .with_rotation(Quat::from_rotation_z(side * 0.18)),
            );
        }
    }
}
