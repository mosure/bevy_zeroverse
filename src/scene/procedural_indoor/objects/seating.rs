//! Modular upholstery: armchairs, loveseats, sofas and left/right chaise returns.
use super::*;
mod pillows;
#[cfg(test)]
mod tests;

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
    pub cushion_crown: f32,
    pub cushion_dish_m: f32,
    pub seam_radius_m: f32,
    pub module_asymmetry: f32,
    pub cushion_gap_m: f32,
    pub arm_rise_m: f32,
    pub arm_crown: f32,
    pub arm_roundness: f32,
    pub pillow_width_m: f32,
    pub pillow_roll: f32,
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
        cushion_crown: rng.random_range(2.8..4.7),
        cushion_dish_m: rng.random_range(0.003..0.014),
        seam_radius_m: rng.random_range(0.0015..0.0030),
        module_asymmetry: rng.random_range(0.03..0.24),
        cushion_gap_m: rng.random_range(0.008..0.028),
        arm_rise_m: rng.random_range(0.12..0.23),
        arm_crown: rng.random_range(2.2..4.8),
        arm_roundness: rng.random_range(2.2..6.8),
        pillow_width_m: rng.random_range(0.24..0.34),
        pillow_roll: rng.random_range(-0.26..0.26),
    }
}
/// Realized cushion module widths, shared with qualification metrics.
/// Normalization reserves both arm structures and keeps asymmetric cushions
/// physically contained in the same nominal sofa footprint.
pub fn module_widths(o: &IndoorObject) -> Vec<f32> {
    module_widths_for(o, &parameters(o))
}
fn module_widths_for(o: &IndoorObject, p: &SofaProgram) -> Vec<f32> {
    let mut rng = stream(o.seed, 3032);
    let weights: Vec<_> = (0..p.modules)
        .map(|_| rng.random_range(1. - p.module_asymmetry..1. + p.module_asymmetry))
        .collect();
    let total: f32 = weights.iter().sum();
    weights
        .into_iter()
        .map(|w| w / total * (o.size.x - p.arm_width * 2.))
        .collect()
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
    // Unequal cushions are sampled as a normalized construction programme,
    // leaving the outer footprint and seating datum exactly unchanged.
    let widths = module_widths_for(o, &p);
    let back_z = s.z * 0.5 - 0.11;
    let mut cursor = -s.x * 0.5 + p.arm_width;
    for (index, &module_width) in widths.iter().enumerate() {
        let is_return = p.chaise && index == if p.left_return { 0 } else { p.modules - 1 };
        let depth = if is_return { s.z } else { base_depth };
        let z = (s.z - depth) * 0.5;
        let x = cursor + module_width * 0.5;
        cursor += module_width;
        let base_top = p.seat_height - p.cushion;
        a.box_part(
            frame,
            label,
            Vec3::new(x, (p.leg_height + base_top) * 0.5, z),
            Vec3::new(module_width - 0.004, base_top - p.leg_height, depth - 0.018),
            p.roundness * 0.5,
        );
        let cushion_size = Vec3::new(module_width - p.cushion_gap_m, p.cushion, depth - 0.19);
        let cushion_center = Vec3::new(x, p.seat_height - p.cushion * 0.5, z - 0.085);
        a.part(p.upholstery, label).cushion(
            cushion_size,
            super::super::geometry::CushionProfile {
                roundness: 4.0 + p.roundness * 20.0,
                crown: p.cushion_crown,
                dish: p.cushion_dish_m,
            },
            Transform::from_translation(cushion_center),
        );
        let piping: Vec<_> = (0..=48)
            .map(|i| {
                let t = TAU * (i % 48) as f32 / 48.;
                let shape = |v: f32| {
                    if v.abs() < 1e-6 {
                        0.
                    } else {
                        v.signum() * v.abs().powf(2. / (4.0 + p.roundness * 20.0))
                    }
                };
                cushion_center
                    + Vec3::new(
                        shape(t.cos()) * cushion_size.x * 0.5,
                        0.,
                        shape(t.sin()) * cushion_size.z * 0.5,
                    )
            })
            .collect();
        a.part(p.upholstery, label)
            .tube(&piping, p.seam_radius_m, 6);
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
        // A covered structural back joins the base to the loose back cushions.
        // Its rear edge is inset from the nominal envelope, avoiding co-planar
        // seams while preventing cushions from reading as unsupported slabs.
        let back_frame_top = (s.y - 0.04).max(base_top + 0.08);
        a.box_part(
            frame,
            label,
            Vec3::new(x, (base_top + back_frame_top) * 0.5, s.z * 0.5 - 0.065),
            Vec3::new(module_width - 0.004, back_frame_top - base_top, 0.10),
            p.roundness.min(0.035),
        );
        let bh = (s.y - p.seat_height + 0.015).max(0.18);
        a.part(p.upholstery, label).cushion(
            Vec3::new(module_width - p.cushion_gap_m, 0.18, bh),
            super::super::geometry::CushionProfile {
                roundness: 4.0 + p.roundness * 20.0,
                crown: p.cushion_crown,
                dish: 0.,
            },
            Transform::from_xyz(
                x,
                p.seat_height + bh * 0.5 - 0.04,
                back_z - bh * p.back_tilt.abs() * 0.5,
            )
            .with_rotation(Quat::from_rotation_x(p.back_tilt - FRAC_PI_2)),
        );
    }
    for side in [-1., 1.] {
        let is_return = p.chaise && ((side < 0.) == p.left_return);
        let depth = if is_return { s.z } else { base_depth };
        let height = (p.seat_height + p.arm_rise_m).min(s.y - 0.08);
        let arm_x = side * (s.x - p.arm_width) * 0.5;
        let arm_z = (s.z - depth) * 0.5;
        let pad_height = (p.cushion * 0.90).clamp(0.075, 0.13);
        let stem_top = height - pad_height * 0.60;
        a.box_part(
            frame,
            label,
            Vec3::new(arm_x, (p.leg_height + stem_top) * 0.5, arm_z),
            Vec3::new(p.arm_width * 0.82, stem_top - p.leg_height, depth - 0.014),
            p.roundness * 0.55,
        );
        // A rounded, continuously crowned upholstered arm pad supplies the
        // soft silhouette; its complete horizontal footprint stays in bounds.
        a.part(p.upholstery, label).cushion(
            Vec3::new(p.arm_width, pad_height, depth - 0.008),
            super::super::geometry::CushionProfile {
                roundness: p.arm_roundness,
                crown: p.arm_crown,
                dish: 0.,
            },
            Transform::from_xyz(arm_x, height - pad_height * 0.5, arm_z),
        );
    }
    for pillow in pillows::build(o, &p, &widths) {
        a.part(Surface::FabricAlt, "pillow").append(pillow);
    }
}
