//! Modular upholstery: armchairs, loveseats, sofas and left/right chaise returns.
use super::*;
mod pillows;
pub mod recliner;
mod structure;
#[cfg(test)]
mod tests;
mod upholstery;

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
    /// Track, rolled, swept, open bentwood, flared shell, or armless.
    pub arm_style: u8,
    /// Tapered feet, hairpins, sleds, plinth, pedestal/rails, or block feet.
    pub base_style: u8,
    /// Loose cushion, lumbar split, channels, bolster crest, or exposed rails.
    pub back_style: u8,
    /// Individual cushions, continuous bench, or split front bolster.
    pub seat_style: u8,
    pub arm_sweep_m: f32,
    pub back_division: f32,
    pub recliner: bool,
    /// Static mechanism position; never creates a per-frame system.
    pub recliner_extension: f32,
}
pub fn parameters(o: &IndoorObject) -> SofaProgram {
    let mut rng = stream(o.seed, 3031);
    let mut p = SofaProgram {
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
        arm_style: 0,
        base_style: 0,
        back_style: 0,
        seat_style: 0,
        arm_sweep_m: 0.,
        back_division: 0.,
        recliner: is_recliner(o),
        recliner_extension: 0.,
    };
    let mut structure = stream(o.seed, 6434);
    p.arm_style = structure.random_range(0..6);
    p.base_style = structure.random_range(0..6);
    p.back_style = structure.random_range(0..5);
    p.seat_style = structure.random_range(0..3);
    if (p.chaise || p.modules == 1) && p.seat_style == 1 {
        // A continuous rectangular bench cannot cover a sectional's void.
        p.seat_style = 0;
    }
    p.arm_sweep_m = structure.random_range(0.035..0.09);
    p.back_division = structure.random_range(0.32..0.55);
    p.recliner_extension = structure.random_range(0.0..1.0);
    if p.arm_style == 5 && !p.recliner {
        p.arm_width = 0.025;
    }
    if p.recliner {
        // These are the realized dedicated recliner constructions.
        p.arm_style %= 3;
        p.back_style = 1;
        p.seat_style = 0;
        p.pillows = false;
    }
    p
}

pub fn is_recliner(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Chair
        && o.variant == 8
        && o.size.z >= 1.12
        && o.size.y >= 1.02
        && recliner::selected(o.seed)
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
    build_program(a, o, p);
}

fn build_program(a: &mut Assembly, o: &IndoorObject, p: SofaProgram) {
    if p.recliner {
        recliner::build(a, o, &p);
        return;
    }
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
        if p.seat_style != 1 || index == 0 {
            let (seat_x, seat_width) = if p.seat_style == 1 {
                (0., s.x - p.arm_width * 2.)
            } else {
                (x, module_width)
            };
            upholstery::seat(
                a,
                o,
                &p,
                Vec3::new(seat_x, p.seat_height - p.cushion * 0.5, z - 0.085),
                Vec3::new(seat_width - p.cushion_gap_m, p.cushion, depth - 0.19),
            );
        }
        structure::feet(a, o, &p, x, z, module_width, depth);
        upholstery::back(a, o, &p, x, module_width, base_top);
    }
    structure::arms(a, o, &p, base_depth);
    for pillow in pillows::build(o, &p, &widths) {
        a.part(Surface::FabricAlt, "pillow").append(pillow);
    }
}
