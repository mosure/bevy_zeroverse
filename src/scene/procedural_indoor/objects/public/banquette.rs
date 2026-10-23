//! Contract-sized cafe seating with independent base, padding and back programs.
use super::*;
use crate::scene::procedural_indoor::geometry::CushionProfile;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BanquetteProgram {
    /// Recessed plinth, exposed legs, or continuous sleds.
    pub base: u8,
    /// Continuous upholstery, vertical channels, horizontal panels, or timber slats.
    pub back: u8,
    pub seat_height_m: f32,
    pub cushion_m: f32,
    pub back_thickness_m: f32,
    pub back_lean: f32,
    pub roundness: f32,
    pub crown: f32,
    pub leg_m: f32,
    pub setback: f32,
    pub seat_weights: Vec<f32>,
    pub back_divisions: usize,
    pub upholstery: Surface,
    pub frame: Surface,
}

pub fn parameters(o: &IndoorObject) -> BanquetteProgram {
    let mut r = stream(o.seed, 0x4241_4e51_5545_5454);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4241_4e51_4649_4e49);
    let seats = if r.random_bool(0.30) {
        1
    } else {
        (o.size.x / r.random_range(0.48..0.72))
            .round()
            .clamp(2., 6.) as usize
    };
    BanquetteProgram {
        base: r.random_range(0..3),
        back: r.random_range(0..4),
        seat_height_m: r.random_range(0.435..0.485),
        cushion_m: r.random_range(0.065..0.13),
        back_thickness_m: r.random_range(0.055..0.085),
        back_lean: r.random_range(0.025..0.10),
        roundness: r.random_range(3.2..8.5),
        crown: r.random_range(2.4..5.5),
        leg_m: r.random_range(0.025..0.052),
        setback: r.random_range(0.075..0.15),
        seat_weights: (0..seats).map(|_| r.random_range(0.85..1.15)).collect(),
        back_divisions: (o.size.x / r.random_range(0.12..0.28))
            .round()
            .clamp(4., 18.) as usize,
        upholstery: [Surface::Fabric, Surface::FabricAlt, Surface::Leather][f.random_range(0..3)],
        frame: [Surface::Wood, Surface::Metal, Surface::Paint][f.random_range(0..3)],
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let deck_top = p.seat_height_m - p.cushion_m;
    let deck = 0.026;
    let width = s.x * 0.96;
    let depth = s.z * 0.88;
    let foot_x = width * (0.5 - p.setback);
    let foot_z = depth * (0.5 - p.setback);
    if p.base == 0 {
        a.box_part(
            p.frame,
            l,
            Vec3::Y * (deck_top - deck) * 0.5,
            Vec3::new(width * (1. - p.setback), deck_top - deck, depth * 0.84),
            0.005,
        );
    } else {
        let count = if width > 2.1 { 3 } else { 2 };
        for i in 0..count {
            let x = -foot_x + 2. * foot_x * i as f32 / (count - 1) as f32;
            let foot_height = if p.base == 2 { p.leg_m } else { 0. };
            for z in [-foot_z, foot_z] {
                a.box_part(
                    p.frame,
                    l,
                    Vec3::new(x, (deck_top - deck + foot_height) * 0.5, z),
                    Vec3::new(p.leg_m, deck_top - deck - foot_height, p.leg_m),
                    p.leg_m * 0.12,
                );
            }
            if p.base == 2 {
                a.box_part(
                    p.frame,
                    l,
                    Vec3::new(x, p.leg_m * 0.5, 0.),
                    Vec3::new(p.leg_m * 1.1, p.leg_m, foot_z * 2. + p.leg_m),
                    0.005,
                );
            }
        }
    }
    a.box_part(
        p.frame,
        l,
        Vec3::Y * (deck_top - deck * 0.5),
        Vec3::new(width, deck, depth),
        0.004,
    );
    // A tilted back with a real rear frame, wholly inside the reserved envelope.
    // The adult seating datum remains independent of the overall back height.
    let bottom = deck_top + 0.02;
    let (sin, cos) = p.back_lean.sin_cos();
    let h = (s.y - 0.008 - bottom - p.back_thickness_m * sin) / cos;
    let z = -s.z * 0.5 + 0.006 + (h * sin + p.back_thickness_m * cos) * 0.5;
    let tf = Transform::from_xyz(0., (bottom + s.y - 0.008) * 0.5, z)
        .with_rotation(Quat::from_rotation_x(-p.back_lean));
    let seat_rear = z + (h * sin + p.back_thickness_m * cos) * 0.5 + 0.003;
    let seat_front = depth * 0.5;
    let profile = CushionProfile {
        roundness: p.roundness,
        crown: p.crown,
        dish: 0.,
    };
    let total: f32 = p.seat_weights.iter().sum();
    let mut left = -width * 0.5;
    for &weight in &p.seat_weights {
        let w = width * weight / total;
        a.part(p.upholstery, l).cushion(
            Vec3::new(w - 0.008, p.cushion_m, seat_front - seat_rear),
            profile,
            Transform::from_xyz(
                left + w * 0.5,
                p.seat_height_m - p.cushion_m * 0.5,
                (seat_front + seat_rear) * 0.5,
            ),
        );
        left += w;
    }
    let frame_depth = 0.018;
    a.part(p.frame, l).cuboid(
        Vec3::new(width, h, frame_depth),
        0.003,
        tf.with_translation(
            tf.transform_point(Vec3::Z * (-p.back_thickness_m + frame_depth) * 0.5),
        ),
    );
    for x in [-width * 0.42, width * 0.42] {
        let foot = tf.transform_point(Vec3::new(
            x,
            -h * 0.5 + 0.016,
            (-p.back_thickness_m + frame_depth) * 0.5,
        ));
        a.part(p.frame, l)
            .rod(Vec3::new(x, deck_top - 0.012, foot.z), foot, 0.009);
    }
    let front_depth = p.back_thickness_m - frame_depth;
    let mut pad = |x: f32, y: f32, w: f32, height: f32| {
        let center = tf.transform_point(Vec3::new(x, y, frame_depth * 0.5));
        // Local cushion Y is thickness and local Z is height.
        a.part(p.upholstery, l).cushion(
            Vec3::new(w, front_depth, height),
            profile,
            Transform::from_translation(center)
                .with_rotation(tf.rotation * Quat::from_rotation_x(FRAC_PI_2)),
        );
    };
    match p.back {
        0 => pad(0., 0., width - 0.006, h - 0.006),
        1 => {
            let w = width / p.back_divisions as f32;
            for i in 0..p.back_divisions {
                pad(
                    -width * 0.5 + (i as f32 + 0.5) * w,
                    0.,
                    w - 0.006,
                    h - 0.006,
                );
            }
        }
        2 => {
            let n = ((h / 0.19).round() as usize).clamp(2, 5);
            for i in 0..n {
                pad(
                    0.,
                    -h * 0.5 + (i as f32 + 0.5) * h / n as f32,
                    width - 0.006,
                    h / n as f32 - 0.009,
                );
            }
        }
        _ => {
            // Slats sit against the backing; shared geometry does not force
            // every cafe bench to have a padded rectangular back.
            let w = width / p.back_divisions as f32;
            for i in 0..p.back_divisions {
                let center = tf.transform_point(Vec3::new(
                    -width * 0.5 + (i as f32 + 0.5) * w,
                    0.,
                    frame_depth * 0.5,
                ));
                a.part(p.frame, l).cuboid(
                    Vec3::new(w * 0.76, h - 0.008, front_depth),
                    0.006,
                    tf.with_translation(center),
                );
            }
        }
    }
}
