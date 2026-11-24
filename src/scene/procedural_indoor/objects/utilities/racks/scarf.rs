//! One continuous doubled textile over the actual supporting cylinder. Gravity
//! sets the vertical tails independently of the angle of the wooden peg.
use super::{cloth, dressing::Garment, *};

pub(super) fn build(g: &mut Geometry, p: &Garment) {
    const W: usize = 11;
    const H: usize = 19;
    const TAIL: usize = 6;
    let axis = p.support_axis;
    let side = axis.cross(Vec3::Y).normalize();
    let up = side.cross(axis);
    let thickness = 0.0016;
    let layer = |lining: bool| -> [[Vec3; W]; H] {
        let radius = p.support_radius_m + if lining { 0.00012 } else { thickness };
        std::array::from_fn(|j| {
            let (angle, t, end) = if j < TAIL {
                (0., (TAIL - j) as f32 / TAIL as f32, -1.)
            } else if j > H - 1 - TAIL {
                (PI, (j - (H - 1 - TAIL)) as f32 / TAIL as f32, 1.)
            } else {
                (PI * (j - TAIL) as f32 / (H - 1 - 2 * TAIL) as f32, 0., 1.)
            };
            let radial = up * angle.sin() - side * angle.cos();
            let width = p.shoulder_width_m + (p.width_m - p.shoulder_width_m) * t * (2. - t);
            std::array::from_fn(|i| {
                let u = i as f32 / (W - 1) as f32;
                let phase = u * TAU * (p.fold_count as f32 * 0.5) + p.fold_phase;
                let drop = p.length_m * if end < 0. { 1. } else { p.tail_fraction };
                // Finish the hem on a slight bias; the gathers open up gradually.
                let wrinkle =
                    (0.62 + 0.30 * (phase + end * t * 1.4).sin() + 0.08 * (u * PI + t * 1.7).cos())
                        * t;
                p.anchor_m
                    + axis * ((u - 0.5) * width)
                    + radial * radius
                    + side * end * p.half_depth_m * wrinkle
                    - Vec3::Y * (drop * t + 0.008 * t * t * (phase + 0.4).sin())
            })
        })
    };
    let outer = layer(false);
    let inner = layer(true);
    // Along the centre of the bend, width follows the rod and length crosses it.
    cloth::patch(g, outer, true);
    cloth::patch(g, inner, false);
    for i in [0, W - 1] {
        cloth::band(
            g,
            outer.map(|row| row[i]),
            inner.map(|row| row[i]),
            axis * if i == 0 { -1. } else { 1. },
        );
    }
    for j in [0, H - 1] {
        cloth::band(g, outer[j], inner[j], Vec3::NEG_Y);
    }
}
