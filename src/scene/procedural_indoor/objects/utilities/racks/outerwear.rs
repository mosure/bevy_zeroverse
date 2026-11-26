//! Front, back and raglan sleeves share a sewn pattern. The garment drops from
//! the shoulders of the wire hanger and retains open cuffs, neck and hem.
use super::{
    cloth,
    dressing::{Garment, GarmentKind},
    *,
};
mod details;

const U: usize = 13;
const V: [f32; 9] = [0., 0.05, 0.25, 0.48, 0.68, 0.76, 0.84, 0.92, 1.];
const ARM: usize = 4;
const CUFF: usize = 5;
const NECK: usize = 2;

fn across(i: usize) -> f32 {
    (i as f32 - ((U - 1) / 2) as f32) / ((U - 1) / 2) as f32
}

struct Pattern<'a>(&'a Garment);
impl Pattern<'_> {
    fn panel(&self, u: f32, v: f32, face: f32) -> Vec3 {
        let p = self.0;
        let w = p.shoulder_width_m * 0.5;
        let shoulder = (v - 0.68).max(0.) / 0.32;
        let width = w * match p.kind {
            GarmentKind::Dress => 0.78 + 0.48 * (1. - v).powi(2) + 0.22 * shoulder,
            GarmentKind::Shirt => 0.92 + 0.08 * shoulder - 0.035 * (v * PI).sin(),
            _ => 0.86 - 0.075 * (v * PI).sin() + 0.14 * shoulder,
        };
        let collar = (1. - (u.abs() * 3.).min(1.)).powi(2);
        let top = -w * 0.51 * u.abs()
            - p.shoulder_width_m * (if face > 0. { 0.21 } else { 0.075 }) * collar;
        let hem = 0.013 * (2.2 * u + p.fold_phase).sin();
        let fold = (p.fold_phase + u * PI * p.fold_count as f32 + v * 0.77).cos();
        let fullness = 0.36 + 0.64 * (1. - u * u);
        let x = face
            * p.half_depth_m
            * fullness
            * (0.74 + 0.18 * fold * (1. - 0.7 * shoulder) + 0.05 * (v * PI).sin());
        p.anchor_m + Vec3::new(x, (hem - p.length_m) * (1. - v) + top * v, u * width)
    }
    fn sleeve(&self, side: f32, j: usize, s: f32, face: f32) -> Vec3 {
        let p = self.0;
        let t = j as f32 / (V.len() - 1 - ARM) as f32;
        let root = self.panel(side, V[ARM + j], face);
        let extent = p.width_m * 0.5;
        if matches!(p.kind, GarmentKind::Shirt | GarmentKind::Dress) {
            // A short sleeve projects out from the armhole. Using a coat's
            // hanging cuff here can pull the underarm back through its seam.
            // Keep both ends outside the shoulder and preserve shoulder-to-
            // underarm order around the cuff opening as the sleeve narrows.
            let shoulder = self.panel(side, 1., face).y;
            let underarm = self.panel(side, V[ARM], face).y;
            let center = (shoulder + underarm) * 0.5;
            let drop = (shoulder - underarm) * p.tail_fraction
                + (extent - p.shoulder_width_m * 0.5) * 0.55;
            let cuff = Vec3::new(
                p.anchor_m.x + face * p.half_depth_m * (0.23 + 0.26 * (t * PI).sin()),
                center + (root.y - center) * 0.58 - drop,
                p.anchor_m.z + side * extent,
            );
            return root.lerp(cuff, s) + Vec3::X * (face * (s * PI).sin() * p.half_depth_m * 0.065);
        }
        let cuff_w = (extent - p.shoulder_width_m * 0.43).min(0.084);
        let cuff = p.anchor_m
            + Vec3::new(
                face * p.half_depth_m * (0.22 + 0.52 * (t * PI).sin()),
                -p.length_m * p.tail_fraction + (t - 0.5) * 0.026,
                side * (extent - (1. - t) * cuff_w),
            );
        let elbow = (s * PI).sin();
        root.lerp(cuff, s) + Vec3::new(face * elbow * p.half_depth_m * 0.11, elbow * 0.018, 0.)
    }
    fn lining(&self, p: Vec3) -> Vec3 {
        let center = self.0.anchor_m - Vec3::Y * self.0.length_m * 0.50;
        center + (p - center) * Vec3::new(0.91, 0.994, 0.991)
    }
    fn skin(&self, g: &mut Geometry, inner: bool) {
        let map = |x| if inner { self.lining(x) } else { x };
        let torso = |face: f32| -> [[Vec3; U]; 9] {
            std::array::from_fn(|j| std::array::from_fn(|i| map(self.panel(across(i), V[j], face))))
        };
        let back = torso(-1.);
        let front = torso(1.);
        cloth::patch(g, front, !inner);
        cloth::patch(g, back, inner);
        for side in [-1., 1.] {
            let i = if side < 0. { 0 } else { U - 1 };
            // The side seam stops at the underarm. Sleeves share the remainder.
            let seam = |f: [[Vec3; U]; 9]| -> [Vec3; ARM + 1] { std::array::from_fn(|j| f[j][i]) };
            let polarity = if inner { -1. } else { 1. };
            cloth::band(g, seam(front), seam(back), Vec3::Z * (side * polarity));
            let sleeve = |face: f32| -> [[Vec3; ARM + 1]; CUFF] {
                std::array::from_fn(|k| {
                    std::array::from_fn(|j| {
                        map(self.sleeve(side, j, k as f32 / (CUFF - 1) as f32, face))
                    })
                })
            };
            let a = sleeve(1.);
            let b = sleeve(-1.);
            // Reversed on the right/left according to the orientation of the cut.
            let flipped = (a[0][1] - a[0][0]).cross(a[1][0] - a[0][0]).x < 0.;
            cloth::patch(g, a, flipped ^ inner);
            cloth::patch(g, b, (!flipped) ^ inner);
            for j in [0, ARM] {
                let outward = if j == 0 {
                    -side * Vec3::Z - Vec3::Y
                } else {
                    side * Vec3::Z + Vec3::Y
                };
                cloth::band(g, a.map(|r| r[j]), b.map(|r| r[j]), outward * polarity);
            }
            // Shoulder seam ends at the neckline, leaving a genuine opening.
            let first = if side < 0. { 0 } else { (U - 1) / 2 + NECK };
            let shoulder =
                |f: [[Vec3; U]; 9]| -> [Vec3; 5] { std::array::from_fn(|j| f[8][first + j]) };
            cloth::band(g, shoulder(front), shoulder(back), Vec3::Y * polarity);
        }
    }
    fn turn_edges(&self, g: &mut Geometry) {
        for face in [-1., 1.] {
            let hem: [Vec3; U] = std::array::from_fn(|i| self.panel(across(i), 0., face));
            cloth::band(g, hem, hem.map(|p| self.lining(p)), Vec3::NEG_Y);
            let neck: [Vec3; 2 * NECK + 1] =
                std::array::from_fn(|i| self.panel(across(i + (U - 1) / 2 - NECK), 1., face));
            cloth::band(g, neck, neck.map(|p| self.lining(p)), Vec3::Y);
            for side in [-1., 1.] {
                let cuff: [Vec3; ARM + 1] = std::array::from_fn(|j| self.sleeve(side, j, 1., face));
                cloth::band(g, cuff, cuff.map(|p| self.lining(p)), Vec3::NEG_Y);
            }
        }
        // Close the turned thickness at the two edges of each sewn opening.
        for side in [-1., 1.] {
            for (u, v, n) in [(side, 0., Vec3::NEG_Y), (side / 3., 1., Vec3::Y)] {
                let p = [self.panel(u, v, -1.), self.panel(u, v, 1.)];
                cloth::band(g, p, p.map(|q| self.lining(q)), n);
            }
            for j in [0, ARM] {
                let p = [self.sleeve(side, j, 1., -1.), self.sleeve(side, j, 1., 1.)];
                cloth::band(g, p, p.map(|q| self.lining(q)), Vec3::NEG_Y);
            }
        }
    }
}

pub(super) fn build(a: &mut Assembly, p: &Garment, label: &str) {
    let pattern = Pattern(p);
    let g = a.part(p.surface, label);
    pattern.skin(g, false);
    pattern.skin(g, true);
    pattern.turn_edges(g);
    if matches!(p.kind, GarmentKind::Shirt | GarmentKind::Dress) {
        return;
    }
    details::sew(g, &pattern);
    if p.closure == 0 {
        // A zipper follows the actual front pattern, including its shallow folds.
        let points = std::array::from_fn::<_, 8, _>(|i| {
            pattern.panel(0., 0.07 + i as f32 * 0.123, 1.) + Vec3::X * 0.001
        });
        a.part(Surface::Metal, "clothes").tube(&points, 0.0015, 6);
        let grip = points[7] + Vec3::new(0.002, -0.007, 0.);
        a.box_part(
            Surface::Metal,
            "clothes",
            grip,
            Vec3::new(0.003, 0.015, 0.005),
            0.001,
        );
    } else {
        for i in 0..4 {
            let center = pattern.panel(0., 0.20 + i as f32 * 0.19, 1.);
            a.part(Surface::Plastic, "clothes").lathe(
                &[
                    (0.0055, -0.0003),
                    (0.006, 0.001),
                    (0.005, 0.0025),
                    (0., 0.0025),
                ],
                10,
                Transform::from_translation(center)
                    .with_rotation(Quat::from_rotation_z(-FRAC_PI_2)),
            );
        }
    }
}
