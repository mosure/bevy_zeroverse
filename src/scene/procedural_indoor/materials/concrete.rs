//! Casting, air voids, cement curing and finishing over the mineral matrix.
use super::{
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConcreteFinish {
    pub formwork: f32,
    pub board_width_m: f32,
    pub form_relief_m: f32,
    pub seam_width_m: f32,
    pub trowel: f32,
    pub trowel_scale_m: f32,
    pub cure_variation: f32,
    pub bleed: f32,
    pub sand_exposure: f32,
    pub bughole_density: f32,
    pub bughole_radius_m: f32,
    pub bughole_depth_m: f32,
}
impl ConcreteFinish {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x434f4e4352455445);
        Self {
            formwork: rng.random_range(0.0_f32..1.).powi(2),
            board_width_m: rng.random_range(0.08..0.28),
            form_relief_m: rng.random_range(0.00001..0.00025),
            seam_width_m: rng.random_range(0.0006..0.0020),
            trowel: rng.random_range(0.0_f32..1.).powi(2),
            trowel_scale_m: rng.random_range(0.025..0.15),
            cure_variation: rng.random_range(0.035..0.22),
            bleed: rng.random_range(0.0_f32..1.).powi(2) * 0.12,
            sand_exposure: rng.random_range(0.0..0.60),
            bughole_density: rng.random_range(0.0_f32..1.).powi(2) * 0.18,
            bughole_radius_m: rng.random_range(0.0008..0.004),
            bughole_depth_m: rng.random_range(0.00015..0.0013),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.formwork,
            self.trowel,
            self.cure_variation,
            self.bleed,
            self.sand_exposure,
            self.bughole_density,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !self.board_width_m.is_finite()
            || !(0.04..=0.5).contains(&self.board_width_m)
            || !self.trowel_scale_m.is_finite()
            || !(0.01..=0.3).contains(&self.trowel_scale_m)
            || [
                self.form_relief_m,
                self.seam_width_m,
                self.bughole_radius_m,
                self.bughole_depth_m,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.000005..=0.008).contains(v))
        {
            return Err("invalid concrete casting/finish program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, r: &MaterialRecipe, uv: [f32; 2], polish: f32, t: &mut Texel) {
        let [u, v] = uv;
        let boards = frequency(r.period_m, self.board_width_m);
        let warped = v * boards as f32
            + (periodic_noise(u, v, 3, 5, r.seed.wrapping_add(607)) - 0.5) * 0.025;
        let edge = (warped.rem_euclid(1.) - 0.5).abs();
        let seam = band(
            0.5 - edge,
            self.seam_width_m / r.period_m * boards as f32,
            boards as f32 / r.map_size(0) as f32,
        ) * self.formwork;
        let imprint = periodic_noise(u, v, 4, boards * 19, r.seed.wrapping_add(613)) - 0.5;
        let board_id = (warped.floor() as i32).rem_euclid(boards as i32) as u32;
        let board_tone =
            (super::hash(0, board_id, r.seed.wrapping_add(615)) - 0.5) * self.formwork * 0.12;
        let pass_n = frequency(r.period_m, self.trowel_scale_m);
        let passes = deposit(uv, [pass_n, pass_n], 0.16, r.seed.wrapping_add(617)) - 0.5;
        let cure = deposit(uv, [3, 5], 0.17, r.seed.wrapping_add(619)) - 0.5;
        let bleed =
            (periodic_noise(u, v, 21, 3, r.seed.wrapping_add(631)) - 0.45).max(0.) * self.bleed;
        let sand = periodic_noise(u, v, 173, 167, r.seed.wrapping_add(641)) - 0.5;
        let void_n = frequency(r.period_m, self.bughole_radius_m * 14.);
        let cell = cellular(uv, [void_n, void_n], r.seed.wrapping_add(643));
        let radius = self.bughole_radius_m / r.period_m * void_n as f32 * (0.6 + cell.dye);
        let void = disk(cell.radius, radius, void_n as f32 / r.map_size(0) as f32)
            * smooth((self.bughole_density - cell.dye) * 15.);
        // Grinding removes board impressions and closes the paste's fine relief,
        // while a few casting voids remain. No geological veins are added here.
        let open = 1. - polish * 0.92;
        t.color = tint(
            t.color,
            1. + cure * self.cure_variation + passes * self.trowel * 0.06 + board_tone
                - bleed
                - seam * 0.065
                - void * 0.28 * (1. - polish * 0.5),
        );
        t.height += open
            * (imprint * self.formwork * self.form_relief_m
                + passes * self.trowel * 0.000055
                + sand * self.sand_exposure * 0.000035
                - seam * self.form_relief_m)
            - void * self.bughole_depth_m * (1. - polish * 0.65);
        t.roughness = (t.roughness + cure * 0.05 - passes * self.trowel * 0.12
            + sand * self.sand_exposure * 0.035
            + void * 0.12
            + seam * 0.035)
            .clamp(0.18, 0.99);
        t.occlusion = (t.occlusion * (1. - void * 0.10 - seam * 0.008)).max(0.80);
    }
}
