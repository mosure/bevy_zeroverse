//! Casting, air voids, cement curing and finishing over the mineral matrix.
use super::{
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod prepared;
pub(super) use prepared::PreparedConcrete;

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
    pub(super) fn apply_prepared(
        &self,
        r: &MaterialRecipe,
        uv: [f32; 2],
        polish: f32,
        t: &mut Texel,
        prepared: Option<&PreparedConcrete>,
    ) {
        self.apply_color_carry(r, uv, polish, t, prepared, None);
    }

    pub(super) fn apply_carried(
        &self,
        r: &MaterialRecipe,
        uv: [f32; 2],
        polish: f32,
        t: &mut Texel,
        prepared: Option<&PreparedConcrete>,
        color: super::mineral::ColorCarry,
    ) {
        self.apply_color_carry(r, uv, polish, t, prepared, Some(color));
    }

    fn apply_color_carry(
        &self,
        r: &MaterialRecipe,
        uv: [f32; 2],
        polish: f32,
        t: &mut Texel,
        prepared: Option<&PreparedConcrete>,
        color: Option<super::mineral::ColorCarry>,
    ) {
        let [u, v] = uv;
        let boards =
            prepared.map_or_else(|| frequency(r.period_m, self.board_width_m), |p| p.boards);
        let warp = prepared.map_or_else(
            || periodic_noise(u, v, 3, 5, r.seed.wrapping_add(607)),
            |p| p.noise[0].sample(u, v),
        );
        let warped = v * boards as f32 + (warp - 0.5) * 0.025;
        let edge = (warped.rem_euclid(1.) - 0.5).abs();
        let seam = band(
            0.5 - edge,
            prepared.map_or_else(
                || self.seam_width_m / r.period_m * boards as f32,
                |p| p.seam_width,
            ),
            prepared.map_or_else(
                || boards as f32 / r.map_size(0) as f32,
                |p| p.board_footprint,
            ),
        ) * self.formwork;
        let imprint = prepared.map_or_else(
            || periodic_noise(u, v, 4, boards * 19, r.seed.wrapping_add(613)),
            |p| p.noise[1].sample(u, v),
        ) - 0.5;
        let board_id = (warped.floor() as i32).rem_euclid(boards as i32) as u32;
        let board_tone = prepared.map_or_else(
            || (super::hash(0, board_id, r.seed.wrapping_add(615)) - 0.5) * self.formwork * 0.12,
            |p| p.board_tone(self, r, board_id),
        );
        let passes = prepared.map_or_else(
            || {
                let pass_n = frequency(r.period_m, self.trowel_scale_m);
                deposit(uv, [pass_n, pass_n], 0.16, r.seed.wrapping_add(617))
            },
            |p| p.passes.sample(uv),
        ) - 0.5;
        let cure = prepared.map_or_else(
            || deposit(uv, [3, 5], 0.17, r.seed.wrapping_add(619)),
            |p| p.cure.sample(uv),
        ) - 0.5;
        let bleed = (prepared.map_or_else(
            || periodic_noise(u, v, 21, 3, r.seed.wrapping_add(631)),
            |p| p.noise[2].sample(u, v),
        ) - 0.45)
            .max(0.)
            * self.bleed;
        let sand = prepared.map_or_else(
            || periodic_noise(u, v, 173, 167, r.seed.wrapping_add(641)),
            |p| p.noise[3].sample(u, v),
        ) - 0.5;
        let void_n = prepared.map_or_else(
            || frequency(r.period_m, self.bughole_radius_m * 14.),
            |p| p.void_n,
        );
        let cell = prepared.and_then(|p| p.voids.as_ref()).map_or_else(
            || cellular(uv, [void_n, void_n], r.seed.wrapping_add(643)).into(),
            |cells| cells.sample_primary(uv),
        );
        let radius = prepared.map_or_else(
            || self.bughole_radius_m / r.period_m * void_n as f32,
            |p| p.void_radius,
        ) * (0.6 + cell.dye);
        let void = disk(
            cell.radius,
            radius,
            prepared.map_or_else(
                || void_n as f32 / r.map_size(0) as f32,
                |p| p.void_footprint,
            ),
        ) * smooth((self.bughole_density - cell.dye) * 15.);
        // Grinding removes board impressions and closes the paste's fine relief,
        // while a few casting voids remain. No geological veins are added here.
        let open = 1. - polish * 0.92;
        let gain = 1. + cure * self.cure_variation + passes * self.trowel * 0.06 + board_tone
            - bleed
            - seam * 0.065
            - void * 0.28 * (1. - polish * 0.5);
        t.color = color.map_or_else(|| tint(t.color, gain), |color| color.tinted(gain).encoded());
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

#[cfg(test)]
pub(super) mod replay_tests;
