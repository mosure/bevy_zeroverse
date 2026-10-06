//! Fired bodies and glaze: wheel marks, iron speckles, reactive melt and crazing.
//! This is a surface approximation, not a simulation of kiln chemistry.
use super::{
    coating::CoatingRecipe,
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod prepared;
pub(super) use prepared::PreparedGlaze;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlazeRecipe {
    pub body_grain_m: f32,
    pub cloud_scale_m: f32,
    pub reactive_mix: f32,
    pub reactive_color: [f32; 3],
    pub thickness_variation: f32,
    pub flow: f32,
    pub throwing: f32,
    pub turning_pitch_m: f32,
    pub speckle_density: f32,
    pub speckle_radius_m: f32,
    pub speckle_color: [f32; 3],
    pub crack_spacing_m: f32,
}
impl GlazeRecipe {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x474c415a45);
        let warm = rng.random_range(-0.12_f32..0.20);
        let value = rng.random_range(0.50_f32..0.92);
        Self {
            body_grain_m: rng.random_range(0.0005..0.0018),
            cloud_scale_m: rng.random_range(0.008..0.055),
            reactive_mix: rng.random_range(0.0_f32..1.).powi(3) * 0.90,
            reactive_color: [
                value,
                (value * (1. - warm)).min(0.98),
                (value * (1. - warm * 1.7)).min(0.98),
            ],
            thickness_variation: rng.random_range(0.0_f32..1.).powi(2) * 0.85,
            flow: rng.random_range(0.0..0.70),
            throwing: rng.random_range(0.0_f32..1.).powi(2) * 0.80,
            turning_pitch_m: rng.random_range(0.002..0.008),
            speckle_density: rng.random_range(0.0_f32..1.).powi(2) * 0.28,
            speckle_radius_m: rng.random_range(0.00018..0.00065),
            speckle_color: [
                0.20,
                rng.random_range(0.10..0.18),
                rng.random_range(0.06..0.13),
            ],
            crack_spacing_m: rng.random_range(0.006..0.022),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.body_grain_m.is_finite()
            || !(0.0002..=0.004).contains(&self.body_grain_m)
            || !self.cloud_scale_m.is_finite()
            || !(0.003..=0.10).contains(&self.cloud_scale_m)
            || !self.turning_pitch_m.is_finite()
            || !(0.001..=0.020).contains(&self.turning_pitch_m)
            || !self.speckle_radius_m.is_finite()
            || !(0.00005..=0.002).contains(&self.speckle_radius_m)
            || !self.crack_spacing_m.is_finite()
            || !(0.003..=0.06).contains(&self.crack_spacing_m)
            || [
                self.reactive_mix,
                self.thickness_variation,
                self.flow,
                self.throwing,
                self.speckle_density,
            ]
            .iter()
            .chain(self.reactive_color.iter())
            .chain(self.speckle_color.iter())
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid fired ceramic/glaze program".into());
        }
        Ok(())
    }
    pub(super) fn evaluate(&self, c: &CoatingRecipe, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        let [u, v] = uv;
        let count = frequency(r.period_m, self.cloud_scale_m);
        let melt = deposit(
            uv,
            [count, count],
            0.12 + self.flow * 0.18,
            r.seed.wrapping_add(401),
        );
        let pool = smooth((melt - 0.25) * 1.5) * self.thickness_variation;
        let grains = frequency(r.period_m, self.body_grain_m);
        let body = periodic_noise(u, v, grains, grains, r.seed.wrapping_add(409)) - 0.5;
        let turns = frequency(r.period_m, self.turning_pitch_m);
        let turn_warp = (periodic_noise(u, v, 7, 3, r.seed.wrapping_add(419)) - 0.5) * 0.30;
        let rings = band(
            ((v * turns as f32 + turn_warp).rem_euclid(1.) - 0.5).abs(),
            0.16,
            turns as f32 / r.map_size(0) as f32,
        ) - 0.32;
        let speck_count = frequency(r.period_m, self.speckle_radius_m * 12.);
        let speck = cellular(uv, [speck_count, speck_count], r.seed.wrapping_add(431));
        let radius = self.speckle_radius_m / r.period_m * speck_count as f32 * (0.6 + speck.dye);
        let inclusion = disk(
            speck.radius,
            radius,
            speck_count as f32 / r.map_size(0) as f32,
        ) * smooth((self.speckle_density - speck.dye) * 18.);
        let cracks = if c.crackle > 0. {
            let n = frequency(r.period_m, self.crack_spacing_m);
            let cell = cellular(uv, [n, n], r.seed.wrapping_add(439));
            band(
                cell.edge * 0.5,
                0.000028 / r.period_m * n as f32,
                n as f32 / r.map_size(0) as f32,
            ) * c.crackle
        } else {
            0.
        };
        // Relative reflectance: the production base multiplier supplies each
        // item's pigment exactly once. Thickness/oxide inclusions share their
        // color, roughness and relief footprints.
        let matrix = tint([0.975; 3], 1. + body * c.pigment_variation - pool * 0.10);
        let reactive = smooth((melt - 0.30) * 1.7) * self.reactive_mix;
        let color = tint(
            mix(
                mix(matrix, self.reactive_color, reactive),
                self.speckle_color,
                inclusion,
            ),
            1. - cracks * 0.30,
        );
        Texel {
            color,
            height: r.relief_m
                * (body * (1. - c.gloss * 0.85) * 0.35
                    + rings * self.throwing * 0.65
                    + (melt - 0.5) * self.thickness_variation * 0.25
                    - inclusion * 0.08
                    - cracks * 0.10),
            roughness: (r.roughness + body * (1. - c.gloss) * 0.06 + reactive * 0.13 - pool * 0.07
                + inclusion * 0.12
                + cracks * 0.08)
                .clamp(0.075, 0.97),
            occlusion: (1. - cracks * 0.025).max(0.96),
        }
    }
}

#[cfg(test)]
mod replay_tests;
