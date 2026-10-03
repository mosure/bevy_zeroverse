//! Paint, plaster/knockdown relief and glazed ceramic with independent finish.
use super::{
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
    Surface,
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoatingRecipe {
    pub cells: [u32; 2],
    pub texture_mix: f32,
    pub knockdown: f32,
    pub roller: f32,
    pub trowel: f32,
    pub pinholes: f32,
    pub pigment_variation: f32,
    pub gloss: f32,
    pub clearcoat: f32,
    pub coat_roughness: f32,
    pub crackle: f32,
}
impl CoatingRecipe {
    pub fn sample(seed: u64, surface: Surface) -> Self {
        let mut rng = stream(seed, 0x434f4154494e47);
        let glaze = surface == Surface::Ceramic;
        let gloss = if glaze {
            rng.random_range(0.45..0.95)
        } else {
            rng.random_range(0.0_f32..1.0).powi(2) * 0.80
        };
        Self {
            cells: [rng.random_range(32..=64), rng.random_range(32..=64)],
            texture_mix: rng.random_range(0.0_f32..1.0).powi(2),
            knockdown: rng.random_range(0.0..0.8),
            roller: rng.random_range(0.0..0.75),
            trowel: if glaze {
                0.
            } else {
                rng.random_range(0.0_f32..1.0).powi(2) * 0.65
            },
            pinholes: rng.random_range(0.0..if glaze { 0.07 } else { 0.20 }),
            pigment_variation: rng.random_range(0.005..0.055),
            gloss,
            clearcoat: if glaze {
                rng.random_range(0.25..0.85)
            } else {
                0.
            },
            coat_roughness: rng.random_range(0.06..0.28),
            crackle: if glaze && rng.random_bool(0.30) {
                rng.random_range(0.05..0.35)
            } else {
                0.
            },
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.iter().any(|n| !(16..=96).contains(n))
            || [
                self.texture_mix,
                self.knockdown,
                self.roller,
                self.trowel,
                self.pinholes,
                self.pigment_variation,
                self.gloss,
                self.clearcoat,
                self.coat_roughness,
                self.crackle,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid wall/coating program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.clearcoat = self.clearcoat;
        mat.clearcoat_perceptual_roughness = self.coat_roughness;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
        let n = |x, y, s| periodic_noise(u, v, x, y, r.seed.wrapping_add(s));
        let micro = n(self.cells[0], self.cells[1], 11);
        // Flattening the highest peaks models knocking down sprayed plaster.
        let stipple = smooth((micro - 0.28) * 1.7).min(1. - self.knockdown * 0.45);
        let roller = n(7, 53, 19) - 0.5;
        let trowel =
            smooth((deposit([u, v], [5, 9], 0.12, r.seed.wrapping_add(23)) - 0.45) * 4.) - 0.5;
        let pores = smooth((0.24 - micro) * 5.) * self.pinholes;
        let cracks = if self.crackle > 0. {
            (1. - smooth(cellular([u, v], [19, 23], r.seed.wrapping_add(31)).edge * 55.))
                * self.crackle
        } else {
            0.
        };
        let pigment = n(13, 17, 41) - 0.5;
        Texel {
            color: tint(
                r.color,
                1. + pigment * self.pigment_variation - pores * 0.08 - cracks * 0.16,
            ),
            height: r.relief_m
                * ((stipple - 0.5) * (0.20 + self.texture_mix * 0.65)
                    + roller * self.roller * 0.12
                    + trowel * self.trowel * 0.35
                    - pores * 0.5
                    - cracks * 0.20),
            roughness: (r.roughness
                + (micro - 0.5) * 0.055 * (1. - self.gloss * 0.6)
                + pores * 0.06
                + cracks * 0.10)
                .clamp(0.08, 1.),
            occlusion: (1. - pores * 0.02 - cracks * 0.015).clamp(0.95, 1.),
        }
    }
}
