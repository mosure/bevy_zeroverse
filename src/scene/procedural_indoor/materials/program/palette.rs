//! Correlated palette priors, independent of material rasterization.
use super::*;

/// Continuous, correlated interior pigments. Wall, ceiling and accent colours
/// share a design hue, while wood stays inside plausible stain reflectances.
/// These parameters describe materials, never RGB-only postprocessing effects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaletteProgram {
    pub hue: f32,
    pub harmony_angle: f32,
    pub pigment_strength: f32,
    pub wall_lightness: f32,
    pub ceiling_lightness: f32,
    pub furnishing_lightness: f32,
    pub furnishing_saturation: f32,
    pub wood_hue: f32,
    pub wood_lightness: f32,
    pub wood_saturation: f32,
}
impl PaletteProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 5720);
        let dark_wall = rng.random_bool(0.08);
        let neutral = rng.random_bool(0.52);
        let mut palette = Self {
            hue: rng.random_range(0.0..360.0),
            harmony_angle: if rng.random_bool(0.70) {
                rng.random_range(-42.0..42.0)
            } else {
                rng.random_range(145.0..215.0)
            },
            pigment_strength: if neutral {
                rng.random_range(0.005..0.065)
            } else {
                rng.random_range(0.07..0.28)
            },
            wall_lightness: if dark_wall {
                rng.random_range(0.18..0.42)
            } else {
                rng.random_range(0.53..0.94)
            },
            ceiling_lightness: rng.random_range(0.72..0.96),
            furnishing_lightness: rng.random_range(0.13..0.68),
            furnishing_saturation: rng.random_range(0.08..0.58),
            wood_hue: rng.random_range(16.0..48.0),
            wood_lightness: rng.random_range(0.10..0.65),
            wood_saturation: rng.random_range(0.12..0.63),
        };
        // Preserve the bright acoustical/plaster stock and the sampling of
        // other finishes. Mineral tints and dark overhead paint also occur.
        let mut ceiling = stream(seed, 0x4345494c5041494e);
        let finish = ceiling.random::<f32>();
        if finish < 0.055 {
            palette.ceiling_lightness = ceiling.random_range(0.29..0.52);
        } else if finish < 0.23 {
            palette.ceiling_lightness = ceiling.random_range(0.52..0.72);
        }
        palette
    }

    pub(super) fn ceiling_color(&self, seed: u64) -> [f32; 3] {
        if self.ceiling_lightness >= 0.72 {
            return Self::rgb(
                self.hue,
                self.pigment_strength * 0.16,
                self.ceiling_lightness,
            );
        }
        let mut pigment = stream(seed, 0x4345494c54494e54);
        let hue = if pigment.random_bool(0.45) {
            self.hue + pigment.random_range(-36.0..36.0)
        } else {
            pigment.random_range(0.0..360.0)
        };
        Self::rgb(
            hue,
            pigment.random_range(0.012..0.19),
            self.ceiling_lightness,
        )
    }
    /// A feature wall can coordinate with furnishings without inheriting their
    /// upholstery lightness or saturation. Sample paint on its own product stream.
    pub(super) fn feature_wall(&self, seed: u64) -> [f32; 3] {
        let mut ink = stream(seed, 0x57414c4c544f4e45);
        let hue =
            self.hue + self.harmony_angle * ink.random::<f32>() + ink.random_range(-34.0..34.0);
        let (saturation, lightness) = if ink.random_bool(0.65) {
            (ink.random_range(0.025..0.26), ink.random_range(0.46..0.87))
        } else {
            (ink.random_range(0.08..0.62), ink.random_range(0.16..0.65))
        };
        Self::rgb(hue, saturation, lightness)
    }
    pub fn rgb(hue: f32, saturation: f32, lightness: f32) -> [f32; 3] {
        let c = bevy::prelude::Color::hsl(
            hue.rem_euclid(360.0),
            saturation.clamp(0.0, 1.0),
            lightness.clamp(0.0, 1.0),
        )
        .to_srgba();
        [c.red, c.green, c.blue]
    }
}
