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
        Self {
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
        }
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
