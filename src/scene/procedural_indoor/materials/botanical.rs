//! Blade-aligned venation, chlorophyll patches and wax; leaf UVs are an atlas.
use super::{
    field::*,
    hash, periodic_noise,
    program::{MaterialRecipe, Texel},
    Surface,
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeafRecipe {
    pub vein_pairs: u32,
    pub vein_rake: f32,
    pub vein_curve: f32,
    pub vein_width: f32,
    pub midrib_width: f32,
    pub chlorophyll_variation: f32,
    pub variegation: f32,
    pub patch_cells: [u32; 2],
    pub pale_color: [f32; 3],
    pub wax: f32,
    pub transmission: f32,
    /// Reference blade width/length for converting atlas derivatives to metres.
    pub reference_size_m: [f32; 2],
}
impl LeafRecipe {
    pub fn sample(seed: u64, surface: Surface) -> Self {
        let mut rng = stream(seed, 0x4c454146);
        let n = rng.random_range(0.62..0.90);
        Self {
            vein_pairs: rng.random_range(5..=18),
            vein_rake: rng.random_range(0.15..0.70),
            vein_curve: rng.random_range(-0.18..0.18),
            vein_width: rng.random_range(0.0015..0.006),
            midrib_width: rng.random_range(0.003..0.013),
            chlorophyll_variation: rng.random_range(0.03..0.30),
            variegation: if surface == Surface::LeafVariegated {
                rng.random_range(0.20..0.85)
            } else {
                rng.random_range(0.0..0.10)
            },
            patch_cells: [rng.random_range(2..=7), rng.random_range(3..=11)],
            pale_color: [
                n,
                n * rng.random_range(0.97..1.03),
                n * rng.random_range(0.65..0.91),
            ],
            wax: rng.random_range(0.0..0.80),
            transmission: rng.random_range(0.05..0.38),
            reference_size_m: [rng.random_range(0.04..0.10), rng.random_range(0.12..0.30)],
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(3..=24).contains(&self.vein_pairs)
            || self.patch_cells.iter().any(|n| !(1..=16).contains(n))
            || [
                self.vein_rake,
                self.vein_width,
                self.midrib_width,
                self.chlorophyll_variation,
                self.variegation,
                self.wax,
                self.transmission,
            ]
            .iter()
            .chain(self.pale_color.iter())
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !self.vein_curve.is_finite()
            || self.vein_curve.abs() > 0.3
            || self.vein_width <= 0.
            || self.midrib_width <= 0.
            || self
                .reference_size_m
                .iter()
                .any(|v| !v.is_finite() || !(0.02..=0.5).contains(v))
        {
            return Err("invalid botanical material program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.diffuse_transmission = self.transmission;
        mat.clearcoat = self.wax * 0.25;
        mat.clearcoat_perceptual_roughness = 0.18 + (1. - self.wax) * 0.20;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
        let x = u - 0.5;
        let side = u32::from(x > 0.);
        let branch = v - x.abs() * self.vein_rake - x * x * self.vein_curve;
        let row = (branch * self.vein_pairs as f32).round();
        let jitter = (hash(row.max(0.) as u32, side, r.seed) - 0.5) * 0.15;
        let d = ((branch * self.vein_pairs as f32 - row - jitter) / self.vein_pairs as f32).abs();
        let secondary = (1. - smooth(d / self.vein_width)) * smooth(x.abs() / self.midrib_width);
        let midrib = 1. - smooth(x.abs() / self.midrib_width);
        let n = |nx, ny, s| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(s));
        let cells = deposit([u, v], self.patch_cells, 0.08, r.seed.wrapping_add(29));
        let patch = smooth((cells - (0.88 - self.variegation * 0.52)) * 9.);
        let green = tint(
            r.color,
            1. + (n(7, 11, 41) - 0.5) * self.chlorophyll_variation,
        );
        let veins = (midrib * 0.20 + secondary * 0.06).min(0.25);
        let color = mix(mix(green, self.pale_color, patch), self.pale_color, veins);
        let micro = n(83, 79, 47) - 0.5;
        Texel {
            color,
            height: r.relief_m * (midrib * 0.65 + secondary * 0.25 + micro * 0.12),
            roughness: (r.roughness + micro * 0.06 + veins * 0.12 + patch * 0.03).clamp(0.20, 0.85),
            occlusion: 1.,
        }
    }
}
