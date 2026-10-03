//! Irregular dermal grain, rounded pores, restrained creases and variable finish.
use super::{
    hash, periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeatherRecipe {
    pub cells: [u32; 2],
    pub stretch: f32,
    pub roundness: f32,
    pub pore_depth: f32,
    pub crease_strength: f32,
    pub mottling: f32,
    pub polish: f32,
    pub nap: f32,
    pub coat: f32,
    pub coat_roughness: f32,
}
impl LeatherRecipe {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4c454154484552);
        Self {
            cells: [rng.random_range(18..=52), rng.random_range(18..=52)],
            stretch: rng.random_range(0.65..1.5),
            roundness: rng.random_range(0.45..1.6),
            pore_depth: rng.random_range(0.02..0.22),
            crease_strength: rng.random_range(0.0..0.22),
            mottling: rng.random_range(0.03..0.25),
            polish: rng.random_range(0.0..1.0),
            nap: rng.random_range(0.0_f32..1.0).powi(3),
            coat: rng.random_range(0.0..0.35),
            coat_roughness: rng.random_range(0.14..0.40),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.cells.iter().any(|n| !(12..=64).contains(n))
            || !self.stretch.is_finite()
            || !(0.5..=2.).contains(&self.stretch)
            || !self.roundness.is_finite()
            || !(0.3..=2.).contains(&self.roundness)
            || [
                self.pore_depth,
                self.crease_strength,
                self.mottling,
                self.polish,
                self.nap,
                self.coat,
                self.coat_roughness,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid leather grain program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.clearcoat = self.coat * self.polish * (1. - self.nap);
        mat.clearcoat_perceptual_roughness = self.coat_roughness;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> Texel {
        let noise = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
        let [nx, ny] = self.cells;
        // A low-frequency warp changes cell shape and size while preserving seams.
        let x = (u + (noise(3, 5, 11) - 0.5) * 0.008) * nx as f32;
        let y = (v + (noise(5, 3, 19) - 0.5) * 0.008) * ny as f32;
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let mut nearest = [f32::INFINITY; 2];
        let mut cell = [0, 0];
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (ix + dx, iy + dy);
                let (hx, hy) = (
                    cx.rem_euclid(nx as i32) as u32,
                    cy.rem_euclid(ny as i32) as u32,
                );
                let px = cx as f32 + 0.15 + 0.70 * hash(hx, hy, r.seed);
                let py = cy as f32 + 0.15 + 0.70 * hash(hx, hy, r.seed.wrapping_add(31));
                let d = (px - x).powi(2) + (py - y).powi(2);
                if d < nearest[0] {
                    nearest = [d, nearest[0]];
                    cell = [hx, hy];
                } else if d < nearest[1] {
                    nearest[1] = d;
                }
            }
        }
        let edge = smooth(((nearest[1].sqrt() - nearest[0].sqrt()) * 9.).clamp(0., 1.));
        let crown = (1. - nearest[0] * 0.50).clamp(0., 1.).powf(self.roundness);
        let grain = edge * (0.55 + 0.45 * crown);
        // Sparse depressions, with pigment/roughness variation from the same field.
        let pore = smooth((noise(83, 79, 47) - 0.68) * 5.);
        let creases = 1. - smooth((noise(7, 9, 71) - 0.48).abs() * 24.);
        let cell_dye = hash(cell[0], cell[1], r.seed.wrapping_add(91)) - 0.5;
        let mottle = (noise(11, 13, 101) - 0.5) * self.mottling;
        let dye = 0.93 + cell_dye * self.mottling * 0.35 + mottle - pore * 0.035;
        let nap = noise(97, 91, 113) - 0.5;
        Texel {
            color: [(dye + nap * self.nap * 0.08).clamp(0.25, 1.); 3],
            height: r.relief_m
                * ((grain - 0.5) * (1. - self.nap * 0.8)
                    - pore * self.pore_depth
                    - creases * self.crease_strength
                    + nap * self.nap * 0.20),
            roughness: (r.roughness
                + (1. - grain) * 0.16 * (1. - self.nap)
                + pore * 0.08
                + creases * 0.04
                + nap * self.nap * 0.12)
                .clamp(0.18, 1.),
            occlusion: 0.95 + 0.05 * grain,
        }
    }
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
