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
        // Finishing a hide couples its nap, pore definition, dye uniformity and
        // sheen. Avoid independently choosing thick shiny lacquer and suede nap.
        let polish = rng.random_range(0.0_f32..1.0);
        let nap = rng.random_range(0.0_f32..1.0).powi(2) * (1. - polish).powi(2);
        let coat = rng.random_range(0.0..0.70) * (1. - nap);
        Self {
            cells: [rng.random_range(18..=52), rng.random_range(18..=52)],
            stretch: rng.random_range(0.72..1.40),
            roundness: rng.random_range(0.45..1.6),
            pore_depth: rng.random_range(0.02..0.22) * (1. - polish * 0.55),
            crease_strength: rng.random_range(0.0..0.22) * (1. - coat * 0.60),
            mottling: rng.random_range(0.03..0.32) * (1. - coat * 0.65),
            polish,
            nap,
            coat,
            coat_roughness: rng.random_range(0.10..0.35) + nap * 0.20,
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
        let [nx, ny] = self.stretched_cells();
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
    /// Stretching the skin changes the density along both principal axes.
    /// Resolving integer periodic cell counts keeps the three-by-three neighbor
    /// search complete, instead of distorting its distance metric and missing pores.
    pub fn stretched_cells(&self) -> [u32; 2] {
        [
            (self.cells[0] as f32 / self.stretch)
                .round()
                .clamp(6., 128.) as u32,
            (self.cells[1] as f32 * self.stretch)
                .round()
                .clamp(6., 128.) as u32,
        ]
    }
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}

#[cfg(test)]
mod tests {
    use super::super::{program, Surface};
    use super::*;

    #[test]
    fn hide_finishes_replay_and_do_not_combine_suede_with_shiny_topcoat() {
        let mut rough = 0;
        let mut glossy = 0;
        for seed in 0..512 {
            let l = LeatherRecipe::sample(seed);
            assert_eq!(l, LeatherRecipe::sample(seed));
            l.validate().unwrap();
            if l.nap > 0.30 {
                assert!(l.polish < 0.46);
                rough += 1;
            }
            if l.polish > 0.80 {
                assert!(l.nap < 0.04);
                glossy += 1;
            }
        }
        assert!(
            rough > 15 && glossy > 60,
            "finish distribution collapsed: {rough} {glossy}"
        );
    }

    #[test]
    fn authored_leather_stretch_changes_grain_without_breaking_periodic_pbr_channels() {
        let r = program::sample(202)[Surface::Leather as usize].clone();
        let mut l = r.leather.clone().unwrap();
        l.cells = [32, 32];
        l.stretch = 0.5;
        let narrow = l.stretched_cells();
        let before = l.evaluate(&r, [0.37, 0.64]);
        l.stretch = 2.;
        assert_ne!(narrow, l.stretched_cells());
        assert_ne!(before.height, l.evaluate(&r, [0.37, 0.64]).height);
        for t in [0., 0.23, 0.67, 1.] {
            for (a, b) in [
                (l.evaluate(&r, [0., t]), l.evaluate(&r, [1., t])),
                (l.evaluate(&r, [t, 0.]), l.evaluate(&r, [t, 1.])),
            ] {
                for (x, y) in a.color.into_iter().zip(b.color) {
                    assert!((x - y).abs() < 0.0001);
                }
                assert!((a.height - b.height).abs() < 1e-7);
                assert!((a.roughness - b.roughness).abs() < 0.0001);
                assert!((0.18..=1.).contains(&a.roughness));
                assert!((0.90..=1.).contains(&a.occlusion));
            }
        }
    }
}
