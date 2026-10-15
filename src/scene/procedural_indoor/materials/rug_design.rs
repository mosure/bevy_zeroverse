//! Whole-rug dye/nap program, independent of the periodic floor-carpet field.
use super::{
    carpet::CarpetDesign,
    periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RugDesign {
    /// Stripe, stepped diamond, organic field and broken lattice; continuously mixed.
    pub weights: [f32; 4],
    pub scale: [f32; 2],
    pub phase: [f32; 2],
    /// Unit dye-field direction, precomputed to avoid per-texel trigonometry.
    pub direction: [f32; 2],
    pub patchwork: f32,
    pub warp: f32,
    pub center: f32,
    pub center_roundness: f32,
    pub border_width: f32,
    pub border_strength: f32,
    pub abrash: f32,
    pub wear: f32,
    pub quiet: f32,
}
impl RugDesign {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x5255474752414d);
        let mut weights: [f32; 4] = std::array::from_fn(|_| rng.random_range(0.0..1.0_f32).powi(3));
        let sum = weights.iter().sum::<f32>();
        if sum == 0. {
            weights = [0.25; 4];
        } else {
            // Small sampled weights are valid mixtures too. A denominator
            // floor would silently stop them summing to one on rare seeds.
            weights.iter_mut().for_each(|x| *x /= sum);
        }
        let angle = rng.random_range(0.0..std::f32::consts::TAU);
        Self {
            direction: [angle.cos(), angle.sin()],
            patchwork: if rng.random_bool(0.36) {
                rng.random_range(0.35..0.95)
            } else {
                0.
            },
            weights,
            scale: [rng.random_range(1.0..8.), rng.random_range(1.0..10.)],
            phase: [rng.random(), rng.random()],
            warp: rng.random_range(0.02..0.65),
            center: if rng.random_bool(0.5) {
                0.
            } else {
                rng.random_range(0.25..0.95)
            },
            center_roundness: rng.random(),
            border_width: if rng.random_bool(0.22) {
                0.
            } else {
                rng.random_range(0.035..0.16)
            },
            border_strength: rng.random_range(0.15..0.85),
            abrash: rng.random_range(0.02..0.20),
            wear: rng.random_range(0.0..0.30),
            quiet: if rng.random_bool(0.18) {
                rng.random_range(0.75..0.98)
            } else {
                rng.random_range(0.0..0.32)
            },
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self
            .weights
            .into_iter()
            .chain(self.phase)
            .chain([
                self.warp,
                self.center,
                self.center_roundness,
                self.border_width,
                self.border_strength,
                self.abrash,
                self.wear,
                self.quiet,
                self.patchwork,
            ])
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
            || self
                .scale
                .iter()
                .any(|v| !v.is_finite() || !(1.0..=16.).contains(v))
            || (self.weights.iter().sum::<f32>() - 1.).abs() > 0.001
            || self.direction.iter().any(|v| !v.is_finite())
            || (self.direction[0].powi(2) + self.direction[1].powi(2) - 1.).abs() > 0.001
        {
            return Err("invalid whole-rug program".into());
        }
        Ok(())
    }
    pub(super) fn evaluate(&self, c: &CarpetDesign, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        let [u, v] = uv;
        let noise = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
        let smooth = |t: f32| {
            let t = t.clamp(0., 1.);
            t * t * (3. - 2. * t)
        };
        let broad = noise(3, 5, 281);
        let warp = (noise(7, 5, 283) - 0.5) * self.warp;
        let du = (u - 0.5) * self.direction[0] - (v - 0.5) * self.direction[1] + 0.5;
        let dv = (u - 0.5) * self.direction[1] + (v - 0.5) * self.direction[0] + 0.5;
        let x = (du * self.scale[0] + self.phase[0] + warp).rem_euclid(1.);
        let y = (dv * self.scale[1] + self.phase[1] + warp * 0.63).rem_euclid(1.);
        let aa = self.scale[0].max(self.scale[1]) / r.map_size(1) as f32;
        let a = (x - 0.5).abs();
        let b = (y - 0.5).abs();
        let motifs = [
            smooth((c.line_width - a) / aa + 0.5),
            smooth((c.line_width * 0.65 - (a + b - 0.53).abs()) / aa + 0.5),
            smooth((broad - 0.32) / 0.36),
            smooth((c.line_width * 0.38 - a.min(b)) / aa + 0.5),
        ];
        let motif = self
            .weights
            .into_iter()
            .zip(motifs)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let d = [(u - 0.5).abs() * 2., (v - 0.5).abs() * 2.];
        let radius = (d[0] + d[1]) * (1. - self.center_roundness)
            + (d[0] * d[0] + d[1] * d[1]).sqrt() * self.center_roundness;
        let center = smooth((0.53 - radius) / (aa * 2.) + 0.5) * self.center;
        let center_rule = smooth((0.022 - (radius - 0.59).abs()) / (aa * 2.) + 0.5) * self.center;
        let edge = u.min(v).min(1. - u).min(1. - v);
        let border = if self.border_width > 0. {
            1. - smooth((edge - self.border_width) * r.map_size(1) as f32 * 0.5 + 0.5)
        } else {
            0.
        };
        let rule = if self.border_width > 0. {
            smooth(
                (0.007 - (edge - self.border_width - 0.018).abs()) * r.map_size(1) as f32 * 0.5
                    + 0.5,
            )
        } else {
            0.
        };
        let nap = noise(173, 191, 257) - 0.5;
        let fleck = noise(43, 47, 269) - 0.5;
        let drift = (noise(3, 47, 287) - 0.5) * self.abrash;
        let wear = self.wear * smooth((noise(5, 7, 293) - 0.35) / 0.55);
        let dye = (motif * c.contrast * (1. - self.quiet) * (1. - center * 0.65)).clamp(0., 1.);
        let border_mix =
            ((border * self.border_strength + rule * 0.38) * (1. - self.quiet * 0.7)).clamp(0., 1.);
        let patch = super::hash(
            (u * self.scale[0]).floor() as u32,
            (v * self.scale[1]).floor() as u32,
            r.seed.wrapping_add(301),
        );
        let color = [0, 1, 2].map(|i| {
            let field = c.palette[0][i] * (1. - dye) + c.palette[1][i] * dye;
            let patch_color = c.palette[2][i] * (1. - patch) + c.palette[1][i] * patch;
            let field = field * (1. - self.patchwork) + patch_color * self.patchwork;
            let center_color = field * (1. - center * 0.45) + c.palette[2][i] * center * 0.45;
            let edge_color = c.palette[2][i] * 0.65 + c.palette[1][i] * 0.35;
            let value = center_color * (1. - border_mix) + edge_color * border_mix;
            let value = value * (1. - center_rule * 0.5) + c.palette[1][i] * center_rule * 0.5;
            (value * (1. + drift + fleck * c.heather + nap * 0.035) * (1. - wear * 0.3)
                + c.palette[1][i] * wear * 0.3)
                .clamp(0.01, 0.95)
        });
        Texel {
            color,
            height: r.relief_m * (nap * 0.65 + fleck * c.pile * 0.35) * (1. - wear),
            roughness: (r.roughness + nap * 0.055 + fleck * 0.025 - wear * 0.06).clamp(0.80, 1.),
            occlusion: 0.97 + (nap + 0.5) * 0.03,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn low_weight_tail_remains_a_valid_normalized_program() {
        for seed in 0..100_000 {
            RugDesign::sample(seed)
                .validate()
                .unwrap_or_else(|error| panic!("rug seed {seed}: {error}"));
        }
    }
}
