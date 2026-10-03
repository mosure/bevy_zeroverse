//! A periodic weaving program: individual irregular yarns pass over/under one
//! another. Dyed yarns, backing, crimp and filaments share the same surface field.
use super::{
    hash, periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextileRecipe {
    pub yarns: [u32; 2],
    pub repeat: u32,
    pub float_length: u32,
    pub advance: u32,
    pub bundle: u32,
    pub herringbone: bool,
    pub width: [f32; 2],
    pub slub: f32,
    pub crimp: f32,
    pub twist: f32,
    pub fuzz: f32,
    pub lustre: f32,
    pub dye_variation: f32,
    pub yarn_tint: [[f32; 3]; 2],
    pub pile: f32,
}
impl TextileRecipe {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x5745415645);
        let repeat = rng.random_range(2..=7);
        let bundle = rng.random_range(1..=3);
        // Every topology closes at the tile edges, including herringbone turns.
        let unit = repeat * bundle * 2;
        let yarns = [0, 1].map(|_| rng.random_range(16u32.div_ceil(unit)..=48 / unit) * unit);
        Self {
            yarns,
            repeat,
            bundle,
            float_length: rng.random_range(1..repeat),
            advance: rng.random_range(1..repeat),
            herringbone: rng.random_bool(0.25),
            width: [rng.random_range(0.62..0.91), rng.random_range(0.60..0.91)],
            slub: rng.random_range(0.01..0.20),
            crimp: rng.random_range(0.12..0.38),
            twist: rng.random_range(-1.5..1.5),
            fuzz: rng.random_range(0.05..0.95),
            lustre: rng.random_range(0.0..0.85),
            dye_variation: rng.random_range(0.02..0.22),
            yarn_tint: [0, 1].map(|_| [0, 1, 2].map(|_| rng.random_range(0.88..1.0))),
            pile: 0.,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=7).contains(&self.repeat) || !(1..=3).contains(&self.bundle) {
            return Err("invalid textile repeat".into());
        }
        let unit = self.repeat * self.bundle * 2;
        if !(1..self.repeat).contains(&self.float_length)
            || !(1..self.repeat).contains(&self.advance)
            || self
                .yarns
                .iter()
                .any(|n| !(16..=48).contains(n) || !n.is_multiple_of(unit))
            || self
                .width
                .iter()
                .any(|v| !v.is_finite() || !(0.5..=0.99).contains(v))
            || [
                self.slub,
                self.crimp,
                self.fuzz,
                self.lustre,
                self.dye_variation,
                self.pile,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !self.twist.is_finite()
            || self.twist.abs() > 2.
            || self
                .yarn_tint
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.5..=1.0).contains(v))
        {
            return Err("invalid textile weaving program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, r: &MaterialRecipe, mat: &mut bevy::prelude::StandardMaterial) {
        mat.clearcoat = 0.;
        mat.reflectance = 0.45;
        // Woven sheen is an aggregate directional GGX approximation. No claim
        // of a measured fibre BSDF or additional multiple-scattering cloth lobe.
        mat.anisotropy_strength = self.lustre * (1. - self.fuzz * 0.65) * 0.65;
        // Warp runs along V before the baked quarter-turn. The GGX direction
        // lives in the untransformed mesh tangent frame.
        let turn = r.layers.as_ref().map_or(0., |l| l.quarter_turn as f32);
        mat.anisotropy_rotation = (1. - turn) * std::f32::consts::FRAC_PI_2;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        let [u, v] = uv;
        let n = self.yarns.map(|n| n as f32);
        let x = u * n[0];
        let y = v * n[1];
        let ix = x.floor() as u32 % self.yarns[0];
        let iy = y.floor() as u32 % self.yarns[1];
        let noise = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
        let slub = [noise(self.yarns[0], 3, 11), noise(3, self.yarns[1], 17)];
        let jitter = [hash(ix, 0, r.seed) - 0.5, hash(iy, 1, r.seed) - 0.5];
        let profile = |p: f32, w: f32| {
            let q = (p / (w * 0.5)).abs();
            // Rounded yarn crowns taper smoothly into gaps; no full-tile wave.
            (1. - q * q).max(0.).sqrt() * (1. - smooth((q - 0.82) / 0.18))
        };
        let p = [
            x.fract() - 0.5 - jitter[0] * 0.05,
            y.fract() - 0.5 - jitter[1] * 0.05,
        ];
        let crown =
            [0, 1].map(|i| profile(p[i], self.width[i] * (1. + self.slub * (slub[i] - 0.5))));
        let row = iy / self.bundle;
        let row = if self.herringbone {
            let t = row % (2 * self.repeat);
            t.min(2 * self.repeat - 1 - t)
        } else {
            row
        };
        let warp_over = ((ix / self.bundle + row * self.advance) % self.repeat < self.float_length)
            as u32 as f32;
        // Crimp depresses the lower yarn at a crossing, not along the entire UV.
        let h = [
            crown[0] * (1. - self.crimp * crown[1] * (1. - warp_over)),
            crown[1] * (1. - self.crimp * crown[0] * warp_over),
        ];
        let top = if h[0] + h[1] > 0.0001 {
            h[0] / (h[0] + h[1])
        } else {
            0.5
        };
        let coverage = crown[0].max(crown[1]);
        let dye = [hash(ix, 13, r.seed), hash(iy, 19, r.seed)];
        let bands = if let Some(l) = &r.layers {
            [0, 1].map(|axis| {
                let count = l.bands[axis];
                let index = [ix, iy][axis];
                if count > 0 && (index * count / self.yarns[axis]).is_multiple_of(3) {
                    1. - l.stripe_strength * 0.25
                } else {
                    1.
                }
            })
        } else {
            [1.; 2]
        };
        let twist = self.twist * (noise(3, 3, 83) - 0.5);
        let filament = [ridge(x * 2. + twist), ridge(y * 2. - twist)];
        let fibres = top * filament[0] + (1. - top) * filament[1];
        let nap = noise(79, 83, 73) - 0.5;
        let yarn = [0, 1, 2].map(|c| {
            let a = self.yarn_tint[0][c] * (0.94 + self.dye_variation * (dye[0] - 0.5)) * bands[0];
            let b = self.yarn_tint[1][c] * (0.94 + self.dye_variation * (dye[1] - 0.5)) * bands[1];
            (0.76 + (a * top + b * (1. - top) - 0.76) * coverage + nap * self.fuzz * 0.035)
                .clamp(0.25, 1.)
        });
        let height = r.relief_m
            * (h[0].max(h[1]) - 0.5 + fibres * 0.045 * (1. - self.fuzz) + nap * self.fuzz * 0.09);
        let pile = noise(37, 41, 101);
        Texel {
            color: yarn.map(|c| (c * (1. - self.pile * 0.08 * (pile - 0.5))).clamp(0., 1.)),
            height: height * (1. - self.pile * 0.45) + self.pile * r.relief_m * (pile - 0.5),
            roughness: (r.roughness + self.fuzz * nap * 0.10 + (1. - coverage) * 0.10
                - self.lustre * (fibres - 0.5) * 0.06)
                .clamp(0.30, 1.),
            occlusion: 0.93 + 0.07 * coverage,
        }
    }
}
fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn ridge(t: f32) -> f32 {
    1. - 2. * (t.rem_euclid(1.) - 0.5).abs()
}
