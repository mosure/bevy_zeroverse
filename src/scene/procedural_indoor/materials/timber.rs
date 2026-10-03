//! Uneven growth contours, cathedral cuts, knots and elongated open pores.
use super::{hash, periodic_noise, program::MaterialRecipe, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WoodFinish {
    #[serde(default)]
    pub seed: u64,
    pub stain_color: [f32; 3],
    pub stain_strength: f32,
    pub bleach: f32,
    pub pore_fill: f32,
    pub ring_contrast: f32,
    pub clearcoat: f32,
    pub coat_roughness: f32,
}
impl WoodFinish {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x574f4f4446494e);
        let neutral = rng.random_bool(0.25);
        let c = bevy::prelude::Color::hsl(
            rng.random_range(12.0..44.0),
            if neutral {
                rng.random_range(0.0..0.10)
            } else {
                rng.random_range(0.12..0.55)
            },
            rng.random_range(0.035..0.35),
        )
        .to_srgba();
        Self {
            seed,
            stain_color: [c.red, c.green, c.blue],
            stain_strength: rng.random_range(0.0_f32..1.0).powf(1.5) * 0.90,
            bleach: rng.random_range(0.0_f32..1.0).powi(3) * 0.65,
            pore_fill: rng.random_range(0.0..0.95),
            ring_contrast: rng.random_range(0.35..1.25),
            clearcoat: rng.random_range(0.0_f32..1.0).powi(2) * 0.85,
            coat_roughness: rng.random_range(0.045..0.42),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self
            .stain_color
            .iter()
            .chain(
                [
                    self.stain_strength,
                    self.bleach,
                    self.pore_fill,
                    self.clearcoat,
                    self.coat_roughness,
                ]
                .iter(),
            )
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !self.ring_contrast.is_finite()
            || !(0.1..=1.5).contains(&self.ring_contrast)
        {
            return Err("invalid timber stain/finish program".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, mat: &mut bevy::prelude::StandardMaterial) {
        mat.clearcoat = self.clearcoat;
        mat.clearcoat_perceptual_roughness = self.coat_roughness;
        mat.reflectance = 0.45;
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, [u, v]: [f32; 2]) -> super::program::Texel {
        use super::field::*;
        let g = grain(r, u, v);
        let absorbed = (self.stain_strength * (0.92 + (0.65 - g) * 0.22)).clamp(0., 1.);
        let base = mix(
            mix(r.color, self.stain_color, absorbed),
            [0.88, 0.87, 0.82],
            self.bleach,
        );
        let gain = (1. + (g - 0.58) * r.contrast * self.ring_contrast * 2.).clamp(0.35, 1.20);
        super::program::Texel {
            color: tint(base, gain),
            height: (g - 0.5) * r.relief_m * (1. - self.pore_fill * 0.85),
            roughness: (r.roughness + (0.55 - g) * 0.18 * (1. - self.clearcoat * 0.65))
                .clamp(0.12, 0.95),
            occlusion: 1. - (0.5 - g).max(0.) * 0.04 * (1. - self.pore_fill),
        }
    }
}

pub(super) fn bark(r: &MaterialRecipe, [u, v]: [f32; 2]) -> super::program::Texel {
    use super::field::*;
    let warp = periodic_noise(u, v, 3, 5, r.seed) - 0.5;
    let ridges = deposit([u + warp * 0.04, v], [23, 3], 0.02, r.seed.wrapping_add(19));
    let cracks = smooth((0.35 - ridges) * 7.);
    let cross = smooth((0.24 - periodic_noise(u, v, 5, 29, r.seed.wrapping_add(23))) * 6.);
    let grain = periodic_noise(u, v, 79, 13, r.seed.wrapping_add(29)) - 0.5;
    super::program::Texel {
        color: tint(
            r.color,
            (1. + grain * 0.18 - cracks * 0.42 - cross * 0.14).clamp(0.35, 1.2),
        ),
        height: r.relief_m * ((ridges - 0.5) * 0.6 - cracks * 0.35 - cross * 0.12),
        roughness: (r.roughness + cracks * 0.08 + grain * 0.07).clamp(0.70, 1.),
        occlusion: 1. - cracks * 0.06 - cross * 0.02,
    }
}

pub(super) fn grain(r: &MaterialRecipe, mut u: f32, mut v: f32) -> f32 {
    let mut seed = r.seed;
    if r.surface == Surface::Floor {
        let [nx, ny] = r.floor_repetitions(0);
        let x = (u * nx).floor() as u32;
        let stagger = (x % 2) as f32 * 0.5;
        let y = ((v * ny + stagger).floor() as u32) % ny as u32;
        // Different cuts of one species share a spectrum but never continue
        // the very same growth ring across a joint into the next plank.
        u = (u * nx).fract() + hash(x, y, seed.wrapping_add(79));
        v = (v * ny + stagger).fract() + hash(x, y, seed.wrapping_add(83));
        seed = seed
            .wrapping_add(u64::from(x) * 0x9e3779b9)
            .wrapping_add(u64::from(y) * 0x85ebca6b);
    }
    let noise = |u, v, x, y, salt| periodic_noise(u, v, x, y, seed.wrapping_add(salt));
    let rings = r.grain_frequency * 0.30;
    // Intersect cylindrical growth with a continuously offset/tapered log cut.
    // Near the pith this exposes nested cathedral contours; deeper/rift cuts
    // expose elongated rings. Periodic coordinates make the volume tileable.
    let cut = hash(7, 19, seed);
    let bow = (TAU * v).cos() + 0.23 * (TAU * (2. * v + cut)).cos();
    let growth = noise(u, v, 3, 2, 13) - 0.5;
    let x = 0.55 * (TAU * u).sin() + cut * 0.85 + growth * r.warp;
    let depth = 0.08 + cut * cut * 1.4 + (1. - cut) * (0.32 + 0.22 * bow);
    let phase = (x * x + depth * depth).sqrt() * rings + growth * (0.30 + r.warp * 6.);
    let ring_id = phase.floor().max(0.) as u32;
    let width = 0.07 + 0.14 * hash(ring_id, 5, seed);
    // Earlywood gradually darkens into latewood, then starts a new growth year.
    // Ring-specific width/pigmentation avoids uniform barcode-like intervals.
    let annual = phase.rem_euclid(1.);
    let late = smooth((annual - (1. - width)) / width);
    let pigmentation = 0.7 + 0.3 * hash(ring_id, 31, seed);
    let fibre_u = u + bow * (1. - cut) * 0.04 + growth * 0.06;
    let fibres = noise(fibre_u, v, 79, 7, 47);
    let pores = smooth((noise(fibre_u, v, 109, 13, 91) - 0.62) * 5.);
    let mut value = 0.70 + 0.16 * growth + 0.09 * (fibres - 0.5)
        - (0.18 * annual * annual + 0.32 * late) * pigmentation
        - 0.12 * pores;
    // Sparse elliptical branch knots bend the surrounding fibres. Wrapped
    // distances keep both colour and its relief derivative continuous at seams.
    let knot_amount = smooth((hash(43, 9, seed) - 0.38) * 2.);
    let dx = wrapped(u - hash(17, 23, seed));
    let dy = wrapped(v - hash(37, 41, seed));
    let radius = (dx * dx * 100. + dy * dy * 22.).sqrt();
    let influence = (1. - smooth((radius - 0.30) / 0.70)) * knot_amount;
    let knot = 0.30 + 0.13 * (radius * 37. + growth * 4.).sin() - 0.22 * (-radius * 10.).exp();
    value += (knot - value) * influence;
    value.clamp(0., 1.)
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn wrapped(t: f32) -> f32 {
    (t + 0.5).rem_euclid(1.) - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staggered_plank_has_no_colour_seam_at_the_texture_wrap() {
        let mut r = super::super::program::sample(91).remove(Surface::Floor as usize);
        r.phase = [0.0, 0.0];
        r.layers.as_mut().unwrap().quarter_turn = 0;
        r.period_m = 2.3;
        r.panel_count = [5, 3];
        let [nx, _] = r.floor_repetitions(0);
        let a = r.evaluate(1.5 / nx, 0.000001, 0);
        let b = r.evaluate(1.5 / nx, 0.999999, 0);
        assert!(
            (a.0 - b.0).abs() < 0.001 && (a.1 - b.1).abs() < 0.000001 && (a.2 - b.2).abs() < 0.001,
            "{a:?} vs {b:?}"
        );
    }
}
