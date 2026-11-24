//! Person-specific finishes. The existing room textures supply pore and yarn detail.
use super::{stream, HumanSurface};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complexion {
    pub warmth: f32,
    pub flush: f32,
    pub lip_moisture: f32,
    /// Frequency and phase in the shared Anny skin atlas.
    pub pore_repeat: f32,
    pub pore_phase: [f32; 2],
    /// Brow arch position from the bridge towards the temple.
    pub brow_peak: f32,
    pub brow_slope: f32,
}

impl Complexion {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x534b494e544f4e45);
        Self {
            warmth: rng.random_range(-1.0..1.0),
            flush: rng.random_range(-1.0..1.0),
            lip_moisture: rng.random_range(0.08..0.90),
            brow_peak: rng.random_range(0.48..0.76),
            brow_slope: rng.random_range(-0.005..0.005),
            pore_repeat: rng.random_range(18.0..32.0),
            pore_phase: [rng.random(), rng.random()],
        }
    }
    pub(super) fn tint(&self, [r, g, b]: [f32; 3]) -> [f32; 3] {
        [
            r * (1. + 0.024 * self.flush),
            g * (1. + 0.028 * self.warmth - 0.030 * self.flush),
            b * (1. - 0.078 * self.warmth - 0.018 * self.flush),
        ]
    }
    pub(super) fn valid(&self) -> bool {
        (14.0..=38.).contains(&self.pore_repeat)
            && self.pore_phase.iter().all(|v| (0.0..=1.).contains(v))
            && (-1.0..=1.).contains(&self.warmth)
            && (-1.0..=1.).contains(&self.flush)
            && (0.0..=1.).contains(&self.lip_moisture)
            && (0.4..=0.8).contains(&self.brow_peak)
            && (-0.008..=0.008).contains(&self.brow_slope)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerFinishes {
    /// Garment visible inside the front opening.
    pub shirt: [f32; 3],
    /// Trouser and undershirt grain relative to the cloth atlas.
    pub weave_scale: [f32; 2],
    pub weave_angle: [f32; 2],
    pub roughness: [f32; 2],
    pub button_side: f32,
    pub button_radius: f32,
    pub fastening_density: f32,
    pub pocket_side: f32,
    pub pocket_offset: f32,
}

impl LayerFinishes {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4c41594552444945);
        let (light, saturation) = match rng.random_range(0..10) {
            0..=3 => (rng.random_range(0.72..0.94), rng.random_range(0.02..0.25)),
            4..=6 => (rng.random_range(0.14..0.48), rng.random_range(0.025..0.20)),
            _ => (rng.random_range(0.32..0.77), rng.random_range(0.18..0.49)),
        };
        let c = Color::hsl(rng.random_range(0.0..360.0), saturation, light).to_srgba();
        Self {
            shirt: [c.red, c.green, c.blue],
            weave_scale: std::array::from_fn(|_| rng.random_range(0.62..1.68)),
            weave_angle: std::array::from_fn(|_| rng.random_range(-0.27..0.27)),
            roughness: std::array::from_fn(|_| rng.random_range(0.72..0.98)),
            button_side: if rng.random() { 1. } else { -1. },
            button_radius: rng.random_range(0.0045..0.0080),
            fastening_density: rng.random(),
            pocket_side: if rng.random_bool(0.75) { 1. } else { -1. },
            pocket_offset: rng.random_range(0.069..0.102),
        }
    }
    pub(super) fn cloth(&self, surface: HumanSurface) -> Option<(f32, f32, f32)> {
        let i = match surface {
            HumanSurface::Trousers => 0,
            HumanSurface::Shirt => 1,
            _ => return None,
        };
        Some((self.weave_scale[i], self.weave_angle[i], self.roughness[i]))
    }
    pub(super) fn valid(&self) -> bool {
        self.shirt.iter().all(|v| (0.0..=1.).contains(v))
            && self.weave_scale.iter().all(|v| (0.4..=2.).contains(v))
            && self.weave_angle.iter().all(|v| (-0.5..=0.5).contains(v))
            && self.roughness.iter().all(|v| (0.5..=1.).contains(v))
            && self.button_side.abs() == 1.
            && self.pocket_side.abs() == 1.
            && (0.003..=0.012).contains(&self.button_radius)
            && (0.0..=1.).contains(&self.fastening_density)
            && (0.06..=0.12).contains(&self.pocket_offset)
    }
}
