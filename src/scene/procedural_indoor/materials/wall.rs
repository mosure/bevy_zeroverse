//! Mesoscale wall covering over the existing paint/plaster substrate. The
//! continuously mixed fields affect pigment, metric relief and GGX roughness;
//! they do not bake illumination into base color. CPU and GPU evaluate once.
use super::{periodic_noise, program::MaterialRecipe, Surface};
use crate::scene::procedural_indoor::layout::stream;
use bevy::prelude::Vec2;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A bounded selection of opaque primary-room wall faces. There is one shared
/// feature-wall material; windows/reveals and the exterior back stay painted.
pub(crate) fn feature_edges(
    scene: &crate::scene::procedural_indoor::layout::IndoorManifest,
) -> Vec<usize> {
    let Some(envelope) = &scene.envelope else {
        return Vec::new();
    };
    if scene
        .program
        .as_ref()
        .and_then(|p| p.materials.get(Surface::Accent as usize))
        .and_then(|r| r.coating.as_ref())
        .and_then(|c| c.wall.as_ref())
        .is_none()
    {
        return Vec::new();
    }
    let mut rng = stream(scene.material_seed(), 0x57414c4c504c414e);
    if rng.random_bool(0.12) {
        return Vec::new();
    }
    let mut candidates: Vec<_> = envelope
        .walls
        .iter()
        .filter_map(|wall| {
            let p = envelope.footprint[wall.edge];
            let q = envelope.footprint[(wall.edge + 1) % envelope.footprint.len()];
            let area = p.distance(q)
                * (envelope.ceiling_height(scene.room_size, p)
                    + envelope.ceiling_height(scene.room_size, q))
                * 0.5;
            let openings = wall.facade.as_ref().map_or(0., |f| {
                f.openings
                    .iter()
                    .map(|o| {
                        let s = o.max - o.min;
                        s.x * s.y
                    })
                    .sum::<f32>()
            });
            let solid = (area - openings).max(0.);
            (solid > 2. && solid > area * 0.20)
                .then(|| (wall.edge, solid * rng.random_range(0.25_f32..1.)))
        })
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let count = if rng.random_bool(0.32) { 2 } else { 1 };
    candidates.into_iter().take(count).map(|c| c.0).collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WallFinish {
    /// Stripes, diamond lattice, staggered oval, ogee, and leaf silhouettes.
    #[serde(default)]
    pub print_style: u32,
    /// Limewash, longitudinal fibers, cross weave and printed bands.
    pub weights: [f32; 4],
    pub fiber_cells: [u32; 2],
    pub bands: u32,
    pub band_duty: f32,
    pub warp: f32,
    pub pigment: f32,
    pub secondary_tint: [f32; 3],
    pub relief_m: f32,
    pub roughness_variation: f32,
    pub burnish: f32,
}

pub(super) struct Terms {
    pub gain: [f32; 3],
    pub height: f32,
    pub roughness: f32,
}
impl Default for Terms {
    fn default() -> Self {
        Self {
            gain: [1.; 3],
            height: 0.,
            roughness: 0.,
        }
    }
}

impl WallFinish {
    pub fn sample(seed: u64, surface: Surface) -> Option<Self> {
        if !matches!(surface, Surface::Paint | Surface::Accent) {
            return None;
        }
        let mut rng = stream(seed, 0x57414c4c46494e);
        if !rng.random_bool(if surface == Surface::Accent {
            0.82
        } else {
            0.52
        }) {
            return None;
        }
        // Paint is also used by trim, housings and exterior joinery. Keep that
        // shared role a washed/brushed coating; wall-only Accent carries fibers
        // and printed coverings rather than wallpapering equipment.
        let dominant = if surface == Surface::Paint {
            0
        } else {
            rng.random_range(0..4)
        };
        let mut weights = [0.; 4];
        weights[dominant] = rng.random_range(0.65..1.);
        // Fibrous coverings may be cross woven or printed. Mineral wash may
        // carry subtle brush streaks; never give every wall every effect.
        let secondary = if dominant == 0 {
            1
        } else {
            rng.random_range(1..4)
        };
        if secondary != dominant {
            weights[secondary] = rng.random_range(0.0..0.22);
        }
        let sum = weights.iter().sum::<f32>().max(1.);
        weights = weights.map(|v| v / sum);
        let tint = rng.random_range(0.40_f32..0.88);
        let warmth = rng.random_range(-0.18..0.20);
        Some(Self {
            print_style: stream(seed, 0x57414c4c4d4f5449).random_range(0..5),
            weights,
            fiber_cells: [rng.random_range(28..=80), rng.random_range(24..=72)],
            bands: rng.random_range(2..=9),
            band_duty: rng.random_range(0.08..0.45),
            warp: rng.random_range(0.0005..0.006),
            pigment: rng.random_range(0.10..0.65),
            secondary_tint: [tint, tint * (1. - warmth), tint * (1. - warmth * 1.4)]
                .map(|c| c.min(0.98)),
            relief_m: rng.random_range(0.00004..0.00035),
            roughness_variation: rng.random_range(0.025..0.18),
            burnish: rng.random_range(0.0..0.22) * weights[0],
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self
            .weights
            .iter()
            .chain(self.secondary_tint.iter())
            .chain([
                &self.band_duty,
                &self.warp,
                &self.pigment,
                &self.roughness_variation,
                &self.burnish,
            ])
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || self.weights.iter().sum::<f32>() > 1.001
            || self.fiber_cells.iter().any(|n| !(4..=96).contains(n))
            || self.print_style > 4
            || !(1..=16).contains(&self.bands)
            || !self.relief_m.is_finite()
            || !(0.0..=0.001).contains(&self.relief_m)
        {
            return Err("invalid wall covering program".into());
        }
        Ok(())
    }

    pub(super) fn evaluate(&self, r: &MaterialRecipe, uv: [f32; 2]) -> Terms {
        let n = |uv: [f32; 2], x, y, salt| {
            periodic_noise(uv[0], uv[1], x, y, r.seed.wrapping_add(salt))
        };
        let cloud = n(uv, 3, 5, 701);
        let warp = (cloud - 0.5) * self.warp;
        let p = [uv[0] + warp, uv[1] - warp * 0.6];
        let long = n(p, self.fiber_cells[0], 3, 709);
        let cross = n(p, 5, self.fiber_cells[1], 719);
        let wash = cloud * 0.70 + n(p, 9, 11, 727) * 0.30;
        let at = (uv[0] * self.bands as f32 + warp).rem_euclid(1.);
        let aa = self.bands as f32 / r.map_size(0) as f32;
        let y = (uv[1] * self.bands as f32 + warp * 0.5).rem_euclid(1.);
        let x = at - 0.5;
        let y = y - 0.5;
        let distance = match self.print_style {
            1 => ((x.abs() + y.abs()) - 0.5).abs(),
            2 => (Vec2::new(x * 0.85, y * 1.3).length() - 0.25).abs(),
            3 => (x.abs() - (0.13 + 0.55 * (0.5 - y.abs()).powi(2))).abs(),
            4 => {
                let u = x + y * 0.38;
                (u * u * 5. + y * y * 1.7).sqrt()
            }
            _ => x.abs(),
        };
        let stripe = super::field::smooth((self.band_duty * 0.5 + aa - distance) / (2. * aa));
        let [wash_w, fiber_w, weave_w, print_w] = self.weights;
        let woven = (long * 0.65 + cross * 0.35) * (0.85 + long * cross * 0.30);
        let pigment = ((wash - 0.5) * wash_w + (long - 0.5) * fiber_w + (woven - 0.5) * weave_w)
            * self.pigment
            * r.detail_scale.unwrap_or(1.);
        let ink = stripe * print_w * self.pigment * r.detail_scale.unwrap_or(1.);
        Terms {
            gain: self
                .secondary_tint
                .map(|t| (1. + pigment) * (1. + (t - 1.) * ink)),
            height: ((wash - 0.5) * wash_w * 0.25
                + (long - 0.5) * fiber_w
                + (woven - 0.5) * weave_w)
                * self.relief_m
                * r.detail_scale.unwrap_or(1.),
            roughness: ((long - 0.5) * fiber_w + (woven - 0.5) * weave_w + (wash - 0.5) * wash_w)
                * self.roughness_variation
                - wash * self.burnish
                + ink * 0.025,
        }
    }
}

#[cfg(test)]
mod tests;
