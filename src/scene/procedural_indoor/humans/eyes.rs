//! Binocular fixation in Anny rest space. Both eyeballs retain their original
//! vertices, normals and skinning. Only their optical texture coordinates change.
use super::{garments::GarmentVertex, stream, HumanPoseKind};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeOptics {
    /// Nominal radius on the unscaled Anny head.
    pub iris_radius_m: f32,
    /// Pupil radius as a fraction of iris radius around 200 lux.
    pub pupil_fraction: f32,
    pub tear_roughness: f32,
}

impl EyeOptics {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"EyeMedia"));
        Self {
            iris_radius_m: rng.random_range(0.0050..0.0060),
            pupil_fraction: rng.random_range(0.29..0.44),
            tear_roughness: rng.random_range(0.13..0.22),
        }
    }

    /// Room-level adaptation from incident illuminance. Camera exposure and
    /// camera motion do not change the underlying actor.
    pub(crate) fn adapted_pupil(self, lux: f32) -> f32 {
        (self.pupil_fraction - 0.042 * (lux.max(1.) / 200.).ln()).clamp(0.20, 0.62)
    }
}

/// Continuous mixtures of cool stromal colors, amber and brown pigment in sRGB.
pub(super) fn pigment(density: f32, stroma: f32, value: f32) -> [f32; 3] {
    let mineral = Vec3::new(0.27, 0.37, 0.46).lerp(Vec3::new(0.38, 0.42, 0.30), stroma);
    let amber = Vec3::new(0.49, 0.345, 0.195);
    let brown = Vec3::new(0.18, 0.103, 0.062);
    let color = mineral
        .lerp(amber, smoothstep(0.17, 0.56, density))
        .lerp(brown, smoothstep(0.40, 1.0, density));
    (color * (0.78 + 0.37 * value)).to_array()
}

pub(crate) fn smoothstep(lo: f32, hi: f32, v: f32) -> f32 {
    let t = ((v - lo) / (hi - lo)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

/// Half the square texture width in iris radii. The outer area is sclera.
pub(crate) const CHART_EXTENT: f32 = 1.65;

/// A shared metric fixation point relative to the midpoint of the two eyes.
fn fixation(seed: u64, pose: HumanPoseKind) -> Vec3 {
    let mut rng = stream(seed, u64::from_le_bytes(*b"Fixation"));
    let near = matches!(
        pose,
        HumanPoseKind::StandingReading
            | HumanPoseKind::StandingPreparing
            | HumanPoseKind::SeatedWorking
    );
    let yaw: f32 = (rng.random::<f32>() - rng.random::<f32>()) * if near { 0.18 } else { 0.32 };
    let pitch: f32 = if near {
        rng.random_range(-0.23..0.03)
    } else {
        (rng.random::<f32>() - rng.random::<f32>()) * 0.19
    };
    let distance = if near {
        rng.random_range(0.40..0.94)
    } else {
        rng.random_range(1.1..5.5)
    };
    Vec3::new(yaw.tan(), pitch.tan(), -1.).normalize() * distance
}

pub(super) struct EyeCharts {
    bones: [usize; 2],
    charts: [EyeChart; 2],
}

pub(super) struct EyeChart {
    origin: Vec3,
    forward: Vec3,
    u: Vec3,
    v: Vec3,
}

impl EyeCharts {
    pub(super) fn new(
        optics: EyeOptics,
        seed: u64,
        pose: HumanPoseKind,
        bones: [usize; 2],
        origins: [Vec3; 2],
        scale: f32,
    ) -> Self {
        // The rig's rest coordinates predate the per-person stature scale.
        let target = (origins[0] + origins[1]) * 0.5 + fixation(seed, pose) / scale;
        let units = (2. * CHART_EXTENT * optics.iris_radius_m.clamp(0.004, 0.007)).recip();
        let charts = std::array::from_fn(|i| {
            let forward = (target - origins[i]).normalize();
            let right = forward.cross(Vec3::Y).normalize();
            let up = right.cross(forward);
            // Mirrored texture orientation between the eyes preserves their common
            // iris pigment and pupil size without cloning the pattern in image space.
            EyeChart {
                origin: origins[i],
                forward,
                u: right * if i == 0 { units } else { -units },
                v: -up * units,
            }
        });
        Self { bones, charts }
    }

    pub(super) fn for_bone(&self, bone: usize) -> Option<&EyeChart> {
        if bone == self.bones[0] {
            Some(&self.charts[0])
        } else if bone == self.bones[1] {
            Some(&self.charts[1])
        } else {
            None
        }
    }
}

impl EyeChart {
    pub(super) fn map_face(&self, corners: &mut [GarmentVertex; 4]) {
        let center = corners.iter().map(|p| p.rest).sum::<Vec3>() * 0.25;
        let anterior = (center - self.origin).dot(self.forward) > 0.;
        for corner in corners {
            let p = corner.rest - self.origin;
            corner.uv = if anterior {
                // The image clamps at its outer sclera. Keep UVs affine across
                // front triangles to avoid bending the iris on coarse eyeballs.
                Vec2::splat(0.5) + Vec2::new(p.dot(self.u), p.dot(self.v))
            } else {
                Vec2::ZERO
            };
        }
    }
}

#[cfg(test)]
mod tests;
