//! Ceiling-mounted overview cameras. The mounting geometry is separate
//! from the calibrated lens, and all rig visibility constraints remain active.
mod overview;
mod placement;
mod planner;
mod room;
#[cfg(test)]
mod tests;
mod trajectory;
use super::{intrinsics::IntrinsicsSettings, CameraSettings, IndoorCamera};
use crate::camera::lens::LensSampling;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Independent per-view ceiling prior. Unselected views use the room camera
/// policy. Defaults preserve the original all-mounted, fixed rig.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CeilingCameraSettings {
    /// Seeded probability of assigning a capture slot to the ceiling.
    pub probability: f32,
    /// Fraction of mounted cameras traveling along a ceiling rail.
    pub motion_fraction: f32,
    /// Metric travel bounds, independent of room/handheld trajectories.
    pub travel_m: [f32; 2],
    /// Optical center distance along the inward ceiling normal, in metres.
    pub standoff_m: [f32; 2],
    /// Conservative mounting-body radius, also used for fixture clearance.
    pub body_radius_m: f32,
    pub wall_clearance_m: f32,
    /// Mounted-camera lens prior, independent of the walkthrough lens prior.
    pub lenses: LensSampling,
    /// Used when `lenses.probability < 1`; vertical rectilinear FOV.
    pub intrinsics: IntrinsicsSettings,
}
impl Default for CeilingCameraSettings {
    fn default() -> Self {
        Self {
            probability: 1.,
            motion_fraction: 0.,
            travel_m: [0.1, 3.],
            standoff_m: [0.12, 0.24],
            body_radius_m: 0.055,
            wall_clearance_m: 0.35,
            lenses: LensSampling {
                probability: 1.,
                fov_degrees: [170., 200.],
                circular_probability: 0.75,
                anisotropy: [1., 1.],
                ..default()
            },
            intrinsics: IntrinsicsSettings {
                fov_y_min_degrees: 100.,
                fov_y_max_degrees: 135.,
                bias: 0.,
                ..default()
            },
        }
    }
}
impl CeilingCameraSettings {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.).contains(&self.probability)
            || !(0.0..=1.).contains(&self.motion_fraction)
            || self
                .travel_m
                .iter()
                .any(|v| !v.is_finite() || *v <= 0. || *v > 100.)
            || self.travel_m[0] > self.travel_m[1]
        {
            return Err("ceiling_mount requires probability and motion_fraction in [0,1], and ordered travel_m in (0,100] metres".into());
        }
        self.lenses.validate()?;
        self.intrinsics.validate()?;
        if !(0.02..=0.20).contains(&self.body_radius_m)
            || !(0.20..=2.).contains(&self.wall_clearance_m)
            || self.standoff_m.iter().any(|v| !(0.08..=1.).contains(v))
            || self.standoff_m[0] > self.standoff_m[1]
            || self.standoff_m[0] < 2. * self.body_radius_m + 0.005
        {
            return Err("ceiling_mount requires ordered standoff_m in [0.08,1], body_radius_m in [0.02,0.20], standoff >= 2 * radius + 0.005, and wall_clearance_m in [0.20,2]".into());
        }
        Ok(())
    }
}

/// Realized mounting metadata, in room-local metres. The runtime and export
/// both use this exact orientation, avoiding a look-at pole at straight down.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CeilingCameraMount {
    pub surface: Vec3,
    pub inward_normal: Vec3,
    pub rotation: Quat,
    pub standoff_m: f32,
    pub body_radius_m: f32,
}
impl CeilingCameraMount {
    /// Conservative body and attachment envelope for temporal collision checks.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let base = self.surface + self.inward_normal * (self.body_radius_m + 0.006);
        let front = self.surface + self.inward_normal * (self.standoff_m + self.body_radius_m);
        let pad = Vec3::splat(self.body_radius_m);
        (base.min(front) - pad, base.max(front) + pad)
    }
}

impl CameraSettings {
    /// Stable per-slot assignment, independent of retries, lenses and rig size.
    pub fn ceiling_mounted(&self, seed: u64, index: usize) -> bool {
        self.ceiling_mount
            .as_ref()
            .is_some_and(|p| super::motion::role(seed, index, 0x4345_494c_494e_4753, p.probability))
    }
    pub(crate) fn lens_for(
        &self,
        seed: u64,
        index: usize,
    ) -> Option<crate::calibration::FisheyeSensor> {
        self.ceiling_mount
            .as_ref()
            .filter(|_| self.ceiling_mounted(seed, index))
            .map(|p| &p.lenses)
            .or(self.lenses.as_ref())
            .and_then(|p| p.sample(seed, index))
    }
    pub(crate) fn intrinsics_for(&self, seed: u64, index: usize) -> &IntrinsicsSettings {
        self.ceiling_mount
            .as_ref()
            .filter(|_| self.ceiling_mounted(seed, index))
            .map_or(&self.intrinsics, |p| &p.intrinsics)
    }
}

impl IndoorCamera {
    pub fn is_ceiling_mounted(&self) -> bool {
        self.ceiling_mount.is_some()
    }

    /// Conservative envelope of the attachment and body throughout its rail.
    /// Room cameras have no mounting body. Useful to temporal scene planners.
    pub fn ceiling_sweep_bounds(&self) -> Option<(Vec3, Vec3)> {
        self.ceiling_mount.as_ref().map(|m| {
            let (lo, hi) = m.bounds();
            let delta = self.end - self.start;
            (lo.min(lo + delta), hi.max(hi + delta))
        })
    }
}

pub(super) use placement::clear as mount_clear;
pub(super) use planner::sample;

pub(crate) use trajectory::bodies_overlap;
