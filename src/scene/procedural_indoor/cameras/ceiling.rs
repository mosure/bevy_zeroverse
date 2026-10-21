//! Fixed ceiling-mounted overview cameras. The mounting geometry is separate
//! from the calibrated lens, and all rig visibility constraints remain active.
mod placement;
mod planner;
#[cfg(test)]
mod tests;
use super::{intrinsics::IntrinsicsSettings, CameraSettings, IndoorCamera};
use crate::camera::lens::LensSampling;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Opt-in policy for a fixed ceiling-mounted capture rig. Applies to every
/// capture view; editor cameras and object/human animation are unaffected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CeilingCameraSettings {
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
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        let (a, b) = self.bounds();
        let (c, d) = other.bounds();
        a.cmplt(d).all() && b.cmpgt(c).all()
    }
}

impl CameraSettings {
    pub(crate) fn lens_for(
        &self,
        seed: u64,
        index: usize,
    ) -> Option<crate::calibration::FisheyeSensor> {
        self.ceiling_mount
            .as_ref()
            .map(|p| &p.lenses)
            .or(self.lenses.as_ref())
            .and_then(|p| p.sample(seed, index))
    }
    pub(crate) fn active_intrinsics(&self) -> &IntrinsicsSettings {
        self.ceiling_mount
            .as_ref()
            .map_or(&self.intrinsics, |p| &p.intrinsics)
    }
}

impl IndoorCamera {
    pub fn is_ceiling_mounted(&self) -> bool {
        self.ceiling_mount.is_some()
    }
}

pub(super) use placement::clear as mount_clear;
pub(super) use planner::sample;
