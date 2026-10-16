//! Camera motion roles are seeded once per view, independently of placement retries.
use super::{CameraMotion, CameraSettings, IndoorCamera};
use bevy::prelude::*;
use rand::Rng;

impl CameraSettings {
    /// Seeded Bernoulli assignment; stable when adding views or retrying placement.
    /// The fraction is a dataset probability, not a rounded count in a small rig.
    pub fn camera_moves(&self, seed: u64, index: usize) -> bool {
        if self.motion_fraction == 0. {
            return false;
        }
        if self.motion_fraction == 1. {
            return true;
        }
        // Domain-separated SplitMix64; no consumption of the placement RNG.
        let mut x =
            seed ^ 0x4341_4d45_5241_4d56 ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        ((x >> 11) as f64 / (1_u64 << 53) as f64) < self.motion_fraction as f64
    }

    pub(super) fn minimum_travel(&self, seed: u64, index: usize) -> f32 {
        if self.camera_moves(seed, index) {
            self.path_length_min
        } else {
            0.
        }
    }

    /// Apply before *all* clearance, synchronized spacing and visibility checks.
    /// In particular a moving peer must not inherit a fixed parent's zero path.
    pub(super) fn prepare_motion(
        &self,
        seed: u64,
        index: usize,
        camera: &mut IndoorCamera,
        rng: &mut impl Rng,
    ) {
        if !self.camera_moves(seed, index) {
            camera.end = camera.start;
            camera.motion = None;
        } else if let Some(handheld) = &self.handheld {
            handheld.apply(camera, rng);
        } else if camera.start == camera.end && camera.motion.is_none() && self.path_length_max > 0.
        {
            let min = self.path_length_min.max(0.001).min(self.path_length_max);
            let distance = rng.random_range(min.ln()..=self.path_length_max.ln()).exp();
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            camera.end = camera.start + Vec3::new(angle.cos(), 0., angle.sin()) * distance;
            camera.motion = Some(CameraMotion {
                orientations: None,
                control: [
                    camera.start.lerp(camera.end, 1. / 3.),
                    camera.start.lerp(camera.end, 2. / 3.),
                ],
                target_end: camera.target,
                roll: [0.; 2],
                route: Vec::new(),
            });
        }
    }

    pub(crate) fn accepts_motion(&self, seed: u64, index: usize, camera: &IndoorCamera) -> bool {
        if !self.camera_moves(seed, index) {
            return camera.start == camera.end && camera.motion.is_none();
        }
        let length = camera.path_length();
        length + 1e-4 >= self.path_length_min && length <= self.path_length_max + 1e-4
    }
}

#[cfg(test)]
mod tests;
