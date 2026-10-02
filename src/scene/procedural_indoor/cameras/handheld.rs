//! Short metric motion with independently sampled orientation, without a moving
//! look-at constraint. All proposals pass the usual swept collision checks.
use super::{CameraMotion, IndoorCamera};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HandheldSettings {
    /// Inclusive ranges in metres: local right, up, forward. Negative forward is backpedaling.
    pub translation_m: [[f32; 2]; 3],
    /// Inclusive endpoint increments: yaw, pitch, roll, in degrees.
    pub rotation_degrees: [[f32; 2]; 3],
    /// Reverse the entire sampled path and its orientations together.
    pub reverse_probability: f32,
}
impl Default for HandheldSettings {
    fn default() -> Self {
        Self {
            translation_m: [[-0.12, 0.12], [-0.035, 0.035], [-0.35, 0.35]],
            rotation_degrees: [[-8.0, 8.0], [-5.0, 5.0], [-3.0, 3.0]],
            reverse_probability: 0.5,
        }
    }
}
impl HandheldSettings {
    pub fn validate(&self) -> Result<(), String> {
        for (ranges, bound) in [(&self.translation_m, 10.0), (&self.rotation_degrees, 90.0)] {
            if ranges.iter().any(|[lo, hi]| {
                !lo.is_finite() || !hi.is_finite() || lo > hi || lo.abs().max(hi.abs()) > bound
            }) {
                return Err("handheld ranges must be ordered and finite; translation <= 10m, angles <= 90 degrees".into());
            }
        }
        if !(0.0..=1.0).contains(&self.reverse_probability) {
            return Err("handheld reverse_probability must be in [0,1]".into());
        }
        Ok(())
    }
    pub(super) fn apply(&self, camera: &mut IndoorCamera, rng: &mut impl Rng) {
        let start = Transform::from_translation(camera.start).looking_at(camera.target, Vec3::Y);
        let delta = self.translation_m.map(|[lo, hi]| rng.random_range(lo..=hi));
        let angles = self
            .rotation_degrees
            .map(|[lo, hi]| rng.random_range(lo..=hi).to_radians());
        let mut rotations = [
            start.rotation,
            (start.rotation * Quat::from_euler(EulerRot::YXZ, angles[0], angles[1], angles[2]))
                .normalize(),
        ];
        camera.end = camera.start + start.rotation * Vec3::new(delta[0], delta[1], -delta[2]);
        if rng.random_bool(self.reverse_probability as f64) {
            std::mem::swap(&mut camera.start, &mut camera.end);
            rotations.swap(0, 1);
        }
        camera.motion = Some(CameraMotion {
            orientations: Some(rotations),
            control: [
                camera.start.lerp(camera.end, 1.0 / 3.0),
                camera.start.lerp(camera.end, 2.0 / 3.0),
            ],
            target_end: camera.target,
            roll: [0.0; 2],
            route: Vec::new(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::stream;
    #[test]
    fn sampled_short_paths_keep_clearance_and_duration_labels() {
        use crate::scene::procedural_indoor::{
            cameras::CameraSettings,
            layout::{IndoorLayout, IndoorManifest},
            validation::validate_layout,
        };
        for seed in 0..8 {
            for independent in [true, false] {
                let mut scene =
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                        .unwrap();
                let mut settings = CameraSettings {
                    handheld: Some(HandheldSettings::default()),
                    duration_seconds: Some(1.2),
                    path_length_min: 0.01,
                    path_length_max: 0.5,
                    long_path_fraction: 0.,
                    ..Default::default()
                };
                if independent {
                    settings.multiview = None;
                }
                scene.resample_cameras(2, settings, 1.6).unwrap();
                validate_layout(&scene).unwrap();
                assert!(scene.cameras.iter().all(|c| c
                    .motion
                    .as_ref()
                    .unwrap()
                    .orientations
                    .is_some()));
            }
        }
    }
    #[test]
    fn metric_stride_free_rotation_and_reversal_are_exact() {
        let mut camera = IndoorCamera {
            start: Vec3::ZERO,
            end: Vec3::ZERO,
            target: Vec3::NEG_Z,
            fov_degrees: 60.0,
            motion: None,
        };
        let mut settings = HandheldSettings {
            translation_m: [[0., 0.], [0., 0.], [0.3, 0.3]],
            rotation_degrees: [[12., 12.], [4., 4.], [7., 7.]],
            reverse_probability: 0.0,
        };
        let mut reversed = camera.clone();
        settings.apply(&mut camera, &mut stream(4, 3));
        assert!((camera.end - Vec3::new(0., 0., -0.3)).length() < 1e-6);
        assert!((camera.path_length() - 0.3).abs() < 1e-6);
        assert!(
            camera
                .transform_at(1.)
                .rotation
                .angle_between(camera.transform_at(0.).rotation)
                > 0.1
        );
        settings.reverse_probability = 1.0;
        settings.apply(&mut reversed, &mut stream(4, 3));
        for i in 0..=32 {
            let a = camera.transform_at(i as f32 / 32.);
            let b = reversed.transform_at(1. - i as f32 / 32.);
            assert!(a.translation.distance(b.translation) < 1e-6);
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                assert!((a.rotation * axis).distance(b.rotation * axis) < 1e-5);
            }
        }
    }
}
