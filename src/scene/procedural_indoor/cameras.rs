//! The manifest, runtime, camera heatmaps and collision validation share one path.
use super::layout::{IndoorCamera, IndoorManifest};
use crate::camera::{
    ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, TrajectorySampler,
};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraMotion {
    pub control: [Vec3; 2],
    pub target_end: Vec3,
    /// Optical-axis roll in radians; small handheld deviations, not scene rotation.
    pub roll: [f32; 2],
}
impl IndoorCamera {
    pub fn runtime_trajectory(&self) -> TrajectorySampler {
        if let Some(m) = &self.motion {
            let orientation = |position, target, roll| {
                Transform::from_translation(position)
                    .looking_at(target, Vec3::Y)
                    .rotation
                    * Quat::from_rotation_z(roll)
            };
            TrajectorySampler::CubicBezier {
                positions: [self.start, m.control[0], m.control[1], self.end],
                rotations: [
                    orientation(self.start, self.target, m.roll[0]),
                    orientation(self.end, m.target_end, m.roll[1]),
                ],
            }
        } else {
            let pose = |p| ExtrinsicsSampler {
                position: ExtrinsicsSamplerType::Transform(Transform::from_translation(p)),
                looking_at: LookingAtSampler::Exact(self.target),
                ..default()
            };
            TrajectorySampler::Linear {
                start: pose(self.start),
                end: pose(self.end),
            }
        }
    }
    pub fn transform_at(&self, progress: f32) -> Transform {
        self.runtime_trajectory().sample(progress)
    }
    pub fn path_length(&self) -> f32 {
        (0..32)
            .map(|i| {
                self.transform_at(i as f32 / 32.0)
                    .translation
                    .distance(self.transform_at((i + 1) as f32 / 32.0).translation)
            })
            .sum()
    }
}
impl IndoorManifest {
    pub fn camera_curve_clear(&self, camera: &IndoorCamera) -> bool {
        use super::layout::{segment_hits_box, CAMERA_CLEARANCE};
        let mut runtime = camera.runtime_trajectory();
        let (control, error) = if let TrajectorySampler::CubicBezier { positions, .. } = &runtime {
            // The linear interpolation error on a subinterval is <= max|P''| h²/8.
            // Inflate each swept chord by that bound, not just sampled positions.
            let second = 6.0
                * (positions[0] - 2.0 * positions[1] + positions[2])
                    .length()
                    .max((positions[1] - 2.0 * positions[2] + positions[3]).length());
            (*positions, second / (8.0 * 64.0 * 64.0))
        } else {
            ([camera.start, camera.start, camera.end, camera.end], 0.0)
        };
        let ceiling = self.room_size.y
            - self
                .program
                .as_ref()
                .map_or([0.10, 0.42, 0.06][self.lighting_design as usize % 3], |p| {
                    p.light_drop
                })
            - 0.04
            - CAMERA_CLEARANCE;
        // A Bezier stays inside the convex hull of its control points.
        if control.iter().any(|p| {
            !p.is_finite()
                || p.x.abs() > self.room_size.x * 0.5 - 0.50
                || p.z.abs() > self.room_size.z * 0.5 - 0.50
                || p.y < 0.70
                || p.y > ceiling
        }) {
            return false;
        }
        let obstacles: Vec<_> = self
            .camera_obstacles()
            .into_iter()
            .chain(self.objects.iter().map(super::layout::IndoorObject::bounds))
            .chain(self.humans.iter().map(super::humans::IndoorHuman::bounds))
            .collect();
        let pad = Vec3::splat(CAMERA_CLEARANCE + error);
        let mut previous = runtime.sample(0.0).translation;
        for i in 0..=64 {
            let pose = runtime.sample(i as f32 / 64.0);
            if !pose.rotation.is_finite()
                || (pose.rotation * Vec3::X).y.abs() > 0.14
                || pose.translation.distance(camera.target) < 1.5
                || obstacles.iter().any(|(lo, hi)| {
                    segment_hits_box(previous, pose.translation, *lo - pad, *hi + pad)
                        || segment_hits_box(
                            pose.translation,
                            pose.translation + pose.rotation * Vec3::NEG_Z,
                            *lo - Vec3::splat(0.01),
                            *hi + Vec3::splat(0.01),
                        )
                })
            {
                return false;
            }
            previous = pose.translation;
        }
        true
    }
}
