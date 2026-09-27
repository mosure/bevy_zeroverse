//! The manifest, runtime, camera heatmaps and collision validation share one path.
pub(crate) mod coverage;
mod navigation;
mod sampling;
use super::layout::{IndoorCamera, IndoorManifest};
use crate::camera::{
    ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, TrajectorySampler,
};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraSettings {
    /// Keep every capture in the largest furnished room, including its full path.
    pub primary_room: bool,
    pub path_length_min: f32,
    pub path_length_max: f32,
    pub long_path_fraction: f32,
}
impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            primary_room: true,
            path_length_min: 0.03,
            path_length_max: 8.0,
            long_path_fraction: 0.45,
        }
    }
}
impl CameraSettings {
    pub fn parse(json: &str) -> Result<Self, String> {
        let settings: Self =
            serde_json::from_str(json).map_err(|e| format!("indoor_camera: {e}"))?;
        settings.validate()?;
        Ok(settings)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.path_length_min.is_finite()
            || !self.path_length_max.is_finite()
            || self.path_length_min < 0.0
            || self.path_length_max < self.path_length_min
            || self.path_length_max > 100.0
            || !(0.0..=1.0).contains(&self.long_path_fraction)
        {
            return Err("indoor_camera requires 0 <= path_length_min <= path_length_max <= 100 metres and long_path_fraction in [0,1]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraMotion {
    pub control: [Vec3; 2],
    pub target_end: Vec3,
    /// Optical-axis roll in radians; small handheld deviations, not scene rotation.
    pub roll: [f32; 2],
    /// Collision-checked, arc-length-parametrized route including both endpoints.
    /// Empty retains the local cubic trajectory used by older manifests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub route: Vec<Vec3>,
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
            if !m.route.is_empty() {
                return TrajectorySampler::WaypointPath {
                    positions: m.route.clone(),
                    rotations: m
                        .route
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            let t = i as f32 / (m.route.len() - 1).max(1) as f32;
                            orientation(
                                *p,
                                self.target.lerp(m.target_end, t),
                                m.roll[0] + t * (m.roll[1] - m.roll[0]),
                            )
                        })
                        .collect(),
                };
            }
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
            if self.start == self.end {
                TrajectorySampler::Static {
                    start: pose(self.start),
                }
            } else {
                TrajectorySampler::Linear {
                    start: pose(self.start),
                    end: pose(self.end),
                }
            }
        }
    }
    pub fn transform_at(&self, progress: f32) -> Transform {
        self.runtime_trajectory().sample(progress)
    }
    pub fn path_length(&self) -> f32 {
        if let Some(m) = &self.motion {
            if !m.route.is_empty() {
                return m.route.windows(2).map(|p| p[0].distance(p[1])).sum();
            }
        }
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
    pub fn primary_room_bounds(&self) -> (Vec2, Vec2) {
        self.program
            .as_ref()
            .and_then(|p| {
                p.zones.iter().max_by(|a, b| {
                    ((a.max - a.min).element_product())
                        .total_cmp(&(b.max - b.min).element_product())
                })
            })
            .map(|z| (z.min, z.max))
            .unwrap_or_else(|| {
                let half = Vec2::new(self.room_size.x, self.room_size.z) * 0.5;
                (-half, half)
            })
    }
    pub fn in_primary_room(&self, p: Vec3, clearance: f32) -> bool {
        let (lo, hi) = self.primary_room_bounds();
        p.xz().cmpge(lo + Vec2::splat(clearance)).all()
            && p.xz().cmple(hi - Vec2::splat(clearance)).all()
    }
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
            (positions.to_vec(), second / (8.0 * 64.0 * 64.0))
        } else if let TrajectorySampler::WaypointPath { positions, .. } = &runtime {
            if positions.len() < 2
                || positions
                    .windows(2)
                    .any(|p| !self.camera_path_clear(p[0], p[1]))
            {
                return false;
            }
            (positions.clone(), 0.0)
        } else {
            (vec![camera.start, camera.end], 0.0)
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
                || (self.camera_settings.primary_room
                    && !self.in_primary_room(*p, CAMERA_CLEARANCE))
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

#[cfg(test)]
mod tests {
    use super::super::layout::{IndoorLayout, ObjectKind};
    use super::*;

    #[test]
    fn primary_room_paths_and_long_routes_respect_furniture() {
        let mut routed = 0;
        for seed in 0..48 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 2, 0.25)
                    .unwrap();
            for c in &scene.cameras {
                assert!(scene.camera_curve_clear(c));
                for step in 0..=64 {
                    assert!(
                        scene.in_primary_room(c.transform_at(step as f32 / 64.0).translation, 0.0)
                    );
                }
                if c.path_length() > 2.0 {
                    routed += 1;
                }
            }
        }
        assert!(
            routed >= 8,
            "long trajectories collapsed into static viewpoints: {routed}"
        );
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.5, 0, 0.0).unwrap();
        let mut obstacle = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Table)
            .unwrap()
            .clone();
        scene.room_size = Vec3::new(10.0, 3.5, 10.0);
        scene.program = None;
        scene.floor_plan = super::super::floorplan::FloorPlan::OpenHall;
        obstacle.position = Vec3::ZERO;
        obstacle.yaw = 0.0;
        obstacle.size = Vec3::new(2.0, 2.0, 3.0);
        scene.objects = vec![obstacle];
        let a = Vec3::new(-3.0, 1.5, 0.0);
        let b = Vec3::new(3.0, 1.5, 0.0);
        assert!(!scene.camera_path_clear(a, b));
        let route = super::navigation::route(&scene, a, b).expect("route around central obstacle");
        assert!(route.len() > 2);
        assert!(route
            .windows(2)
            .all(|p| scene.camera_path_clear(p[0], p[1])));
    }

    #[test]
    fn zero_length_capture_paths_freeze_orientation_as_well_as_position() {
        for seed in 0..8 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 0, 0.5)
                    .unwrap();
            scene.camera_settings = CameraSettings {
                path_length_min: 0.0,
                path_length_max: 0.0,
                ..default()
            };
            scene.sample_cameras(2).unwrap();
            for camera in &scene.cameras {
                assert_eq!(camera.path_length(), 0.0);
                assert_eq!(camera.transform_at(0.0), camera.transform_at(0.5));
                assert_eq!(camera.transform_at(0.0), camera.transform_at(1.0));
            }
        }
    }

    #[test]
    fn camera_policy_rejects_bad_ranges_and_round_trips_routes() {
        for json in [
            r#"{"path_length_min":-1}"#,
            r#"{"path_length_min":9,"path_length_max":2}"#,
            r#"{"long_path_fraction":1.1}"#,
        ] {
            assert!(CameraSettings::parse(json).is_err());
        }
        let scene =
            IndoorManifest::generate_with_humans(3, IndoorLayout::Mixed, 0.6, 2, 0.0).unwrap();
        let restored: IndoorManifest =
            serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
        for (a, b) in scene.cameras.iter().zip(restored.cameras) {
            for step in 0..=20 {
                assert_eq!(
                    a.transform_at(step as f32 / 20.0),
                    b.transform_at(step as f32 / 20.0)
                );
            }
        }
    }
}
