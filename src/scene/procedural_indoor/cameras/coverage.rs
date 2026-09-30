//! Cheap first-surface visibility proposals. Rendered labels remain the oracle.
//! Glazing blocks labels, matching the dataset's annotation policy.
use super::super::envelope::EnvelopeProgram;
use crate::scene::procedural_indoor::layout::{IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use std::collections::BTreeMap;

struct Collider {
    center: Vec3,
    half: Vec3,
    inverse: Quat,
    label: &'static str,
}

pub(crate) struct Coverage {
    colliders: Vec<Collider>,
    body_points: Vec<Vec3>,
    shell: Option<(EnvelopeProgram, Vec3, f32)>,
}

/// Pixel-centred proxy rays over the full calibrated frustum, not just its centre.
pub(super) struct VisibilityView {
    pose: Transform,
    scale: Vec2,
    points: Vec<Option<Vec3>>,
}

impl Coverage {
    fn shell_hit(&self, origin: Vec3, ray: Vec3) -> Option<(f32, &'static str)> {
        self.shell
            .as_ref()
            .and_then(|(e, size, door)| e.ray_hit(*size, *door, origin, ray))
    }
    fn first_hit(&self, origin: Vec3, ray: Vec3) -> Option<f32> {
        self.colliders
            .iter()
            .filter_map(|c| ray_box(c.inverse * (origin - c.center), c.inverse * ray, c.half))
            .chain(self.shell_hit(origin, ray).map(|hit| hit.0))
            .min_by(f32::total_cmp)
    }

    pub(super) fn view(&self, camera: &IndoorCamera, aspect: f32, t: f32) -> VisibilityView {
        let pose = camera.transform_at(t);
        let scale = Vec2::new(aspect, 1.0) * (camera.fov_degrees.to_radians() * 0.5).tan();
        let mut points = Vec::with_capacity(117);
        for y in 0..9 {
            for x in 0..13 {
                let local = Vec3::new(
                    (2.0 * (x as f32 + 0.5) / 13.0 - 1.0) * scale.x,
                    (2.0 * (y as f32 + 0.5) / 9.0 - 1.0) * scale.y,
                    -1.0,
                )
                .normalize();
                let ray = pose.rotation * local;
                points.push(
                    self.first_hit(pose.translation, ray)
                        .filter(|&d| (-local.z * d) >= 0.1 && (-local.z * d) <= 50.0)
                        .map(|d| pose.translation + ray * d),
                );
            }
        }
        VisibilityView {
            pose,
            scale,
            points,
        }
    }

    /// Fraction of source pixels whose first surface is also visible in target,
    /// and the mean triangulation angle of those correspondences (degrees).
    pub(super) fn shared(&self, source: &VisibilityView, target: &VisibilityView) -> (f32, f32) {
        let mut visible = 0;
        let mut angle = 0.0;
        for point in source.points.iter().flatten() {
            let delta = *point - target.pose.translation;
            let local = target.pose.rotation.inverse() * delta;
            if -local.z < 0.1
                || -local.z > 50.0
                || local.x.abs() >= -local.z * target.scale.x
                || local.y.abs() >= -local.z * target.scale.y
            {
                continue;
            }
            let distance = delta.length();
            if self
                .first_hit(target.pose.translation, delta / distance)
                .is_some_and(|hit| (hit - distance).abs() <= 0.002 * distance + 0.005)
            {
                visible += 1;
                angle += (delta
                    .normalize()
                    .dot((*point - source.pose.translation).normalize()))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            }
        }
        (
            visible as f32 / source.points.len() as f32,
            angle / visible.max(1) as f32,
        )
    }

    pub fn new(scene: &IndoorManifest) -> Self {
        let mut colliders = Vec::new();
        let mut bounds = |lo: Vec3, hi: Vec3, label| {
            colliders.push(Collider {
                center: (lo + hi) * 0.5,
                half: (hi - lo) * 0.5,
                inverse: Quat::IDENTITY,
                label,
            })
        };
        for (lo, hi) in scene.camera_obstacles() {
            bounds(lo, hi, "wall");
        }
        if scene.envelope.is_none() {
            let half = scene.room_size * 0.5;
            for axis in [0, 2] {
                for sign in [-1.0, 1.0] {
                    let mut lo = -half;
                    let mut hi = half;
                    lo.y = 0.0;
                    hi.y = scene.room_size.y;
                    lo[axis] = sign * half[axis] - 0.02;
                    hi[axis] = sign * half[axis] + 0.02;
                    bounds(lo, hi, "wall");
                }
            }
            bounds(
                Vec3::new(-half.x, -0.02, -half.z),
                Vec3::new(half.x, 0.0, half.z),
                "floor",
            );
            bounds(
                Vec3::new(-half.x, scene.room_size.y, -half.z),
                Vec3::new(half.x, scene.room_size.y + 0.02, half.z),
                "ceiling",
            );
        }
        for h in scene.humans.iter().filter(|h| !h.neighbor) {
            let (lo, hi) = h.bounds();
            bounds(lo, hi, "person");
        }
        for o in scene.objects.iter().filter(|o| !o.neighbor) {
            if o.kind == super::super::layout::ObjectKind::Chair {
                // A tall chair's full bounding box hides the seated person's
                // face even when the front of the real chair is completely open.
                // Use its seat/back/headrest structure for view proposals.
                let program = super::super::objects::chairs::parameters(o);
                let back_top = o.size.y - if program.headrest { 0.22 } else { 0.02 };
                let mut parts = vec![
                    (Vec3::new(0.0, 0.43, -0.015), Vec3::new(0.27, 0.045, 0.25)),
                    (
                        Vec3::new(0.0, (0.57 + back_top) * 0.5, 0.24),
                        Vec3::new(0.24, (back_top - 0.57) * 0.5, 0.055),
                    ),
                    (Vec3::new(0.0, 0.23, 0.0), Vec3::new(0.055, 0.20, 0.055)),
                ];
                if program.headrest {
                    parts.push((
                        Vec3::new(0.0, o.size.y - 0.10, 0.22),
                        Vec3::new(0.17, 0.075, 0.055),
                    ));
                }
                if program.armrests {
                    for x in [-0.285, 0.285] {
                        parts.push((
                            Vec3::new(x, program.arm_height, 0.0),
                            Vec3::new(0.023, 0.025, 0.125),
                        ));
                    }
                }
                for (center, half) in parts {
                    colliders.push(Collider {
                        center: o.transform().transform_point(center),
                        half,
                        inverse: Quat::from_rotation_y(-o.yaw),
                        label: "chair",
                    });
                }
                continue;
            }
            colliders.push(Collider {
                center: o.position + Vec3::Y * o.size.y * 0.5,
                half: o.size * 0.5,
                inverse: Quat::from_rotation_y(-o.yaw),
                label: o.kind.class_name(),
            });
        }
        let body_points = scene
            .humans
            .iter()
            .filter(|h| !h.neighbor)
            .flat_map(|h| {
                [h.joints[2], h.joints[4], h.joints[4] + Vec3::Y * 0.08]
                    .map(|p| h.transform().transform_point(p))
            })
            .collect();
        Self {
            colliders,
            body_points,
            shell: scene
                .envelope
                .clone()
                .map(|e| (e, scene.room_size, scene.door_x)),
        }
    }

    fn body_visible(&self, pose: Transform, scale: f32) -> bool {
        self.body_points.iter().any(|point| {
            let local = pose.rotation.inverse() * (*point - pose.translation);
            // Test anatomical points, not empty corners of a person's envelope.
            // Otherwise two rays through a doorway can hit the padded envelope
            // while the actual person remains behind annotation-opaque glazing.
            if local.z >= -0.1 || local.x.abs().max(local.y.abs()) > -local.z * scale * 0.8 {
                return false;
            }
            let displacement = *point - pose.translation;
            let distance = displacement.length();
            let ray = displacement / distance;
            !self
                .shell_hit(pose.translation, ray)
                .is_some_and(|hit| hit.0 < distance - 0.06)
                && !self
                    .colliders
                    .iter()
                    .filter(|c| c.label != "person")
                    .any(|c| {
                        ray_box(
                            c.inverse * (pose.translation - c.center),
                            c.inverse * ray,
                            c.half,
                        )
                        .is_some_and(|hit| hit < distance - 0.06)
                    })
        })
    }

    pub fn suitable(&self, camera: &IndoorCamera, require_person: bool) -> bool {
        [0.0, 0.5, 1.0].into_iter().all(|t| {
            let pose = camera.transform_at(t);
            let scale = (camera.fov_degrees.to_radians() * 0.5).tan();
            if require_person && !self.body_visible(pose, scale) {
                return false;
            }
            let mut counts = BTreeMap::new();
            // Central square stays within standard landscape captures, so the
            // policy does not rely on peripheral content just outside the frame.
            for y in 0..5 {
                for x in 0..7 {
                    let ray = pose.rotation
                        * Vec3::new(
                            (x as f32 / 6.0 - 0.5) * 1.6 * scale,
                            (y as f32 / 4.0 - 0.5) * 1.6 * scale,
                            -1.0,
                        )
                        .normalize();
                    let (mut nearest, mut label) = self
                        .shell_hit(pose.translation, ray)
                        .unwrap_or((f32::INFINITY, "background"));
                    for c in &self.colliders {
                        let origin = c.inverse * (pose.translation - c.center);
                        let direction = c.inverse * ray;
                        if let Some(distance) = ray_box(origin, direction, c.half) {
                            if distance < nearest {
                                nearest = distance;
                                label = c.label;
                            }
                        }
                    }
                    *counts.entry(label).or_insert(0usize) += 1;
                }
            }
            counts.values().filter(|&&n| n >= 2).count() >= 4
                && counts.values().copied().max().unwrap_or(35) <= 28
                && (!require_person || counts.get("person").copied().unwrap_or(0) >= 2)
        })
    }
}

fn ray_box(origin: Vec3, direction: Vec3, half: Vec3) -> Option<f32> {
    let mut near: f32 = 0.0;
    let mut far: f32 = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-7 {
            if origin[axis].abs() > half[axis] {
                return None;
            }
        } else {
            let a = (-half[axis] - origin[axis]) / direction[axis];
            let b = (half[axis] - origin[axis]) / direction[axis];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
            if far < near {
                return None;
            }
        }
    }
    (far > 0.0).then_some(near)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_surfaces_require_visibility_and_actual_frustum() {
        let point = Vec3::new(0.0, 0.0, -4.0);
        let a = VisibilityView {
            pose: Transform::IDENTITY,
            scale: Vec2::ONE,
            points: vec![Some(point)],
        };
        let mut b = VisibilityView {
            pose: Transform::from_xyz(2.0, 0.0, 0.0).looking_at(point, Vec3::Y),
            scale: Vec2::ONE,
            points: vec![Some(point)],
        };
        let mut coverage = Coverage {
            colliders: vec![Collider {
                center: point - Vec3::Z * 0.01,
                half: Vec3::new(10.0, 10.0, 0.01),
                inverse: Quat::IDENTITY,
                label: "wall",
            }],
            body_points: Vec::new(),
            shell: None,
        };
        let (fraction, angle) = coverage.shared(&a, &b);
        assert_eq!(fraction, 1.0);
        assert!((angle - 26.56505).abs() < 0.001);
        coverage.colliders.push(Collider {
            center: Vec3::new(1.0, 0.0, -2.0),
            half: Vec3::splat(0.2),
            inverse: Quat::IDENTITY,
            label: "wall",
        });
        assert_eq!(
            coverage.shared(&a, &b).0,
            0.0,
            "a common target behind an occluder is not shared geometry"
        );
        coverage.colliders.pop();
        b.pose.rotation = Quat::IDENTITY;
        b.scale.x = 0.1; // narrow horizontal / portrait view excludes point
        assert_eq!(coverage.shared(&a, &b).0, 0.0);
        b.scale.x = 1.0;
        assert_eq!(coverage.shared(&a, &b).0, 1.0);
        b.pose.rotation = Quat::from_rotation_y(std::f32::consts::PI);
        assert_eq!(coverage.shared(&a, &b).0, 0.0);
    }

    #[test]
    fn body_points_behind_annotation_opaque_glazing_are_hidden() {
        let mut coverage = Coverage {
            colliders: vec![Collider {
                center: Vec3::new(0.0, 1.0, -2.0),
                half: Vec3::new(2.0, 2.0, 0.01),
                inverse: Quat::IDENTITY,
                label: "window",
            }],
            body_points: vec![Vec3::new(0.0, 1.5, -3.0)],
            shell: None,
        };
        let pose = Transform::from_xyz(0.0, 1.5, 0.0);
        assert!(!coverage.body_visible(pose, 1.0));
        coverage.body_points[0].z = -1.0;
        assert!(coverage.body_visible(pose, 1.0));
        coverage.body_points[0].x = 2.0;
        assert!(!coverage.body_visible(pose, 1.0));
    }

    #[test]
    fn walls_occlude_content_and_rays_handle_parallel_faces() {
        assert_eq!(
            ray_box(Vec3::new(0.0, 0.0, 2.0), Vec3::NEG_Z, Vec3::ONE),
            Some(1.0)
        );
        assert_eq!(
            ray_box(Vec3::new(2.0, 0.0, 2.0), Vec3::NEG_Z, Vec3::ONE),
            None
        );
        let scene = IndoorManifest::generate_with_humans(
            10,
            super::super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.0,
        )
        .unwrap();
        let wall_view = IndoorCamera {
            start: Vec3::new(0.0, 1.5, 0.0),
            end: Vec3::new(0.0, 1.5, 0.0),
            target: Vec3::new(0.0, 1.5, 1.0),
            fov_degrees: 2.0,
            motion: None,
        };
        assert!(!Coverage::new(&scene).suitable(&wall_view, false));
    }
}
