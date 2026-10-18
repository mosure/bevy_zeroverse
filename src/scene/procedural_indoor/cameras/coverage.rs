//! Cheap first-surface visibility proposals. Rendered labels remain the oracle.
//! Glazing blocks labels, matching the dataset's annotation policy.
use super::super::envelope::EnvelopeProgram;
use crate::scene::procedural_indoor::layout::{IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use std::collections::BTreeMap;
mod index;

struct Collider {
    center: Vec3,
    half: Vec3,
    inverse: Quat,
    label: &'static str,
}

pub(crate) struct Coverage {
    seed: u64,
    jitter: Vec2,
    aspect: f32,
    colliders: Vec<Collider>,
    index: Option<index::Index>,
    body_points: Vec<Vec3>,
    shell: Option<(EnvelopeProgram, Vec3, f32, f32, f32)>,
}

/// Pixel-centred proxy rays over the full calibrated frustum, not just its centre.
pub(super) struct VisibilityView {
    pose: Transform,
    scale: Vec2,
    offset: Vec2,
    points: Vec<Option<Vec3>>,
}

impl VisibilityView {
    pub(super) fn surfaces(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.points.iter().flatten().copied()
    }
}

impl Coverage {
    /// Separate deterministic lens stream; zero jitter retains exact legacy
    /// proposals. Final manifests archive the realized value, not just this prior.
    pub(super) fn optical_center(&self, camera: &IndoorCamera) -> Vec2 {
        use rand::Rng;
        camera.principal_point_offset.unwrap_or_else(|| {
            if self.jitter == Vec2::ZERO {
                return Vec2::ZERO;
            }
            let p = camera.start.to_array().map(f32::to_bits);
            let key =
                u64::from(p[0]) ^ u64::from(p[1]).rotate_left(21) ^ u64::from(p[2]).rotate_left(43);
            let mut rng = super::super::layout::stream(self.seed ^ key, 7515);
            Vec2::new(rng.random_range(-1.0..=1.0), rng.random_range(-1.0..=1.0)) * self.jitter
        })
    }

    fn shell_hit(&self, origin: Vec3, ray: Vec3) -> Option<(f32, &'static str)> {
        self.shell
            .as_ref()
            .and_then(|(e, size, door, width, height)| {
                e.ray_hit(*size, *door, *width, *height, origin, ray)
            })
    }
    fn first_hit(&self, origin: Vec3, ray: Vec3) -> Option<f32> {
        self.closest_collider(origin, ray, f32::INFINITY, false)
            .map(|(_, distance)| distance)
            .into_iter()
            .chain(self.shell_hit(origin, ray).map(|hit| hit.0))
            .min_by(f32::total_cmp)
    }

    fn closest_collider(
        &self,
        origin: Vec3,
        ray: Vec3,
        limit: f32,
        skip_people: bool,
    ) -> Option<(usize, f32)> {
        if let Some(index) = &self.index {
            return index.closest(&self.colliders, origin, ray, limit, skip_people);
        }
        // The scalar oracle also keeps small test fixtures independent of the
        // acceleration structure. Original collider order resolves equal hits.
        self.colliders
            .iter()
            .enumerate()
            .filter(|(_, c)| !skip_people || c.label != "person")
            .filter_map(|(i, c)| {
                ray_box(c.inverse * (origin - c.center), c.inverse * ray, c.half)
                    .filter(|&distance| distance < limit)
                    .map(|distance| (i, distance))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
    }

    pub(super) fn view(&self, camera: &IndoorCamera, aspect: f32, t: f32) -> VisibilityView {
        let pose = camera.transform_at(t);
        let scale = Vec2::new(aspect, 1.0) * (camera.fov_degrees.to_radians() * 0.5).tan();
        let offset = self.optical_center(camera);
        let mut points = Vec::with_capacity(117);
        for y in 0..9 {
            for x in 0..13 {
                let local = Vec3::new(
                    (2.0 * (x as f32 + 0.5) / 13.0 - 1.0 - 2. * offset.x) * scale.x,
                    (2.0 * (y as f32 + 0.5) / 9.0 - 1.0 + 2. * offset.y) * scale.y,
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
            offset,
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
                || (local.x / (-local.z * target.scale.x) + 2. * target.offset.x).abs() >= 1.
                || (local.y / (-local.z * target.scale.y) - 2. * target.offset.y).abs() >= 1.
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
            if o.kind == super::super::layout::ObjectKind::Cable {
                // A hanging lead bounds a large empty volume. Only its jacket
                // occludes geometry; its overall AABB is not a solid object.
                for (lo, hi) in o.obstacle_bounds() {
                    colliders.push(Collider {
                        center: (lo + hi) * 0.5,
                        half: (hi - lo) * 0.5,
                        inverse: Quat::IDENTITY,
                        label: "cable",
                    });
                }
                continue;
            }
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
            seed: scene.seed,
            jitter: scene.camera_settings.principal_point_jitter,
            aspect: scene.camera_aspect_ratio,
            index: Some(index::Index::new(&colliders)),
            colliders,
            body_points,
            shell: scene.envelope.clone().map(|e| {
                (
                    e,
                    scene.room_size,
                    scene.door_x,
                    scene.entry_width(),
                    scene.entry_height(),
                )
            }),
        }
    }

    fn body_visible(&self, pose: Transform, scale: f32, offset: Vec2) -> bool {
        self.body_points.iter().any(|point| {
            let local = pose.rotation.inverse() * (*point - pose.translation);
            // Test anatomical points, not empty corners of a person's envelope.
            // Otherwise two rays through a doorway can hit the padded envelope
            // while the actual person remains behind annotation-opaque glazing.
            if local.z >= -0.1
                || (local.x / (-local.z * scale * self.aspect.min(1.))
                    + 2. * offset.x * self.aspect / self.aspect.min(1.))
                .abs()
                    > 0.8
                || (local.y / (-local.z * scale) - 2. * offset.y).abs() > 0.8
            {
                return false;
            }
            let displacement = *point - pose.translation;
            let distance = displacement.length();
            let ray = displacement / distance;
            !self
                .shell_hit(pose.translation, ray)
                .is_some_and(|hit| hit.0 < distance - 0.06)
                && self
                    .closest_collider(pose.translation, ray, distance - 0.06, true)
                    .is_none()
        })
    }

    pub fn suitable(&self, camera: &IndoorCamera, require_person: bool) -> bool {
        [0.0, 0.5, 1.0].into_iter().all(|t| {
            let pose = camera.transform_at(t);
            let scale = (camera.fov_degrees.to_radians() * 0.5).tan();
            let offset = self.optical_center(camera);
            if require_person && !self.body_visible(pose, scale, offset) {
                return false;
            }
            let mut counts = BTreeMap::new();
            // Central square stays within standard landscape captures, so the
            // policy does not rely on peripheral content just outside the frame.
            for y in 0..5 {
                for x in 0..7 {
                    let ray = pose.rotation
                        * Vec3::new(
                            ((x as f32 / 6.0 - 0.5) * 1.6 * self.aspect.min(1.)
                                - 2. * offset.x * self.aspect)
                                * scale,
                            ((y as f32 / 4.0 - 0.5) * 1.6 + 2. * offset.y) * scale,
                            -1.0,
                        )
                        .normalize();
                    let (nearest, mut label) = self
                        .shell_hit(pose.translation, ray)
                        .unwrap_or((f32::INFINITY, "background"));
                    if let Some((index, _)) =
                        self.closest_collider(pose.translation, ray, nearest, false)
                    {
                        label = self.colliders[index].label;
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
            offset: Vec2::ZERO,
            pose: Transform::IDENTITY,
            scale: Vec2::ONE,
            points: vec![Some(point)],
        };
        let mut b = VisibilityView {
            offset: Vec2::ZERO,
            pose: Transform::from_xyz(2.0, 0.0, 0.0).looking_at(point, Vec3::Y),
            scale: Vec2::ONE,
            points: vec![Some(point)],
        };
        let mut coverage = Coverage {
            seed: 0,
            jitter: Vec2::ZERO,
            aspect: 1.,
            index: None,
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
            seed: 0,
            jitter: Vec2::ZERO,
            aspect: 1.,
            index: None,
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
        assert!(!coverage.body_visible(pose, 1.0, Vec2::ZERO));
        coverage.body_points[0].z = -1.0;
        assert!(coverage.body_visible(pose, 1.0, Vec2::ZERO));
        coverage.body_points[0].x = 2.0;
        assert!(!coverage.body_visible(pose, 1.0, Vec2::ZERO));
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
            principal_point_offset: None,
            start: Vec3::new(0.0, 1.5, 0.0),
            end: Vec3::new(0.0, 1.5, 0.0),
            target: Vec3::new(0.0, 1.5, 1.0),
            fov_degrees: 2.0,
            motion: None,
        };
        assert!(!Coverage::new(&scene).suitable(&wall_view, false));
    }
}
