//! Immutable physical reservations shared by proposals for one furnished room.
//! Optical visibility has a separate index because glass, open furniture and
//! sloped ceiling mounts have different optical and collision representations.
use super::{intrinsics, IndoorCamera, IndoorManifest, TrajectorySampler};
use crate::scene::procedural_indoor::layout::{segment_hits_box, IndoorObject, CAMERA_CLEARANCE};
use bevy::prelude::*;
use std::sync::OnceLock;

type Bounds = (Vec3, Vec3);

/// Reuse nearby reservations for consecutive chords. The actual forward ray
/// and previous/current poses determine each query, including on a banked or
/// reversed path. The final collision test is identical for every candidate.
#[derive(Default)]
struct CurveObstacles {
    region: Option<Bounds>,
    candidates: Vec<Bounds>,
}

impl CurveObstacles {
    fn near<'a>(
        &'a mut self,
        all: &'a [Bounds],
        previous: Vec3,
        p: Vec3,
        ahead: Vec3,
        pad: Vec3,
    ) -> &'a [Bounds] {
        if all.len() < 24 || !ahead.is_finite() || !p.is_finite() || !previous.is_finite() {
            return all;
        }
        let min = previous.min(p).min(ahead) - pad;
        let max = previous.max(p).max(ahead) + pad;
        if self
            .region
            .is_none_or(|(lo, hi)| !min.cmpge(lo).all() || !max.cmple(hi).all())
        {
            // Amortize the room scan across successive poses and include
            // touching boxes on all axes. Extra padding is broad phase only.
            let lo = min - Vec3::ONE;
            let hi = max + Vec3::ONE;
            self.candidates.clear();
            self.candidates.extend(
                all.iter()
                    .copied()
                    .filter(|(a, b)| a.cmple(hi).all() && b.cmpge(lo).all()),
            );
            // Keep a roundoff guard between a reused query and an excluded
            // box, including for large coordinates and a grazing segment.
            let guard = (lo.abs().max(hi.abs()) + Vec3::ONE) * (8. * f32::EPSILON);
            self.region = Some((lo + guard, hi - guard));
        }
        &self.candidates
    }
}

fn fixture_drop(scene: &IndoorManifest) -> f32 {
    scene.program.as_ref().map_or(
        [0.10, 0.42, 0.06][scene.lighting_design as usize % 3],
        |p| p.light_drop,
    )
}

#[derive(Default)]
pub(in crate::scene::procedural_indoor) struct Clearance {
    obstacles: OnceLock<Vec<Bounds>>,
}

impl Clearance {
    pub(super) fn bounds(&self, scene: &IndoorManifest) -> &[Bounds] {
        self.obstacles.get_or_init(|| {
            let mut obstacles = scene.camera_obstacles();
            obstacles.extend(scene.objects.iter().flat_map(IndoorObject::obstacle_bounds));
            obstacles.extend(scene.humans.iter().map(|h| h.bounds()));
            obstacles
        })
    }

    pub(in crate::scene::procedural_indoor) fn point_clear(
        &self,
        scene: &IndoorManifest,
        p: Vec3,
    ) -> bool {
        let half = scene.room_size * 0.5;
        if !p.is_finite()
            || (scene.camera_settings.primary_room && !scene.in_primary_room(p, CAMERA_CLEARANCE))
            || p.x.abs() > half.x - 0.50
            || p.z.abs() > half.z - 0.50
            || p.y < scene.floor_height(p.xz()) + 0.70
            || scene
                .envelope
                .as_ref()
                .is_some_and(|e| !e.volume_clear(scene.room_size, p, CAMERA_CLEARANCE))
            || p.y > scene.ceiling_height(p.xz()) - fixture_drop(scene) - 0.04 - CAMERA_CLEARANCE
        {
            return false;
        }
        let pad = Vec3::splat(CAMERA_CLEARANCE);
        !self
            .bounds(scene)
            .iter()
            .any(|&(lo, hi)| p.cmpge(lo - pad).all() && p.cmple(hi + pad).all())
    }

    pub(in crate::scene::procedural_indoor) fn path_clear(
        &self,
        scene: &IndoorManifest,
        start: Vec3,
        end: Vec3,
    ) -> bool {
        let pad = Vec3::splat(CAMERA_CLEARANCE);
        self.point_clear(scene, start)
            && self.point_clear(scene, end)
            && scene
                .envelope
                .as_ref()
                .is_none_or(|e| e.segment_clear(scene.room_size, start, end, CAMERA_CLEARANCE))
            && !self
                .bounds(scene)
                .iter()
                .any(|&(lo, hi)| segment_hits_box(start, end, lo - pad, hi + pad))
    }

    pub(in crate::scene::procedural_indoor) fn view_clear(
        &self,
        scene: &IndoorManifest,
        position: Vec3,
        target: Vec3,
    ) -> bool {
        let distance = position.distance(target);
        if !target.is_finite() || distance < 1.5 {
            return false;
        }
        let end = position.lerp(target, 1.0 / distance);
        let pad = Vec3::splat(0.01);
        !self
            .bounds(scene)
            .iter()
            .any(|&(lo, hi)| segment_hits_box(position, end, lo - pad, hi + pad))
    }

    pub(in crate::scene::procedural_indoor) fn curve_clear(
        &self,
        scene: &IndoorManifest,
        camera: &IndoorCamera,
    ) -> bool {
        if camera.ceiling_mount.is_some() {
            return super::ceiling::mount_clear(scene, camera);
        }
        // Fisheye clips radially at 10 cm. Its independent pinhole FOV draw is
        // not its projection, and must not inflate the swept lens volume.
        let clearance = CAMERA_CLEARANCE.max(if camera.fisheye.is_some() {
            0.101
        } else {
            intrinsics::near_clearance(
                camera.fov_degrees,
                scene.camera_aspect_ratio,
                camera
                    .principal_point_offset
                    .unwrap_or(scene.camera_settings.principal_point_jitter),
            )
        });
        if !clearance.is_finite() {
            return false;
        }
        let mut runtime = camera.runtime_trajectory();
        let endpoints = [camera.start, camera.end];
        let (control, error): (&[Vec3], f32) =
            if let TrajectorySampler::CubicBezier { positions, .. } = &runtime {
                // The linear interpolation error on a subinterval is <= max|P''| h²/8.
                // Inflate each swept chord by that bound, not just sampled positions.
                let second = 6.0
                    * (positions[0] - 2.0 * positions[1] + positions[2])
                        .length()
                        .max((positions[1] - 2.0 * positions[2] + positions[3]).length());
                (positions, second / (8.0 * 64.0 * 64.0))
            } else if let TrajectorySampler::WaypointPath { positions, .. } = &runtime {
                if positions.len() < 2 {
                    return false;
                }
                (positions, 0.0)
            } else {
                (&endpoints, 0.0)
            };
        let ceiling = scene.room_size.y - fixture_drop(scene) - 0.04 - clearance;
        // A Bezier stays inside the convex hull of its control points.
        if control.iter().any(|p| {
            !p.is_finite()
                || p.x.abs() > scene.room_size.x * 0.5 - 0.50_f32.max(clearance)
                || p.z.abs() > scene.room_size.z * 0.5 - 0.50_f32.max(clearance)
                || p.y < 0.70_f32.max(clearance)
                || p.y > ceiling
                || (scene.camera_settings.primary_room && !scene.in_primary_room(*p, clearance))
        }) {
            return false;
        }
        let obstacles = self.bounds(scene);
        let pad = Vec3::splat(clearance + error);
        // Check every routed segment at the actual lens clearance; a temporal
        // sample grid can straddle a short corner and miss the original route.
        if matches!(runtime, TrajectorySampler::WaypointPath { .. })
            && control.windows(2).any(|p| {
                scene
                    .envelope
                    .as_ref()
                    .is_some_and(|e| !e.segment_clear(scene.room_size, p[0], p[1], clearance))
                    || obstacles
                        .iter()
                        .any(|(lo, hi)| segment_hits_box(p[0], p[1], *lo - pad, *hi + pad))
            })
        {
            return false;
        }
        // The upright prior belongs to automatic look-at paths. A handheld
        // request may intentionally bank farther while clearing the same swept lens volume.
        let limit_automatic_bank = camera
            .motion
            .as_ref()
            .is_none_or(|m| m.orientations.is_none());
        // Interpolation can add a little bank between independently aimed knots.
        let max_slope = scene
            .camera_settings
            .roll
            .support_degrees()
            .max(8.05)
            .to_radians()
            .tan();
        let first = runtime.sample(0.0);
        let mut previous = first.translation;
        if scene
            .envelope
            .as_ref()
            .is_some_and(|e| !e.volume_clear(scene.room_size, previous, clearance + error))
        {
            return false;
        }
        let mut nearby = CurveObstacles::default();
        // The static sampler returns its cached pose at every time.
        let last = if matches!(runtime, TrajectorySampler::Static { .. }) {
            0
        } else {
            64
        };
        for i in 0..=last {
            let pose = if i == 0 {
                first
            } else {
                runtime.sample(i as f32 / 64.0)
            };
            let ahead = pose.translation + pose.rotation * Vec3::NEG_Z;
            if pose.translation.y
                < scene.floor_height(pose.translation.xz()) + 0.70_f32.max(clearance)
                || !pose.rotation.is_finite()
                || scene.envelope.as_ref().is_some_and(|e| {
                    !e.segment_from_clear_start(
                        scene.room_size,
                        previous,
                        pose.translation,
                        clearance + error,
                    )
                })
                || pose.translation.y
                    > scene.ceiling_height(pose.translation.xz()) - (scene.room_size.y - ceiling)
                || (limit_automatic_bank && {
                    let gravity_up = pose.rotation.inverse() * Vec3::Y;
                    gravity_up.x.abs() > max_slope * gravity_up.y + 1e-5
                })
                || pose.translation.distance(camera.target) < 1.5
                || (if last == 0 {
                    obstacles
                } else {
                    nearby.near(
                        obstacles,
                        previous,
                        pose.translation,
                        ahead,
                        pad.max(Vec3::splat(0.01)),
                    )
                })
                .iter()
                .any(|(lo, hi)| {
                    segment_hits_box(previous, pose.translation, *lo - pad, *hi + pad)
                        || segment_hits_box(
                            pose.translation,
                            ahead,
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
    use super::*;
    use crate::scene::procedural_indoor::layout::{stream, IndoorLayout};
    use rand::Rng;

    #[test]
    fn curve_broad_phase_preserves_swept_body_and_rotating_lens_contacts() {
        let bounds: Vec<_> = (-30..30)
            .map(|i| {
                let p = Vec3::new(i as f32 * 0.47, (i % 3) as f32 * 0.12, 0.31);
                (p, p + Vec3::new(0.02, 0.43, 0.09))
            })
            .collect();
        let mut nearby = CurveObstacles::default();
        let mut previous = Vec3::new(-5., 0.25, -0.1);
        let pad = Vec3::splat(0.15);
        let lens = Vec3::splat(0.01);
        let mut counts = [0; 2];
        for i in 0..600 {
            let t = i as f32 * 0.061;
            let p = Vec3::new(5. * t.sin(), 0.25, 0.16 * (t * 2.).cos());
            let ahead = p + Quat::from_rotation_y(t * 3.) * Vec3::Z;
            let hit = |boxes: &[Bounds]| {
                boxes.iter().any(|&(lo, hi)| {
                    segment_hits_box(previous, p, lo - pad, hi + pad)
                        || segment_hits_box(p, ahead, lo - lens, hi + lens)
                })
            };
            let subset = nearby.near(&bounds, previous, p, ahead, pad);
            let expected = hit(&bounds);
            assert_eq!(hit(subset), expected);
            assert!(subset.len() < bounds.len());
            counts[usize::from(expected)] += 1;
            previous = p;
        }
        assert!(counts.into_iter().all(|n| n > 0));
    }

    #[test]
    fn cached_collision_field_matches_fresh_reservations_for_each_room() {
        let mut rng = stream(31, 992);
        for (seed, layout) in [
            (32, IndoorLayout::Conference),
            (81, IndoorLayout::StairHall),
            (49, IndoorLayout::Courtyard),
        ] {
            let room = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.25).unwrap();
            let cache = Clearance::default();
            assert!(!cache.point_clear(&room, Vec3::splat(f32::INFINITY)));
            assert!(cache.obstacles.get().is_none());
            let scalar: Vec<_> = room
                .objects
                .iter()
                .flat_map(IndoorObject::obstacle_bounds)
                .chain(room.humans.iter().map(|h| h.bounds()))
                .chain(room.camera_obstacles())
                .collect();
            // The optical-axis test previously streamed object, human, then
            // architectural reservations. Only their union determines occlusion.
            for _ in 0..256 {
                let p = Vec3::new(
                    rng.random_range(-0.5..0.5),
                    rng.random_range(0.0..1.0),
                    rng.random_range(-0.5..0.5),
                ) * room.room_size;
                let target = Vec3::new(-p.x, 1.2, -p.z);
                let d = p.distance(target);
                let pad = Vec3::splat(0.01);
                let expected = d >= 1.5
                    && !scalar.iter().any(|&(lo, hi)| {
                        segment_hits_box(p, p.lerp(target, 1. / d), lo - pad, hi + pad)
                    });
                assert_eq!(cache.view_clear(&room, p, target), expected);
            }
            assert_eq!(cache.bounds(&room).len(), scalar.len());
            assert!(std::ptr::eq(cache.bounds(&room), cache.bounds(&room)));
        }
    }
}
