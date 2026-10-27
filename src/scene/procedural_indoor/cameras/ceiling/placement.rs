//! Conservative mount-body reservations, cached once per placement search.
use super::*;
use crate::scene::procedural_indoor::layout::{segment_hits_box, IndoorManifest, IndoorObject};

pub(super) struct MountSpace {
    obstacles: Vec<(Vec3, Vec3)>,
}
impl MountSpace {
    pub fn new(scene: &IndoorManifest) -> Self {
        let mut obstacles = scene.camera_obstacles();
        obstacles.extend(scene.objects.iter().flat_map(IndoorObject::obstacle_bounds));
        obstacles.extend(scene.humans.iter().map(|h| h.bounds()));
        if let Some(motion) = &scene.indoor_motion {
            obstacles.extend(motion.reservations());
        }
        Self { obstacles }
    }

    pub fn clear(&self, scene: &IndoorManifest, camera: &IndoorCamera) -> bool {
        if camera.ceiling_mount.is_none()
            || !camera.end.is_finite()
            || !camera.target.is_finite()
            || !camera.fov_degrees.is_finite()
        {
            return false;
        }
        // Fixed CCTV retains the original single-pose checks. Rail support and
        // swept-volume allocation are paid only when motion is requested.
        if camera.motion.is_none() && camera.start == camera.end {
            return self.pose_clear(scene, camera);
        }
        if !super::trajectory::plane_clear(scene, camera) {
            return false;
        }
        // Endpoint tests enforce wall/floor and optical-axis clearance. Sweep
        // each rigid component separately: merging the broad lens clearance
        // into the narrow mount body invents occupied space beside roof ribs.
        let mut end = camera.clone();
        let delta = camera.end - camera.start;
        end.start = camera.end;
        end.end = camera.end;
        end.target += delta;
        end.motion = None;
        end.ceiling_mount.as_mut().unwrap().surface += delta;
        if !self.pose_clear(scene, camera) || !self.pose_clear(scene, &end) {
            return false;
        }
        if delta == Vec3::ZERO {
            return true;
        }
        let body = camera.ceiling_sweep_bounds().unwrap();
        let (a, b) = near_bounds(scene, camera);
        let (c, d) = near_bounds(scene, &end);
        let near = (a.min(c), b.max(d));
        let forward = camera.ceiling_mount.as_ref().unwrap().inward_normal * 0.6;
        let axis = [
            camera.start,
            end.start,
            camera.start + forward,
            end.start + forward,
        ];
        let optical = (
            axis.into_iter().fold(Vec3::splat(f32::INFINITY), Vec3::min) - Vec3::splat(1e-5),
            axis.into_iter()
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max)
                + Vec3::splat(1e-5),
        );
        // The rail is linear with fixed orientation. Endpoint unions therefore
        // conservatively contain each component for every intermediate time.
        [body, near, optical].into_iter().all(|(lo, hi)| {
            !self
                .obstacles
                .iter()
                .any(|(a, b)| lo.cmplt(*b).all() && hi.cmpgt(*a).all())
        })
    }

    fn pose_clear(&self, scene: &IndoorManifest, camera: &IndoorCamera) -> bool {
        let (Some(policy), Some(mount)) =
            (&scene.camera_settings.ceiling_mount, &camera.ceiling_mount)
        else {
            return false;
        };
        if !mount.surface.is_finite()
            || !mount.inward_normal.is_finite()
            || !mount.rotation.is_finite()
            || (mount.rotation.length_squared() - 1.).abs() > 1e-5
            || !(policy.standoff_m[0]..=policy.standoff_m[1]).contains(&mount.standoff_m)
            || mount.body_radius_m != policy.body_radius_m
        {
            return false;
        }
        let (surface, normal) = surface(scene, mount.surface.xz());
        let radius = mount.body_radius_m;
        let eye = camera.start;
        if surface.distance(mount.surface) > 1e-4
            || normal.distance(mount.inward_normal) > 1e-5
            || (mount.rotation * Vec3::NEG_Z).distance(normal) > 1e-5
            || eye.distance(surface + normal * mount.standoff_m) > 1e-4
            || camera.target.distance(eye + normal * 2.) > 1e-4
            || !scene.in_primary_room(eye, policy.wall_clearance_m)
            || eye.y < scene.floor_height(eye.xz()) + 1.5
            || scene
                .envelope
                .as_ref()
                .is_some_and(|e| !e.volume_clear(scene.room_size, eye, radius))
        {
            return false;
        }
        let half = scene.room_size.xz() * 0.5;
        if eye
            .xz()
            .abs()
            .cmpgt(half - Vec2::splat(policy.wall_clearance_m))
            .any()
        {
            return false;
        }
        // A capsule from the mounting base to the lens reserves the camera body.
        // Its base is just inside the supporting surface, never through a soffit.
        let base = surface + normal * (radius + 0.006);
        let front = eye + normal * radius;
        let pad = Vec3::splat(radius);
        if self
            .obstacles
            .iter()
            .any(|(lo, hi)| segment_hits_box(base, front, *lo - pad, *hi + pad))
        {
            return false;
        }
        // Reject near-field obstructions along the optical axis as well as a
        // clipped lens. Peripheral occlusion is legitimate scene content.
        if !near_clear(scene, camera, &self.obstacles) {
            return false;
        }
        !self
            .obstacles
            .iter()
            .any(|(lo, hi)| segment_hits_box(eye, eye + normal * 0.6, *lo, *hi))
    }
}

fn near_bounds(scene: &IndoorManifest, camera: &IndoorCamera) -> (Vec3, Vec3) {
    let pose = camera.transform_at(0.);
    if camera.fisheye.is_some() {
        // Radial near clip: reserve the complete 10 cm sphere, including >180° rays.
        (
            camera.start - Vec3::splat(0.101),
            camera.start + Vec3::splat(0.101),
        )
    } else {
        // Rectilinear near clip: reserve the pyramid from eye to the calibrated
        // near rectangle. A sphere would incorrectly intersect the ceiling
        // behind a downward-facing wide lens.
        let half = Vec2::new(scene.camera_aspect_ratio, 1.)
            * 0.1
            * (camera.fov_degrees.to_radians() * 0.5).tan();
        let offset = camera
            .principal_point_offset
            .unwrap_or(scene.camera_settings.principal_point_jitter);
        let mut lo = camera.start;
        let mut hi = lo;
        for sx in [-1., 1.] {
            for sy in [-1., 1.] {
                // Reserve both signs of requested jitter until the realized center
                // is archived after placement.
                let point = pose.transform_point(Vec3::new(
                    sx * (1. + 2. * offset.x.abs()) * half.x,
                    sy * (1. + 2. * offset.y.abs()) * half.y,
                    -0.1,
                ));
                lo = lo.min(point);
                hi = hi.max(point);
            }
        }
        (lo, hi)
    }
}
fn near_clear(scene: &IndoorManifest, camera: &IndoorCamera, obstacles: &[(Vec3, Vec3)]) -> bool {
    let (lo, hi) = near_bounds(scene, camera);
    !obstacles
        .iter()
        .any(|(a, b)| lo.cmplt(*b).all() && hi.cmpgt(*a).all())
}

/// Analytic normal of the same sloped roof / flat soffit used by construction.
pub(super) fn surface(scene: &IndoorManifest, p: Vec2) -> (Vec3, Vec3) {
    let height = scene.ceiling_height(p);
    let normal = scene.envelope.as_ref().map_or(Vec3::NEG_Y, |e| {
        if height < e.roof_height(scene.room_size, p) - 1e-5 {
            Vec3::NEG_Y
        } else {
            Vec3::new(
                -e.ceiling_drop.x / scene.room_size.x,
                -1.,
                -e.ceiling_drop.y / scene.room_size.z,
            )
            .normalize()
        }
    });
    (Vec3::new(p.x, height, p.y), normal)
}

pub(crate) fn clear(scene: &IndoorManifest, camera: &IndoorCamera) -> bool {
    MountSpace::new(scene).clear(scene, camera)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{cameras::CameraMotion, layout::IndoorLayout};

    #[test]
    fn moving_mount_reserves_body_lens_and_optical_sweeps_without_empty_corners() {
        let mut scene =
            IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.room_size = Vec3::new(10., 4., 10.);
        scene.program = None;
        let e = scene.envelope.as_mut().unwrap();
        e.footprint = vec![
            Vec2::new(-5., -5.),
            Vec2::new(5., -5.),
            Vec2::new(5., 5.),
            Vec2::new(-5., 5.),
        ];
        e.ceiling_drop = Vec2::ZERO;
        e.soffits.clear();
        e.floor_patches.clear();
        e.mezzanine = None;
        scene.rug_supports.clear();
        let policy = CeilingCameraSettings {
            motion_fraction: 1.,
            ..default()
        };
        let rotation = Transform::IDENTITY
            .looking_to(Vec3::NEG_Y, Vec3::Z)
            .rotation;
        let start = Vec3::new(0., 3.78, 0.);
        let delta = Vec3::X * 0.6;
        let target = start + Vec3::NEG_Y * 2.;
        let camera = IndoorCamera {
            ceiling_mount: Some(CeilingCameraMount {
                surface: Vec3::Y * 4.,
                inward_normal: Vec3::NEG_Y,
                rotation,
                standoff_m: 0.22,
                body_radius_m: policy.body_radius_m,
            }),
            fisheye: policy.lenses.sample(scene.seed, 0),
            principal_point_offset: None,
            start,
            end: start + delta,
            target,
            fov_degrees: 120.,
            motion: Some(CameraMotion {
                control: [start + delta / 3., start + delta * (2. / 3.)],
                target_end: target + delta,
                roll: [0.; 2],
                route: vec![],
                orientations: Some([rotation; 2]),
            }),
        };
        scene.camera_settings.ceiling_mount = Some(policy);
        let mut space = MountSpace { obstacles: vec![] };
        assert!(space.clear(&scene, &camera));
        // Above the lens's near sphere, beside the narrower body: real space,
        // even though it lies inside the combined swept bounding box.
        space.obstacles = vec![(Vec3::new(0.2, 3.90, 0.08), Vec3::new(0.4, 3.92, 0.09))];
        assert!(space.clear(&scene, &camera));
        // Thin obstacles only encountered mid-trajectory must still fail, in
        // each independent component (body, lens and first 60 cm of sight).
        for (lo, hi) in [
            (Vec3::new(0.2, 3.90, 0.015), Vec3::new(0.4, 3.92, 0.025)),
            (Vec3::new(0.2, 3.81, 0.08), Vec3::new(0.4, 3.83, 0.09)),
            (Vec3::new(0.2, 3.20, -0.01), Vec3::new(0.4, 3.23, 0.01)),
        ] {
            space.obstacles = vec![(lo, hi)];
            assert!(space.pose_clear(&scene, &camera));
            assert!(
                !space.clear(&scene, &camera),
                "missed swept obstacle {lo:?} {hi:?}"
            );
        }
    }
}
