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
            || camera.start != camera.end
            || camera.motion.is_some()
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

fn near_clear(scene: &IndoorManifest, camera: &IndoorCamera, obstacles: &[(Vec3, Vec3)]) -> bool {
    let pose = camera.transform_at(0.);
    let (lo, hi) = if camera.fisheye.is_some() {
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
    };
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
