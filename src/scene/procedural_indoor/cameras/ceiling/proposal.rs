//! A camera slot's fixed calibration and local mount orientation. Both greedy
//! placement and bounded recovery use this same physical camera construction.
use super::*;
use crate::scene::procedural_indoor::layout::{stream, IndoorManifest};
use rand::Rng;

#[derive(Clone, Copy)]
pub(super) struct Prototype {
    pub lens: Option<crate::calibration::FisheyeSensor>,
    pub fov_degrees: f32,
    pub standoff_m: f32,
    /// Local camera rotation relative to the supporting-plane frame.
    pub local_rotation: Quat,
}

impl Prototype {
    /// Invert the fixed viewing direction to suggest a ceiling contact above a
    /// shared room surface. Two bounded plane evaluations handle soffit edges;
    /// this is only guidance, never a substitute for final clearance/visibility.
    pub fn overlooking(self, scene: &IndoorManifest, target: Vec3) -> Option<Vec2> {
        let mut p = target.xz();
        for _ in 0..2 {
            let (surface, normal) = placement::surface(scene, p);
            let rotation =
                Transform::IDENTITY.looking_to(normal, Vec3::Z).rotation * self.local_rotation;
            let forward = rotation * Vec3::NEG_Z;
            let distance = ((target - surface).dot(normal) - self.standoff_m) / forward.dot(normal);
            if !distance.is_finite() || distance <= 0.1 {
                return None;
            }
            p = (target - forward * distance - normal * self.standoff_m).xz();
        }
        p.is_finite().then_some(p)
    }

    pub fn sample(scene: &IndoorManifest, index: usize, mounting: &CeilingCameraSettings) -> Self {
        let mut prior = stream(scene.seed, 0x4343_0000 + index as u64);
        let lens = scene.camera_settings.lens_for(scene.seed, index);
        let fov_degrees = scene
            .camera_settings
            .intrinsics_for(scene.seed, index)
            .sample(&mut prior);
        let standoff_m = prior.random_range(mounting.standoff_m[0]..=mounting.standoff_m[1]);
        let roll = prior.random_range(0.0..std::f32::consts::TAU);
        let tilt = if scene.camera_settings.ceiling_mounted(scene.seed, index) {
            orientation::sample(mounting, scene.seed, index)
        } else {
            Quat::IDENTITY
        };
        Self {
            lens,
            fov_degrees,
            standoff_m,
            local_rotation: tilt * Quat::from_rotation_z(roll),
        }
    }

    pub fn mounted_at(
        self,
        scene: &IndoorManifest,
        index: usize,
        p: Vec2,
        rng: &mut impl Rng,
    ) -> IndoorCamera {
        let mounting = scene.camera_settings.ceiling_mount.as_ref().unwrap();
        let (surface, normal) = placement::surface(scene, p);
        let start = surface + normal * self.standoff_m;
        let rotation =
            Transform::IDENTITY.looking_to(normal, Vec3::Z).rotation * self.local_rotation;
        let mount = CeilingCameraMount {
            surface,
            inward_normal: normal,
            rotation,
            standoff_m: self.standoff_m,
            body_radius_m: mounting.body_radius_m,
        };
        let mut camera = IndoorCamera {
            target: start + mount.optical_axis() * 2.,
            ceiling_mount: Some(mount),
            fisheye: self.lens,
            principal_point_offset: None,
            start,
            end: start,
            fov_degrees: self.fov_degrees,
            motion: None,
        };
        trajectory::propose(scene, index, &mut camera, rng);
        camera
    }
}
