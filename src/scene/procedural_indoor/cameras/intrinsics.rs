//! Seeded perspective-lens programs, shared by independent views and camera rigs.
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const ULTRA_WIDE_START_DEGREES: f32 = 108.0;
pub const MAX_VERTICAL_FOV_DEGREES: f32 = 150.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IntrinsicsSettings {
    /// Vertical FOV in degrees; horizontal FOV follows the actual image aspect.
    pub fov_y_min_degrees: f32,
    pub fov_y_max_degrees: f32,
    /// Proposal probability above 108 degrees when the interval spans that value.
    /// Within each interval, normalized focal length is sampled logarithmically.
    pub ultra_wide_probability: f32,
}

impl Default for IntrinsicsSettings {
    fn default() -> Self {
        Self {
            fov_y_min_degrees: 28.,
            fov_y_max_degrees: 145.,
            ultra_wide_probability: 0.25,
        }
    }
}

impl IntrinsicsSettings {
    /// Bounds of the historical camera contract, for archived manifests only.
    pub(super) fn legacy() -> Self {
        Self {
            fov_y_min_degrees: 27.,
            fov_y_max_degrees: 108.,
            ultra_wide_probability: 0.,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.fov_y_min_degrees.is_finite()
            || !self.fov_y_max_degrees.is_finite()
            || self.fov_y_min_degrees < 15.
            || self.fov_y_max_degrees > MAX_VERTICAL_FOV_DEGREES
            || self.fov_y_min_degrees > self.fov_y_max_degrees
            || !(0.0..=1.0).contains(&self.ultra_wide_probability)
        {
            return Err("indoor_camera.intrinsics requires 15 <= fov_y_min_degrees <= fov_y_max_degrees <= 150 and ultra_wide_probability in [0,1]".into());
        }
        Ok(())
    }

    pub fn contains(&self, degrees: f32) -> bool {
        (self.fov_y_min_degrees..=self.fov_y_max_degrees).contains(&degrees)
    }

    pub(super) fn sample(&self, rng: &mut impl Rng) -> f32 {
        let (mut lo, mut hi) = (self.fov_y_min_degrees, self.fov_y_max_degrees);
        if lo < ULTRA_WIDE_START_DEGREES && hi > ULTRA_WIDE_START_DEGREES {
            if rng.random_bool(self.ultra_wide_probability as f64) {
                lo = ULTRA_WIDE_START_DEGREES;
            } else {
                hi = ULTRA_WIDE_START_DEGREES;
            }
        }
        if lo == hi {
            return lo;
        }
        let log_tan = rng
            .random_range((lo.to_radians() * 0.5).tan().ln()..(hi.to_radians() * 0.5).tan().ln());
        (2. * log_tan.exp().atan().to_degrees()).clamp(lo, hi)
    }

    /// Rare coverage recovery stays inside the explicitly requested lens range.
    pub(super) fn recovery(&self, preferred: f32) -> f32 {
        preferred.clamp(self.fov_y_min_degrees, self.fov_y_max_degrees)
    }
}

/// A rotation-independent sphere enclosing the complete shifted near rectangle.
/// Sweeping this radius prevents wide lens corners crossing nearby geometry.
pub(super) fn near_clearance(fov_y: f32, aspect: f32, offset_bound: Vec2) -> f32 {
    let half = Vec2::new(aspect, 1.)
        * (Vec2::ONE + 2. * offset_bound.abs())
        * (0.1 * (fov_y.to_radians() * 0.5).tan());
    Vec3::new(half.x, half.y, 0.1).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::stream;

    #[test]
    fn continuous_lens_mixture_has_wide_support_and_replays() {
        let policy = IntrinsicsSettings::default();
        policy.validate().unwrap();
        let mut a = stream(2026, 793);
        let mut b = a.clone();
        let mut wide = 0;
        let mut extreme = 0;
        let mut bins = [0usize; 12];
        for _ in 0..10000 {
            let fov = policy.sample(&mut a);
            assert_eq!(fov, policy.sample(&mut b));
            assert!(policy.contains(fov));
            wide += usize::from(fov > 108.);
            extreme += usize::from(fov > 135.);
            bins[((fov - 28.) / 10.) as usize] += 1;
        }
        assert!((2300..2700).contains(&wide), "{wide}");
        assert!(extreme > 500 && bins.iter().all(|&n| n > 100));
        for (min, max) in [(15., 50.), (125., 150.), (140., 140.)] {
            let p = IntrinsicsSettings {
                fov_y_min_degrees: min,
                fov_y_max_degrees: max,
                ..default()
            };
            p.validate().unwrap();
            for _ in 0..100 {
                assert!(p.contains(p.sample(&mut a)));
            }
        }
        for (min, max, probability) in [
            (0., 150., 0.2),
            (100., 90., 0.2),
            (28., 180., 0.2),
            (28., 145., f32::NAN),
        ] {
            assert!(IntrinsicsSettings {
                fov_y_min_degrees: min,
                fov_y_max_degrees: max,
                ultra_wide_probability: probability
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn clearance_encloses_near_corners_with_aspect_and_shift() {
        for fov in [28_f32, 108., 145., 150.] {
            for aspect in [0.6, 1., 1.6, 2.4] {
                let offset = Vec2::new(0.2, -0.15);
                let radius = near_clearance(fov, aspect, offset);
                for corner in [
                    Vec2::ONE,
                    -Vec2::ONE,
                    Vec2::new(1., -1.),
                    Vec2::new(-1., 1.),
                ] {
                    let xy = (corner + 2. * offset)
                        * Vec2::new(aspect, 1.)
                        * (0.1 * (fov.to_radians() * 0.5).tan());
                    assert!(Vec3::new(xy.x, xy.y, -0.1).length() <= radius + 1e-6);
                }
            }
        }
    }

    #[test]
    fn ultra_wide_near_plane_cannot_cross_a_wall_despite_clear_camera_center() {
        use crate::scene::procedural_indoor::{
            floorplan::FloorPlan,
            layout::{IndoorCamera, IndoorLayout, IndoorManifest},
        };
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
        scene.room_size = Vec3::new(10., 4., 10.);
        scene.envelope = None;
        scene.program = None;
        scene.entry_door = None;
        scene.floor_plan = FloorPlan::OpenHall;
        scene.objects.clear();
        scene.humans.clear();
        scene.camera_aspect_ratio = 1.6;
        let p = Vec3::new(0., 1.5, 4.45);
        let mut camera = IndoorCamera {
            start: p,
            end: p,
            target: p - 3. * Vec3::X,
            principal_point_offset: None,
            fov_degrees: 60.,
            motion: None,
        };
        assert!(scene.camera_clear(p));
        assert!(scene.camera_curve_clear(&camera));
        camera.fov_degrees = 150.;
        assert!(!scene.camera_curve_clear(&camera));
    }

    #[test]
    fn wide_camera_sets_preserve_geometry_clearance_and_replay() {
        use crate::scene::procedural_indoor::{
            cameras::CameraSettings,
            layout::{IndoorLayout, IndoorManifest},
            validation::validate_layout,
        };
        let settings = CameraSettings::parse(r#"{"baseline":0.55,"intrinsics":{"fov_y_min_degrees":125,"fov_y_max_degrees":145},"principal_point_jitter":[0.1,0.1]}"#).unwrap();
        for seed in 0..8 {
            let original =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            for aspect in [0.75, 1.6] {
                for count in [1, 3] {
                    let mut scene = original.clone();
                    scene
                        .resample_cameras(count, settings.clone(), aspect)
                        .unwrap();
                    validate_layout(&scene).unwrap();
                    assert_eq!(scene.objects, original.objects);
                    assert_eq!(scene.envelope, original.envelope);
                    let cameras = scene.cameras.clone();
                    for camera in &cameras {
                        assert!(settings.intrinsics.contains(camera.fov_degrees));
                        assert!(scene.camera_curve_clear(camera));
                    }
                    scene
                        .resample_cameras(count, settings.clone(), aspect)
                        .unwrap();
                    assert_eq!(cameras, scene.cameras);
                }
            }
        }
    }

    #[test]
    fn archived_lens_bounds_and_explicit_policy_survive_serialization() {
        use crate::scene::procedural_indoor::{
            cameras::CameraSettings,
            layout::{IndoorLayout, IndoorManifest},
        };
        let scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut json = serde_json::to_value(&scene).unwrap();
        let roundtrip: IndoorManifest = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            roundtrip.camera_settings.intrinsics,
            IntrinsicsSettings::default()
        );
        json["camera_settings"]
            .as_object_mut()
            .unwrap()
            .remove("intrinsics");
        let archived: IndoorManifest = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(
            archived.camera_settings.intrinsics,
            IntrinsicsSettings::legacy()
        );
        json.as_object_mut().unwrap().remove("camera_settings");
        let archived: IndoorManifest = serde_json::from_value(json).unwrap();
        assert_eq!(
            archived.camera_settings.intrinsics,
            IntrinsicsSettings::legacy()
        );
        assert!(archived.camera_settings.multiview.is_none());
        let policy = CameraSettings::parse(
            r#"{"intrinsics":{"fov_y_min_degrees":135,"fov_y_max_degrees":145}}"#,
        )
        .unwrap();
        assert_eq!(
            CameraSettings::parse(&serde_json::to_string(&policy).unwrap()).unwrap(),
            policy
        );
        assert!(CameraSettings::parse(r#"{"intrinsics":{"fov_y_max_degrees":180}}"#).is_err());
    }
}
