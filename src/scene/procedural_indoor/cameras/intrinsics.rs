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
    /// Continuous lens preference: -1 favors narrow lenses, +1 favors wide lenses.
    /// Shifts the median in log(tan(fov_y / 2)) space, without splitting FOV bands.
    pub bias: f32,
    /// Spread around the median: 0 fixes the lens, 1 spans the full configured range.
    pub spread: f32,
    /// Compatibility with archived two-band policies. None uses the continuous PDF.
    /// New configurations should use bias/spread instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ultra_wide_probability: Option<f32>,
}

impl Default for IntrinsicsSettings {
    fn default() -> Self {
        Self {
            fov_y_min_degrees: 28.,
            fov_y_max_degrees: 145.,
            bias: -0.2,
            spread: 1.,
            ultra_wide_probability: None,
        }
    }
}

impl IntrinsicsSettings {
    /// Bounds of the historical camera contract, for archived manifests only.
    pub(super) fn legacy() -> Self {
        Self {
            fov_y_min_degrees: 27.,
            fov_y_max_degrees: 108.,
            ultra_wide_probability: Some(0.),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.fov_y_min_degrees.is_finite()
            || !self.fov_y_max_degrees.is_finite()
            || self.fov_y_min_degrees < 15.
            || self.fov_y_max_degrees > MAX_VERTICAL_FOV_DEGREES
            || self.fov_y_min_degrees > self.fov_y_max_degrees
            || !(-1.0..=1.0).contains(&self.bias)
            || !(0.0..=1.0).contains(&self.spread)
            || self
                .ultra_wide_probability
                .is_some_and(|p| !(0.0..=1.0).contains(&p))
        {
            return Err("indoor_camera.intrinsics requires 15 <= fov_y_min_degrees <= fov_y_max_degrees <= 150 with bias in [-1,1], spread in [0,1] and optional archived ultra_wide_probability in [0,1]".into());
        }
        Ok(())
    }

    pub fn contains(&self, degrees: f32) -> bool {
        (self.fov_y_min_degrees..=self.fov_y_max_degrees).contains(&degrees)
            && (self.ultra_wide_probability.is_some()
                || self.spread != 0.
                || (degrees - self.quantile(0.5)).abs() <= 1e-4)
    }

    /// Quantile of the proposal distribution (before visibility/clearance rejection).
    /// Inverse-CDF sampling in log focal space is bounded, continuous and allocation-free.
    /// `q` must be in [0,1]. Degenerate bounds or zero spread give a fixed lens.
    pub fn quantile(&self, q: f32) -> f32 {
        assert!((0.0..=1.0).contains(&q), "lens quantile must be in [0,1]");
        let lo = self.fov_y_min_degrees;
        let hi = self.fov_y_max_degrees;
        if lo == hi {
            return lo;
        }
        if let Some(p) = self.ultra_wide_probability {
            if lo < ULTRA_WIDE_START_DEGREES && hi > ULTRA_WIDE_START_DEGREES {
                return if p < 1. && q <= 1. - p {
                    interpolate_lens(lo, ULTRA_WIDE_START_DEGREES, q / (1. - p))
                } else {
                    interpolate_lens(ULTRA_WIDE_START_DEGREES, hi, (q - (1. - p)) / p)
                };
            }
            return interpolate_lens(lo, hi, q);
        }
        let center = 0.5 + 0.49 * self.bias as f64;
        let t = if self.spread == 0. {
            center
        } else if q == 0. || q == 1. {
            q as f64
        } else {
            let logit = (center / (1. - center)).ln()
                + self.spread as f64 * ((q as f64) / (1. - q as f64)).ln();
            1. / (1. + (-logit).exp())
        };
        interpolate_lens(lo, hi, t as f32)
    }

    pub(super) fn sample(&self, rng: &mut impl Rng) -> f32 {
        let (mut lo, mut hi) = (self.fov_y_min_degrees, self.fov_y_max_degrees);
        if let Some(p) = self.ultra_wide_probability {
            // Preserve the exact RNG sequence of explicitly archived policies.
            if lo < ULTRA_WIDE_START_DEGREES && hi > ULTRA_WIDE_START_DEGREES {
                if rng.random_bool(p as f64) {
                    lo = ULTRA_WIDE_START_DEGREES;
                } else {
                    hi = ULTRA_WIDE_START_DEGREES;
                }
            }
            if lo == hi {
                return lo;
            }
            let log_tan = rng.random_range(
                (lo.to_radians() * 0.5).tan().ln()..(hi.to_radians() * 0.5).tan().ln(),
            );
            return (2. * log_tan.exp().atan().to_degrees()).clamp(lo, hi);
        }
        if lo == hi || self.spread == 0. {
            return self.quantile(0.5);
        }
        self.quantile(rng.random())
    }

    /// Rare coverage recovery stays inside the explicitly requested lens range.
    pub(super) fn recovery(&self, preferred: f32) -> f32 {
        if self.ultra_wide_probability.is_none() && self.spread == 0. {
            self.quantile(0.5)
        } else {
            preferred.clamp(self.fov_y_min_degrees, self.fov_y_max_degrees)
        }
    }
}

fn interpolate_lens(lo: f32, hi: f32, t: f32) -> f32 {
    let log_lo = (lo.to_radians() * 0.5).tan().ln();
    let log_hi = (hi.to_radians() * 0.5).tan().ln();
    (2. * (log_lo + t * (log_hi - log_lo)).exp().atan().to_degrees()).clamp(lo, hi)
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
    fn smooth_pdf_quantiles_bias_and_spread_match_samples() {
        let mut rng = stream(17, 793);
        for bias in [-0.8, -0.2, 0., 0.8] {
            for spread in [0., 0.3, 1.] {
                let p = IntrinsicsSettings {
                    bias,
                    spread,
                    ..default()
                };
                p.validate().unwrap();
                let mut samples: Vec<_> = (0..10000).map(|_| p.sample(&mut rng)).collect();
                samples.sort_by(f32::total_cmp);
                for q in [0.1, 0.5, 0.9] {
                    assert!((samples[(q * 9999.) as usize] - p.quantile(q)).abs() < 3.);
                }
                assert!(samples.iter().all(|&fov| p.contains(fov)));
                if spread == 0. {
                    assert!(samples.iter().all(|&fov| fov == p.quantile(0.5)));
                }
                let mut previous = 0.;
                for step in 0..=1000 {
                    let fov = p.quantile(step as f32 / 1000.);
                    assert!(fov >= previous);
                    previous = fov;
                }
            }
        }
        let middle = IntrinsicsSettings::default();
        let narrow = IntrinsicsSettings {
            bias: -0.8,
            ..middle
        };
        let wide = IntrinsicsSettings {
            bias: 0.8,
            ..middle
        };
        assert!(narrow.quantile(0.5) < middle.quantile(0.5));
        assert!(wide.quantile(0.5) > middle.quantile(0.5));
        let concentrated = IntrinsicsSettings {
            spread: 0.2,
            ..middle
        };
        assert!(
            concentrated.quantile(0.9) - concentrated.quantile(0.1)
                < middle.quantile(0.9) - middle.quantile(0.1)
        );
        for json in [r#"{"bias":1.1}"#, r#"{"spread":-0.1}"#, r#"{"spread":1.1}"#] {
            assert!(serde_json::from_str::<IntrinsicsSettings>(json)
                .unwrap()
                .validate()
                .is_err());
        }
        let archived: IntrinsicsSettings = serde_json::from_str(
            r#"{"fov_y_min_degrees":28,"fov_y_max_degrees":145,"ultra_wide_probability":0.25}"#,
        )
        .unwrap();
        assert_eq!(archived.ultra_wide_probability, Some(0.25));
        assert_eq!(
            serde_json::from_str::<IntrinsicsSettings>(&serde_json::to_string(&archived).unwrap())
                .unwrap(),
            archived
        );
        assert!(serde_json::to_value(middle)
            .unwrap()
            .get("ultra_wide_probability")
            .is_none());
    }

    #[test]
    fn continuous_lens_mixture_has_wide_support_and_replays() {
        let policy = IntrinsicsSettings {
            ultra_wide_probability: Some(0.25),
            ..default()
        };
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
                ultra_wide_probability: Some(probability),
                ..default()
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
