//! Connected multi-view sets for reconstruction: camera zero is the reference.
//! Visibility uses first-surface proxy ray casts. It is a proposal constraint,
//! not a guarantee about rendered pixels; the depth audit measures actual overlap.
use super::coverage::{Coverage, VisibilityView};
use crate::scene::procedural_indoor::layout::{stream, IndoorCamera, IndoorManifest};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const CHECK_TIMES: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
const GROUP_ATTEMPTS: usize = 8;
const VIEW_ATTEMPTS: usize = 384;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MultiViewSettings {
    /// Minimum fraction in BOTH directions, at every CHECK_TIMES sample.
    pub min_overlap: f32,
    /// Euclidean reference-to-view baseline in metres, at every time sample.
    pub min_baseline: f32,
    pub max_baseline: f32,
}
impl Default for MultiViewSettings {
    fn default() -> Self {
        Self {
            min_overlap: 0.35,
            min_baseline: 0.25,
            max_baseline: 3.0,
        }
    }
}
impl MultiViewSettings {
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_overlap.is_finite()
            || !(0.0..=1.0).contains(&self.min_overlap)
            || !self.min_baseline.is_finite()
            || !self.max_baseline.is_finite()
            || self.min_baseline <= 0.0
            || self.max_baseline < self.min_baseline
            || self.max_baseline > 100.0
        {
            return Err("indoor_camera.multiview requires min_overlap in [0,1] and 0 < min_baseline <= max_baseline <= 100 metres".into());
        }
        Ok(())
    }
    pub fn accepts(&self, samples: &[OverlapSample]) -> bool {
        samples.iter().all(|s| {
            s.reference_to_view.min(s.view_to_reference) >= self.min_overlap
                && s.baseline_m + 1e-5 >= self.min_baseline
                && s.baseline_m <= self.max_baseline + 1e-5
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapSample {
    pub time: f32,
    pub reference_to_view: f32,
    pub view_to_reference: f32,
    pub baseline_m: f32,
    pub mean_triangulation_degrees: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPairOverlap {
    pub reference: usize,
    pub camera: usize,
    pub samples: Vec<OverlapSample>,
}

fn pair_samples(
    coverage: &Coverage,
    reference: &IndoorCamera,
    reference_views: &[VisibilityView],
    camera: &IndoorCamera,
    aspect: f32,
) -> Vec<OverlapSample> {
    CHECK_TIMES
        .into_iter()
        .zip(reference_views)
        .map(|(time, a)| {
            let b = coverage.view(camera, aspect, time);
            let (ab, angle_a) = coverage.shared(a, &b);
            let (ba, angle_b) = coverage.shared(&b, a);
            OverlapSample {
                time,
                reference_to_view: ab,
                view_to_reference: ba,
                baseline_m: reference
                    .transform_at(time)
                    .translation
                    .distance(camera.transform_at(time).translation),
                mean_triangulation_degrees: (angle_a + angle_b) * 0.5,
            }
        })
        .collect()
}

impl IndoorManifest {
    /// O(cameras) reference edges, including when the placement policy is off.
    pub fn camera_overlap(&self) -> Vec<CameraPairOverlap> {
        let Some(reference) = self.cameras.first() else {
            return Vec::new();
        };
        if self.cameras.len() < 2 {
            return Vec::new();
        }
        let coverage = Coverage::new(self);
        let reference_views: Vec<_> = CHECK_TIMES
            .into_iter()
            .map(|t| coverage.view(reference, self.camera_aspect_ratio, t))
            .collect();
        self.cameras
            .iter()
            .enumerate()
            .skip(1)
            .map(|(camera, view)| CameraPairOverlap {
                reference: 0,
                camera,
                samples: pair_samples(
                    &coverage,
                    reference,
                    &reference_views,
                    view,
                    self.camera_aspect_ratio,
                ),
            })
            .collect()
    }

    pub(crate) fn sample_cameras(&mut self, count: usize) -> Result<(), String> {
        self.camera_settings.validate()?;
        let mut rng = stream(self.seed, 3);
        let coverage = Coverage::new(self);
        self.cameras.clear();
        let Some(policy) = self.camera_settings.multiview.clone().filter(|_| count > 1) else {
            return self.sample_independent_cameras(count, &mut rng, &coverage);
        };
        // Retry the whole set if an anchor lies in a cramped cul-de-sac. The
        // budget is bounded and thresholds are never silently relaxed.
        for _ in 0..GROUP_ATTEMPTS {
            self.cameras.clear();
            self.sample_independent_cameras(1, &mut rng, &coverage)?;
            let reference = self.cameras[0].clone();
            let views: Vec<_> = CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(&reference, self.camera_aspect_ratio, t))
                .collect();
            for _ in 1..count {
                let mut found = None;
                for _ in 0..VIEW_ATTEMPTS {
                    let candidate = propose(&reference, &policy, &mut rng);
                    // Cheap rejection before casts: swept collision checks still
                    // cover the complete paths, including route segments/curves.
                    if !self.camera_clear(candidate.start)
                        || self.cameras.iter().skip(1).any(|c| {
                            c.start.distance(candidate.start) < policy.min_baseline.min(0.18)
                        })
                        || candidate.path_length() + 1e-4 < self.camera_settings.path_length_min
                        || candidate.path_length() > self.camera_settings.path_length_max + 1e-4
                        || !self.camera_curve_clear(&candidate)
                        || !coverage.suitable(&candidate, false)
                    {
                        continue;
                    }
                    let samples = pair_samples(
                        &coverage,
                        &reference,
                        &views,
                        &candidate,
                        self.camera_aspect_ratio,
                    );
                    if policy.accepts(&samples) {
                        found = Some(candidate);
                        break;
                    }
                }
                if let Some(camera) = found {
                    self.cameras.push(camera);
                } else {
                    break;
                }
            }
            if self.cameras.len() == count {
                return Ok(());
            }
        }
        self.cameras.clear();
        Err(format!("seed {}: unable to sample {count} connected multi-view cameras after {GROUP_ATTEMPTS} anchors x {VIEW_ATTEMPTS} proposals/view; min_overlap={}, baseline={}..{}m. Lower overlap/baseline bounds or shorten paths; no unconstrained fallback was used", self.seed, policy.min_overlap, policy.min_baseline, policy.max_baseline))
    }
}

fn propose(
    reference: &IndoorCamera,
    policy: &MultiViewSettings,
    rng: &mut impl Rng,
) -> IndoorCamera {
    let radius = rng
        .random_range(policy.min_baseline.ln()..=policy.max_baseline.ln())
        .exp();
    let azimuth = rng.random_range(0.0..std::f32::consts::TAU);
    let vertical: f32 = rng.random_range(-0.35..0.35);
    let planar = (1.0 - vertical * vertical).sqrt();
    let offset = Vec3::new(planar * azimuth.cos(), vertical, planar * azimuth.sin()) * radius;
    let jitter = Vec3::new(
        rng.random_range(-0.35..0.35),
        rng.random_range(-0.15..0.15),
        rng.random_range(-0.35..0.35),
    );
    let mut camera = reference.clone();
    camera.start += offset;
    camera.end += offset;
    camera.target += jitter;
    // Intrinsics remain independently sampled; overlap rejects extreme mismatches.
    camera.fov_degrees = (0.5 / rng.random_range(0.37_f32.ln()..2.0_f32.ln()).exp())
        .atan()
        .to_degrees()
        * 2.0;
    if let Some(motion) = &mut camera.motion {
        for p in &mut motion.control {
            *p += offset;
        }
        for p in &mut motion.route {
            *p += offset;
        }
        motion.target_end += jitter;
        motion.roll = [rng.random_range(-0.09..0.09), rng.random_range(-0.09..0.09)];
    }
    camera
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::CameraSettings, layout::IndoorLayout, validation::validate_layout,
    };

    #[test]
    fn policy_validation_and_legacy_json() {
        assert!(CameraSettings::parse("{}").unwrap().multiview.is_some());
        assert!(CameraSettings::parse(r#"{"multiview":null}"#)
            .unwrap()
            .multiview
            .is_none());
        for json in [
            r#"{"multiview":{"min_overlap":1.01}}"#,
            r#"{"multiview":{"min_overlap":-0.1}}"#,
            r#"{"multiview":{"min_baseline":0}}"#,
            r#"{"multiview":{"min_baseline":3,"max_baseline":2}}"#,
            r#"{"multiview":{"minimum_overlap":0.3}}"#,
        ] {
            assert!(CameraSettings::parse(json).is_err(), "{json}");
        }
        let policy = CameraSettings::parse(r#"{"multiview":{}}"#).unwrap();
        assert_eq!(policy.multiview.unwrap(), MultiViewSettings::default());
    }

    #[test]
    fn connected_sets_preserve_layout_intrinsics_and_trajectories() {
        let mut translated = 0;
        for seed in 0..32 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25)
                    .unwrap();
            let before = scene.clone();
            let settings = CameraSettings {
                multiview: Some(MultiViewSettings::default()),
                ..default()
            };
            let count = if seed < 8 { 4 } else { 2 };
            let aspect = if seed % 2 == 0 { 4.0 / 3.0 } else { 0.75 };
            scene
                .resample_cameras(count, settings.clone(), aspect)
                .unwrap();
            validate_layout(&scene).unwrap();
            assert_eq!(scene.objects, before.objects);
            assert_eq!(scene.humans, before.humans);
            for pair in scene.camera_overlap() {
                assert!(settings.multiview.as_ref().unwrap().accepts(&pair.samples));
                assert_eq!(pair.reference, 0);
            }
            translated += usize::from(scene.cameras[0].path_length() > 0.1);
            let cameras = scene.cameras.clone();
            scene.resample_cameras(count, settings, aspect).unwrap();
            assert_eq!(scene.cameras, cameras);
            let restored: IndoorManifest =
                serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
            assert_eq!(restored, scene);
        }
        assert!(
            translated > 8,
            "overlap must not freeze all camera trajectories"
        );
    }

    #[test]
    fn default_policy_is_deterministic_and_impossible_policy_is_atomic() {
        let mut scene =
            IndoorManifest::generate_with_humans(1, IndoorLayout::Mixed, 0.65, 2, 0.0).unwrap();
        let original = scene.cameras.clone();
        scene
            .resample_cameras(2, CameraSettings::default(), 4.0 / 3.0)
            .unwrap();
        assert_eq!(scene.cameras, original);
        let original = scene.clone();
        let settings = CameraSettings {
            multiview: Some(MultiViewSettings {
                min_baseline: 99.0,
                max_baseline: 100.0,
                ..default()
            }),
            ..default()
        };
        assert!(scene
            .resample_cameras(2, settings, 1.0)
            .unwrap_err()
            .contains("no unconstrained fallback"));
        assert_eq!(scene, original);
        assert!(scene
            .resample_cameras(257, CameraSettings::default(), 1.0)
            .is_err());
        assert!(scene
            .resample_cameras(2, CameraSettings::default(), f32::NAN)
            .is_err());
    }

    #[test]
    fn archive_without_camera_policy_retains_independent_semantics() {
        let scene =
            IndoorManifest::generate_with_humans(9, IndoorLayout::Mixed, 0.65, 2, 0.0).unwrap();
        let mut json = serde_json::to_value(scene).unwrap();
        json["camera_settings"]
            .as_object_mut()
            .unwrap()
            .remove("multiview");
        let older: IndoorManifest = serde_json::from_value(json.clone()).unwrap();
        assert!(older.camera_settings.multiview.is_none());
        json.as_object_mut().unwrap().remove("camera_settings");
        let archive: IndoorManifest = serde_json::from_value(json).unwrap();
        assert!(archive.camera_settings.multiview.is_none());
    }

    #[test]
    fn fixed_small_stereo_baselines_and_single_views_work() {
        let mut scene =
            IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
        let settings = CameraSettings {
            path_length_min: 0.0,
            path_length_max: 0.0,
            multiview: Some(MultiViewSettings {
                min_baseline: 0.1,
                max_baseline: 0.1,
                min_overlap: 0.6,
            }),
            ..default()
        };
        scene.resample_cameras(2, settings.clone(), 1.0).unwrap();
        for camera in &scene.cameras {
            assert_eq!(camera.transform_at(0.0), camera.transform_at(1.0));
        }
        for pair in scene.camera_overlap() {
            assert!(settings.multiview.as_ref().unwrap().accepts(&pair.samples));
        }
        scene.resample_cameras(1, settings, 1.0).unwrap();
        assert_eq!(scene.cameras.len(), 1);
        assert!(scene.camera_overlap().is_empty());
    }
}
