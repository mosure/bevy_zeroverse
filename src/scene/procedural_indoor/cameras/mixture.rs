//! Requested reference-pair strata are drawn once, before geometry rejection.
//! Rendered overlap is measured separately; no model score selects captures.
use super::{
    multiview::{CameraPairOverlap, MultiViewSettings},
    IndoorManifest,
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlapClass {
    High,
    Low,
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OverlapMixture {
    /// Nonnegative weights for high, low and no proxy overlap. Need not sum to one.
    pub weights: [f32; 3],
    pub high: [f32; 2],
    pub low: [f32; 2],
}
impl Default for OverlapMixture {
    fn default() -> Self {
        Self {
            weights: [0.6, 0.3, 0.1],
            high: [0.45, 1.0],
            low: [0.02, 0.30],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairTarget {
    pub reference: usize,
    pub camera: usize,
    pub class: OverlapClass,
    pub proxy_overlap_range: [f32; 2],
}
impl OverlapMixture {
    pub fn validate(&self) -> Result<(), String> {
        let sum: f32 = self.weights.iter().sum();
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.)
            || !sum.is_finite()
            || sum <= 0.
            || [self.high, self.low]
                .iter()
                .any(|[lo, hi]| !(0.0..=1.0).contains(lo) || !(0.0..=1.0).contains(hi) || lo > hi)
            || self.low[0] <= 0.
            || self.low[1] >= self.high[0]
        {
            return Err("overlap_mixture requires finite nonnegative weights with positive sum and disjoint 0 < low <= 1, low.max < high.min <= high.max <= 1".into());
        }
        Ok(())
    }
    pub fn targets(&self, seed: u64, cameras: usize) -> Vec<PairTarget> {
        let mut rng = stream(seed, 346);
        (1..cameras)
            .map(|camera| {
                let choice = rng.random_range(0.0..self.weights.iter().sum::<f32>());
                let (class, range) = if choice < self.weights[0] {
                    (OverlapClass::High, self.high)
                } else if choice < self.weights[0] + self.weights[1] {
                    (OverlapClass::Low, self.low)
                } else {
                    (OverlapClass::None, [0., 0.])
                };
                PairTarget {
                    reference: 0,
                    camera,
                    class,
                    proxy_overlap_range: range,
                }
            })
            .collect()
    }
}
impl PairTarget {
    pub fn accepts(
        &self,
        policy: &MultiViewSettings,
        samples: &[super::multiview::OverlapSample],
    ) -> bool {
        let mut effective = policy.clone();
        effective.min_overlap = self.proxy_overlap_range[0];
        effective.accepts(samples)
            && samples.iter().all(|s| {
                s.reference_to_view.max(s.view_to_reference) <= self.proxy_overlap_range[1]
            })
    }
}
impl IndoorManifest {
    pub fn camera_pair_targets(&self) -> Vec<PairTarget> {
        self.camera_settings
            .overlap_mixture
            .as_ref()
            .map_or_else(Vec::new, |m| m.targets(self.seed, self.cameras.len()))
    }
    pub fn accepts_camera_pair(&self, pair: &CameraPairOverlap) -> bool {
        let Some(policy) = &self.camera_settings.multiview else {
            return true;
        };
        self.camera_pair_targets()
            .iter()
            .find(|t| t.camera == pair.camera)
            .map_or_else(
                || policy.accepts(&pair.samples),
                |t| t.accepts(policy, &pair.samples),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        cameras::CameraSettings, layout::IndoorLayout, validation::validate_layout,
    };
    #[test]
    fn fixed_strata_are_seeded_before_placement_and_preserved_in_archives() {
        let mixture = OverlapMixture::default();
        let mut counts = [0; 3];
        for seed in 0..1000 {
            let targets = mixture.targets(seed, 4);
            assert_eq!(
                serde_json::to_value(&targets).unwrap(),
                serde_json::to_value(mixture.targets(seed, 4)).unwrap()
            );
            for t in targets {
                counts[match t.class {
                    OverlapClass::High => 0,
                    OverlapClass::Low => 1,
                    OverlapClass::None => 2,
                }] += 1;
            }
        }
        for (observed, weight) in counts.into_iter().zip(mixture.weights) {
            assert!((observed as f32 / 3000. - weight).abs() < 0.04);
        }
        for weights in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] {
            for seed in 0..8 {
                let mut scene =
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                        .unwrap();
                let settings = CameraSettings {
                    overlap_mixture: Some(OverlapMixture {
                        weights,
                        ..Default::default()
                    }),
                    path_length_min: 0.,
                    path_length_max: 0.6,
                    long_path_fraction: 0.,
                    ..Default::default()
                };
                scene.resample_cameras(2, settings.clone(), 1.6).unwrap();
                validate_layout(&scene).unwrap();
                assert!(scene
                    .camera_overlap()
                    .iter()
                    .all(|p| scene.accepts_camera_pair(p)));
                let archive: IndoorManifest =
                    serde_json::from_value(serde_json::to_value(&scene).unwrap()).unwrap();
                assert_eq!(archive.camera_settings, settings);
                assert_eq!(archive.cameras, scene.cameras);
            }
        }
    }
}
