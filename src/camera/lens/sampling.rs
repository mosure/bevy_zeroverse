use super::*;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A seedable per-camera prior, independent of geometry and placement retries.
/// Missing settings or probability zero retain the pinhole generator exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LensSampling {
    pub probability: f32,
    pub fov_degrees: [f32; 2],
    pub circular_probability: f32,
    /// Absolute probability among fisheye cameras. The remaining mass is
    /// full-frame (diagonal); circular + elliptical must not exceed one.
    pub elliptical_probability: f32,
    /// Blend from equidistant to equisolid-like angular compression, plus
    /// bounded independent low-amplitude polynomial manufacturing variation.
    pub distortion_strength: f32,
    pub anisotropy: [f32; 2],
}
impl Default for LensSampling {
    fn default() -> Self {
        Self {
            probability: 0.25,
            fov_degrees: [140., 210.],
            circular_probability: 0.2,
            elliptical_probability: 0.,
            distortion_strength: 1.,
            anisotropy: [0.97, 1.03],
        }
    }
}
impl LensSampling {
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.probability,
            self.circular_probability,
            self.elliptical_probability,
            self.distortion_strength,
        ]
        .iter()
        .any(|v| !(0.0..=1.).contains(v))
            || self.circular_probability + self.elliptical_probability > 1.
            || self.fov_degrees.iter().any(|v| !(40.0..=220.).contains(v))
            || self.fov_degrees[0] > self.fov_degrees[1]
            || self.anisotropy.iter().any(|v| !(0.8..=1.25).contains(v))
            || self.anisotropy[0] > self.anisotropy[1]
        {
            return Err("lenses: probabilities/strength in [0,1], circular + elliptical <= 1, FOV ordered in [40,220], anisotropy ordered in [0.8,1.25] required".into());
        }
        Ok(())
    }
    pub fn sample(&self, seed: u64, camera_index: usize) -> Option<FisheyeSensor> {
        if self.probability == 0. {
            return None;
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
            seed ^ 0x4c45_4e53_4649_5348
                ^ (camera_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        if rng.random::<f32>() >= self.probability {
            return None;
        }
        let strength = self.distortion_strength;
        let lens = Fisheye {
            max_angle: rng
                .random_range(self.fov_degrees[0]..=self.fov_degrees[1])
                .to_radians()
                * 0.5,
            coefficients: [
                strength * rng.random_range(-0.041..=0.02),
                strength * rng.random_range(0.0001..=0.003),
                strength * rng.random_range(-0.00002..=0.00002),
                strength * rng.random_range(0.0..=0.000002),
            ],
        };
        // This bounded coefficient prior is monotonic throughout the maximum
        // domain. Validation covers custom calibrations without silent fallback.
        debug_assert!(lens.validate().is_ok());
        let fit_sample = rng.random::<f32>();
        let fit = if fit_sample < self.circular_probability {
            FisheyeFit::Circular
        } else if fit_sample < self.circular_probability + self.elliptical_probability {
            FisheyeFit::Elliptical
        } else {
            FisheyeFit::Diagonal
        };
        let sampled_anisotropy = rng.random_range(self.anisotropy[0]..=self.anisotropy[1]);
        Some(FisheyeSensor {
            lens,
            fit,
            anisotropy: if fit == FisheyeFit::Circular {
                1.
            } else {
                sampled_anisotropy
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampling_is_bounded_seeded_and_frequency_is_not_count_rounded() {
        assert_eq!(
            serde_json::from_str::<FisheyeFit>("\"full_frame\"").unwrap(),
            FisheyeFit::Diagonal
        );
        let p = LensSampling::default();
        p.validate().unwrap();
        let mut count = 0;
        let mut circular = 0;
        for seed in 0..4096 {
            for slot in 0..4 {
                assert_eq!(p.sample(seed, slot), p.sample(seed, slot));
                if let Some(lens) = p.sample(seed, slot) {
                    count += 1;
                    circular += usize::from(lens.fit == FisheyeFit::Circular);
                    lens.calibration([640, 480], [0.1, -0.05]).unwrap();
                }
            }
        }
        assert!((3800..4400).contains(&count));
        assert!((0.17..0.23).contains(&(circular as f32 / count as f32)));
        let off = LensSampling {
            probability: 0.,
            ..p
        };
        assert!(off.sample(1, 0).is_none());
    }
}
