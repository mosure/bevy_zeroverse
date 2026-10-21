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
    /// Continuous diameter-FOV preference: -1 narrow, +1 wide. Zero is neutral.
    pub fov_bias: f32,
    /// Angular distribution spread: zero fixes the median; one spans the bounds.
    pub fov_spread: f32,
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
            probability: 0.5,
            fov_degrees: [140., 210.],
            fov_bias: 0.,
            fov_spread: 1.,
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
            self.fov_spread,
        ]
        .iter()
        .any(|v| !(0.0..=1.).contains(v))
            || !(-1.0..=1.).contains(&self.fov_bias)
            || self.circular_probability + self.elliptical_probability > 1.
            || self.fov_degrees.iter().any(|v| !(40.0..=220.).contains(v))
            || self.fov_degrees[0] > self.fov_degrees[1]
            || self.anisotropy.iter().any(|v| !(0.8..=1.25).contains(v))
            || self.anisotropy[0] > self.anisotropy[1]
        {
            return Err("lenses: probabilities/strength/spread in [0,1], fov_bias in [-1,1], circular + elliptical <= 1, FOV ordered in [40,220], anisotropy ordered in [0.8,1.25] required".into());
        }
        Ok(())
    }
    /// Quantile of the diameter-FOV proposal in angular space, including >180°.
    /// Unlike a perspective prior, this must not use tan(FOV/2) across its pole.
    pub fn fov_quantile(&self, q: f32) -> f32 {
        let t = unit_quantile(q, self.fov_bias, self.fov_spread);
        let [lo, hi] = self.fov_degrees;
        (lo + t * (hi - lo)).clamp(lo, hi)
    }

    fn sample_fov(&self, rng: &mut impl Rng) -> f32 {
        // Preserve the published uniform prior's exact RNG draw and arithmetic.
        if self.fov_bias == 0. && self.fov_spread == 1. {
            rng.random_range(self.fov_degrees[0]..=self.fov_degrees[1])
        } else {
            self.fov_quantile(rng.random())
        }
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
            max_angle: self.sample_fov(&mut rng).to_radians() * 0.5,
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

/// Bounded inverse-CDF shape in unit coordinates before angular scaling.
fn unit_quantile(q: f32, bias: f32, spread: f32) -> f32 {
    assert!((0.0..=1.0).contains(&q), "lens quantile must be in [0,1]");
    let center = 0.5 + 0.49 * bias as f64;
    if spread == 0. {
        center as f32
    } else if q == 0. || q == 1. || (bias == 0. && spread == 1.) {
        q
    } else {
        let logit =
            (center / (1. - center)).ln() + spread as f64 * (q as f64 / (1. - q as f64)).ln();
        (1. / (1. + (-logit).exp())) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angular_prior_is_bounded_monotonic_and_bias_and_spread_have_physical_effects() {
        let uniform = LensSampling {
            probability: 1.,
            ..Default::default()
        };
        for bias in [-1., -0.6, 0., 0.6, 1.] {
            for spread in [0., 0.2, 0.7, 1.] {
                let p = LensSampling {
                    fov_bias: bias,
                    fov_spread: spread,
                    ..uniform.clone()
                };
                p.validate().unwrap();
                let mut previous = p.fov_degrees[0];
                for i in 0..=1024 {
                    let v = p.fov_quantile(i as f32 / 1024.);
                    assert!(v >= previous && v <= p.fov_degrees[1]);
                    previous = v;
                }
                assert!((p.fov_quantile(0.5) - (175. + 34.3 * bias)).abs() < 0.001);
                if spread == 0. {
                    for seed in 0..32 {
                        assert!(
                            (p.sample(seed, 0).unwrap().lens.max_angle.to_degrees() * 2.
                                - p.fov_quantile(0.5))
                            .abs()
                                < 0.001
                        );
                    }
                }
            }
        }
        let concentrated = LensSampling {
            fov_spread: 0.3,
            ..uniform.clone()
        };
        assert!(concentrated.fov_quantile(0.1) > uniform.fov_quantile(0.1));
        assert!(concentrated.fov_quantile(0.9) < uniform.fov_quantile(0.9));
        for invalid in [f32::NAN, f32::INFINITY, -1.01, 1.01] {
            assert!(LensSampling {
                fov_bias: invalid,
                ..uniform.clone()
            }
            .validate()
            .is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(LensSampling {
                fov_spread: invalid,
                ..uniform.clone()
            }
            .validate()
            .is_err());
        }
    }

    #[test]
    fn default_shape_preserves_existing_uniform_sampler_draws() {
        let prior: LensSampling = serde_json::from_str(r#"{"probability":0.25}"#).unwrap();
        assert_eq!((prior.fov_bias, prior.fov_spread), (0., 1.));
        for seed in 0..128 {
            let mut old = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let mut current = old.clone();
            assert_eq!(
                prior.sample_fov(&mut current),
                old.random_range(140.0f32..=210.)
            );
            assert_eq!(current.random::<u64>(), old.random::<u64>());
        }
    }

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
        assert!((7900..8500).contains(&count));
        assert!((0.17..0.23).contains(&(circular as f32 / count as f32)));
        let off = LensSampling {
            probability: 0.,
            ..p
        };
        assert!(off.sample(1, 0).is_none());
    }
}
