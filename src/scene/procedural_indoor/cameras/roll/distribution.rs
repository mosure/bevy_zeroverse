//! Signed sensor roll in degrees. Both components are symmetric about gravity.
use super::super::motion::slot_unit;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RollDistribution {
    /// Largest absolute sampled roll. Zero levels every room-camera mount.
    pub max_degrees: f32,
    /// Common triangular component's half-width, as a fraction of the maximum.
    pub spread: f32,
    /// Probability of using the uniform, full-range component.
    pub wide_probability: f32,
}

impl Default for RollDistribution {
    fn default() -> Self {
        Self {
            max_degrees: 45.,
            spread: 0.24,
            wide_probability: 0.12,
        }
    }
}

impl RollDistribution {
    pub const fn level() -> Self {
        Self {
            max_degrees: 0.,
            spread: 0.,
            wide_probability: 0.,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=45.0).contains(&self.max_degrees)
            || !(0.0..=1.0).contains(&self.spread)
            || !(0.0..=1.0).contains(&self.wide_probability)
        {
            return Err("indoor_camera.roll requires max_degrees in [0,45] and spread and wide_probability in [0,1]".into());
        }
        Ok(())
    }

    pub fn sample_degrees(&self, seed: u64, camera_index: usize) -> f32 {
        let draw = |domain| slot_unit(seed, camera_index, domain) as f32;
        let u = draw(0x524f_4f4d_524f_4c4c);
        if slot_unit(seed, camera_index, 0x4241_4e4b_5749_4445) < self.wide_probability as f64 {
            (2. * u - 1.) * self.max_degrees
        } else {
            let v = draw(0x4241_4e4b_434f_5245);
            (u - v) * self.max_degrees * self.spread
        }
    }

    /// Signed quantile of the same mixture used by the generator, including a
    /// possible point mass at zero. Use the analytic inverse for Studio's plot.
    pub fn quantile(&self, q: f32) -> f32 {
        let signed = 2. * q.clamp(0., 1.) - 1.;
        let rank = signed.abs();
        let wide = self.wide_probability;
        let width = self.spread;
        // For normalized absolute x <= width the CDF is
        // wide*x + (1-wide)*(2*x/width - (x/width)^2).
        let at_edge = 1. - wide + wide * width;
        let x = if rank > at_edge && wide > 0. {
            (rank - (1. - wide)) / wide
        } else if width == 0. || rank == 0. {
            0.
        } else {
            let b = wide * width + 2. * (1. - wide);
            let discriminant = (b * b - 4. * (1. - wide) * rank).max(0.);
            width * (2. * rank) / (b + discriminant.sqrt())
        };
        signed.signum() * self.max_degrees * x.clamp(0., 1.)
    }

    pub fn support_degrees(&self) -> f32 {
        self.max_degrees
            * if self.wide_probability > 0. {
                1.
            } else {
                self.spread
            }
    }

    pub(super) fn drift(&self) -> f32 {
        (0.25 * self.max_degrees * self.spread)
            .min(1.5)
            .to_radians()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_sampler_matches_quantiles_and_has_two_long_tails() {
        for policy in [
            RollDistribution::default(),
            RollDistribution {
                max_degrees: 15.,
                spread: 0.3,
                wide_probability: 0.,
            },
            RollDistribution {
                wide_probability: 1.,
                ..RollDistribution::default()
            },
            RollDistribution {
                spread: 0.,
                ..RollDistribution::default()
            },
            RollDistribution::level(),
        ] {
            let mut sample: Vec<_> = (0..8192)
                .flat_map(|seed| (0..8).map(move |slot| policy.sample_degrees(seed, slot)))
                .collect();
            assert!(sample.iter().all(|r| r.abs() <= policy.support_degrees()));
            assert!(
                (sample.iter().map(|x| *x as f64).sum::<f64>() / sample.len() as f64).abs() < 0.16
            );
            sample.sort_by(f32::total_cmp);
            for q in [
                0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
            ] {
                let observed = sample[(q * (sample.len() - 1) as f32) as usize];
                let expected = policy.quantile(q);
                assert!(
                    (observed - expected).abs() < 1.1,
                    "{policy:?} q={q}: {observed} vs {expected}"
                );
                assert!((expected + policy.quantile(1. - q)).abs() < 1e-4);
            }
            if policy == RollDistribution::default() {
                for sign in [-1., 1.] {
                    let tail = sample.iter().filter(|r| **r * sign > 40.).count();
                    assert!((300..570).contains(&tail), "unexpected tail {tail}/65536");
                    assert!(sample.iter().any(|r| *r * sign > 44.8));
                }
            }
        }
        for (spread, wide_probability) in [(-0.1, 1.), (1., 1.01), (f32::NAN, 0.)] {
            assert!(RollDistribution {
                spread,
                wide_probability,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
    }
}
