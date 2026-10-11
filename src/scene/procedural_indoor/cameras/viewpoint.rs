//! Continuous, room-seeded priors for low, seated, standing and elevated views.
//!
//! These are proposal distributions, not acceptance rules. Every lens/path still
//! passes the same primary-room, collision, semantic-coverage and multi-view
//! tests. Heights are measured above the actual supporting floor; truncated
//! logistic densities avoid piles of samples clamped onto a ceiling.
use super::super::layout::stream;
use rand::Rng;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ViewpointProgram {
    /// Continuous mixture weights, ordered low/seated/standing/elevated.
    pub weights: [f32; 4],
    pub centers_metres: [f32; 4],
    /// Logistic scale; standard deviation is scale * pi / sqrt(3).
    pub scales_metres: [f32; 4],
}
impl ViewpointProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x56494557504f494e);
        let mut weights = [
            // Low viewpoints need meaningful proposal mass: furnishing and
            // content-framing gates reject many of them. A thin low tail in a
            // standing-height prior effectively disappears after acceptance.
            rng.random_range(0.16..0.34),
            rng.random_range(0.12..0.35),
            rng.random_range(0.30..0.64),
            rng.random_range(0.05..0.30),
        ];
        let total: f32 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        Self {
            weights,
            centers_metres: [
                rng.random_range(0.76..1.00),
                rng.random_range(1.12..1.42),
                rng.random_range(1.47..1.82),
                rng.random_range(1.91..2.85),
            ],
            scales_metres: [
                rng.random_range(0.045..0.14),
                rng.random_range(0.08..0.21),
                rng.random_range(0.08..0.22),
                rng.random_range(0.17..0.46),
            ],
        }
    }
    pub(super) fn height(&self, floor: f32, upper: f32, rng: &mut impl Rng) -> Option<f32> {
        if !floor.is_finite() || !upper.is_finite() {
            return None;
        }
        let low = floor + 0.78;
        let upper = upper.min(floor + 3.25);
        if upper <= low {
            return None;
        }
        let mut draw = rng.random::<f32>();
        let mut component = self.weights.len() - 1;
        for (i, weight) in self.weights.iter().enumerate() {
            if draw < *weight {
                component = i;
                break;
            }
            draw -= weight;
        }
        let center = self.centers_metres[component];
        let scale = self.scales_metres[component];
        let cdf = |height: f32| 1. / (1. + (-(height - center) / scale).exp());
        let a = cdf(low - floor);
        let b = cdf(upper - floor);
        // A component can be far above a low slanted ceiling. Its truncated
        // tail still has a continuous density over all usable eye heights.
        let q = (a + (b - a) * rng.random::<f32>()).clamp(f32::EPSILON, 1. - f32::EPSILON);
        Some((floor + center + scale * (q / (1. - q)).ln()).clamp(low, upper))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn room_priors_have_continuous_independent_parameters_and_replay() {
        let mut parameters = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let p = ViewpointProgram::sample(seed);
            assert_eq!(p, ViewpointProgram::sample(seed));
            assert!((p.weights.iter().sum::<f32>() - 1.).abs() < 0.000001);
            assert!(p.weights.iter().all(|w| *w > 0.));
            parameters.insert(p.weights.map(f32::to_bits));
        }
        assert_eq!(parameters.len(), 128);
    }
    #[test]
    fn height_proposals_follow_the_supporting_floor_and_have_no_band_rails() {
        let p = ViewpointProgram::sample(204);
        for (floor, ceiling) in [(0., 3.25), (1.2, 3.25), (-0.25, 1.3)] {
            let mut rng = stream(900, 17);
            let mut replay_rng = rng.clone();
            let mut quantized = std::collections::BTreeSet::new();
            let mut at_boundary = 0;
            for _ in 0..1024 {
                let height = p.height(floor, ceiling, &mut rng).unwrap();
                assert_eq!(height, p.height(floor, ceiling, &mut replay_rng).unwrap());
                assert!((floor + 0.78..=ceiling.min(floor + 3.25)).contains(&height));
                quantized.insert((height * 10000.).round() as i32);
                if height == floor + 0.78 || height == ceiling.min(floor + 3.25) {
                    at_boundary += 1;
                }
            }
            assert!(quantized.len() > 800);
            assert!(
                at_boundary < 3,
                "density should not concentrate at clearance boundaries"
            );
        }
        assert!(p.height(1.0, 1.75, &mut stream(0, 0)).is_none());
        assert!(p.height(f32::NAN, 3.25, &mut stream(0, 0)).is_none());
        assert!(p.height(0., f32::NAN, &mut stream(0, 0)).is_none());
    }

    #[test]
    fn continuous_mixture_keeps_low_and_elevated_tail_mass() {
        let mut low = 0;
        let mut standing = 0;
        let mut elevated = 0;
        for seed in 0..128 {
            let p = ViewpointProgram::sample(seed);
            let mut rng = stream(seed, 31);
            for _ in 0..64 {
                let height = p.height(0., 3.25, &mut rng).unwrap();
                low += usize::from(height < 0.85);
                standing += usize::from((1.4..2.0).contains(&height));
                elevated += usize::from(height > 2.1);
            }
        }
        // Proposal support must survive practical rejection sampling. Realized
        // scene coverage is checked separately after all placement/view gates.
        assert!(low > 256, "low-view proposal tail disappeared: {low}/8192");
        assert!(
            standing > 1600,
            "ordinary eye heights disappeared: {standing}/8192"
        );
        assert!(
            elevated > 256,
            "elevated-view proposal tail disappeared: {elevated}/8192"
        );
    }
}
