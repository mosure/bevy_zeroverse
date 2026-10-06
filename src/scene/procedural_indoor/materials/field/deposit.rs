//! The original five-field deposit, with caller-budgeted per-map hash lattices.
use super::PreparedPeriodic;

pub(in super::super) struct PreparedDeposit {
    warp: f32,
    fields: [PreparedPeriodic; 5],
}

impl PreparedDeposit {
    pub(in super::super) fn new(
        count: [u32; 2],
        warp: f32,
        seed: u64,
        remaining: &mut usize,
    ) -> Self {
        Self {
            warp,
            fields: [
                PreparedPeriodic::new([3, 5], seed, remaining),
                PreparedPeriodic::new([5, 3], seed.wrapping_add(17), remaining),
                PreparedPeriodic::new(count, seed.wrapping_add(47), remaining),
                PreparedPeriodic::new(count.map(|n| n * 2), seed.wrapping_add(61), remaining),
                PreparedPeriodic::new(count.map(|n| n * 4), seed.wrapping_add(83), remaining),
            ],
        }
    }
    #[cfg(test)]
    pub(in super::super) fn bytes(&self) -> usize {
        self.fields.iter().map(PreparedPeriodic::bytes).sum()
    }
    pub(in super::super) fn sample(&self, [u, v]: [f32; 2]) -> f32 {
        let x = u + (self.fields[0].sample(u, v) - 0.5) * self.warp;
        let y = v + (self.fields[1].sample(u, v) - 0.5) * self.warp;
        0.64 * self.fields[2].sample(x, y)
            + 0.25 * self.fields[3].sample(x, y)
            + 0.11 * self.fields[4].sample(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_deposit_preserves_scalar_bits_and_shared_budget() {
        for count in [[1, 1], [3, 5], [16, 16], [192, 192]] {
            for seed in [0, 202, 43_084_482, u64::MAX] {
                for budget in [0, 32, 256 * 1024] {
                    let mut remaining = budget;
                    let cached = PreparedDeposit::new(count, 0.17, seed, &mut remaining);
                    assert!(remaining <= budget);
                    assert_eq!(
                        cached
                            .fields
                            .iter()
                            .map(PreparedPeriodic::bytes)
                            .sum::<usize>()
                            + remaining,
                        budget,
                    );
                    let uv = (0..32)
                        .flat_map(|y| (0..32).map(move |x| [x as f32 / 32., y as f32 / 32.]));
                    for uv in uv.chain([[-0.25, 1.125], [1. - f32::EPSILON, 1.], [2.3, -0.4]]) {
                        assert_eq!(
                            super::super::deposit(uv, count, 0.17, seed).to_bits(),
                            cached.sample(uv).to_bits(),
                            "count {count:?} seed {seed} budget {budget} uv {uv:?}"
                        );
                    }
                }
            }
        }
    }
}
