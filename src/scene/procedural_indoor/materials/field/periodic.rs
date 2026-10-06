//! Exact per-map periodic hash fields. Interpolation remains dynamic; grids
//! above 192² cells or the caller's byte budget retain the scalar hash path.
//! A wrapped border removes per-sample integer remainder from cached lookups.
use super::{hash, periodic_noise, periodic_unit};

const MAX_LATTICE_CELLS: u32 = 192 * 192;

pub(in super::super) struct PreparedPeriodic {
    count: [u32; 2],
    seed: u64,
    values: Option<Vec<f32>>,
}
impl PreparedPeriodic {
    pub(in super::super) fn new(count: [u32; 2], seed: u64, remaining: &mut usize) -> Self {
        let cells = count[0].checked_mul(count[1]);
        let padded = count[0]
            .checked_add(1)
            .zip(count[1].checked_add(1))
            .and_then(|(x, y)| (x as usize).checked_mul(y as usize));
        let bytes = padded.and_then(|n| n.checked_mul(std::mem::size_of::<f32>()));
        let values = match (cells, padded, bytes) {
            (Some(cells), Some(padded), Some(bytes))
                if count.into_iter().all(|n| n > 0)
                    && cells <= MAX_LATTICE_CELLS
                    && bytes <= *remaining =>
            {
                let mut values = Vec::with_capacity(padded);
                // Charge the allocated table capacity, including the wrapped
                // row/column, against the caller's existing shared byte budget.
                let allocated = values.capacity().checked_mul(std::mem::size_of::<f32>());
                if let Some(allocated) = allocated.filter(|bytes| *bytes <= *remaining) {
                    for y in 0..=count[1] {
                        for x in 0..=count[0] {
                            values.push(hash(x % count[0], y % count[1], seed));
                        }
                    }
                    *remaining -= allocated;
                    Some(values)
                } else {
                    None
                }
            }
            _ => None,
        };
        Self {
            count,
            seed,
            values,
        }
    }
    #[cfg(test)]
    pub(in super::super) fn bytes(&self) -> usize {
        self.values
            .as_ref()
            .map_or(0, |v| v.capacity() * std::mem::size_of::<f32>())
    }
    pub(in super::super) fn sample(&self, u: f32, v: f32) -> f32 {
        let [nx, ny] = self.count;
        let Some(values) = &self.values else {
            return periodic_noise(u, v, nx, ny, self.seed);
        };
        let x = periodic_unit(u) * nx as f32;
        let y = periodic_unit(v) * ny as f32;
        let ix = x.floor() as u32;
        let iy = y.floor() as u32;
        // Tiny negative coordinates can round rem_euclid to exactly 1.0.
        // Nonfinite and exceptional coordinates retain the original scalar
        // calculations, cast/modulo behavior and NaN payload propagation.
        if !x.is_finite() || !y.is_finite() || ix >= nx || iy >= ny {
            return periodic_noise(u, v, nx, ny, self.seed);
        }
        let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
        let tx = smooth(x.fract());
        let ty = smooth(y.fract());
        let stride = nx as usize + 1;
        let offset = iy as usize * stride + ix as usize;
        let a = values[offset];
        let b = values[offset + 1];
        let c = values[offset + stride];
        let d = values[offset + stride + 1];
        (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Copied original expression. Keep this independent of prepared lookup and
    // the production scalar helper to test arithmetic and wrapped hash identity.
    fn original_noise(u: f32, v: f32, nx: u32, ny: u32, seed: u64) -> f32 {
        let x = u.rem_euclid(1.0) * nx as f32;
        let y = v.rem_euclid(1.0) * ny as f32;
        let ix = x.floor() as u32;
        let iy = y.floor() as u32;
        let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
        let tx = smooth(x.fract());
        let ty = smooth(y.fract());
        let a = hash(ix % nx, iy % ny, seed);
        let b = hash((ix + 1) % nx, iy % ny, seed);
        let c = hash(ix % nx, (iy + 1) % ny, seed);
        let d = hash((ix + 1) % nx, (iy + 1) % ny, seed);
        (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty
    }

    #[test]
    fn unit_wrap_noise_replays_original_complete_atlas_fields() {
        for count in [[1, 1], [3, 5], [71, 73], [192, 192]] {
            for seed in [202, 43_084_482] {
                for budget in [0, 256 * 1024] {
                    let mut remaining = budget;
                    let prepared = PreparedPeriodic::new(count, seed, &mut remaining);
                    for y in 0..256 {
                        for x in 0..256 {
                            let [u, v] = [x as f32 / 256., y as f32 / 256.];
                            let original = original_noise(u, v, count[0], count[1], seed);
                            assert_eq!(
                                periodic_noise(u, v, count[0], count[1], seed).to_bits(),
                                original.to_bits(),
                                "scalar count {count:?} seed {seed} uv {:?}",
                                [u, v]
                            );
                            assert_eq!(
                                prepared.sample(u, v).to_bits(),
                                original.to_bits(),
                                "prepared count {count:?} seed {seed} budget {budget} uv {:?}",
                                [u, v]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_periodic_noise_preserves_bits_at_wrapped_edges_and_fallbacks() {
        for count in [[1, 1], [3, 5], [192, 192], [193, 192], [768, 768]] {
            for seed in [0, 202, 43_084_584, u64::MAX] {
                for budget in [0, 32, 256 * 1024] {
                    let mut remaining = budget;
                    let prepared = PreparedPeriodic::new(count, seed, &mut remaining);
                    let bytes = (count[0] as usize + 1) * (count[1] as usize + 1) * 4;
                    assert_eq!(
                        prepared.values.is_some(),
                        count[0] * count[1] <= MAX_LATTICE_CELLS && bytes <= budget
                    );
                    assert_eq!(remaining + prepared.bytes(), budget);
                    let samples = (0..32)
                        .flat_map(|y| (0..32).map(move |x| [x as f32 / 32., y as f32 / 32.]));
                    for [u, v] in samples.chain([
                        [-1. / 256., 1.],
                        [1., -1. / 256.],
                        [2.3, -0.4],
                        [-f32::EPSILON, f32::EPSILON],
                        [1. - f32::EPSILON, 1. - f32::EPSILON],
                    ]) {
                        assert_eq!(
                            periodic_noise(u, v, count[0], count[1], seed).to_bits(),
                            prepared.sample(u, v).to_bits(),
                            "count {count:?} seed {seed} budget {budget} uv {:?}",
                            [u, v]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_periodic_grids_share_a_total_allocation_budget() {
        let budget = 256 * 1024;
        let mut remaining = budget;
        let first = PreparedPeriodic::new([192, 192], 202, &mut remaining);
        let second = PreparedPeriodic::new([192, 192], 207, &mut remaining);
        let small = PreparedPeriodic::new([3, 5], 43_084_584, &mut remaining);
        assert!(first.values.is_some());
        assert!(second.values.is_none());
        assert!(small.values.is_some());
        assert_eq!(
            first.bytes() + second.bytes() + small.bytes() + remaining,
            budget
        );
        let overflow = PreparedPeriodic::new([u32::MAX, 2], 207, &mut remaining);
        assert!(overflow.values.is_none());
    }

    #[test]
    fn periodic_halo_preserves_original_bits_at_rounded_and_nonfinite_edges() {
        let tiny = f32::from_bits(1);
        let around_one = [
            f32::from_bits(1.0f32.to_bits() - 1),
            1.,
            f32::from_bits(1.0f32.to_bits() + 1),
        ];
        let coordinates = [
            -0.,
            0.,
            -tiny,
            tiny,
            -f32::EPSILON,
            f32::EPSILON,
            around_one[0],
            around_one[1],
            around_one[2],
            -around_one[0],
            -around_one[1],
            -around_one[2],
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7fc12345),
        ];
        assert_eq!((-tiny).rem_euclid(1.), 1.);
        for count in [[1, 1], [3, 5], [192, 192], [1, MAX_LATTICE_CELLS]] {
            for seed in [0, 202, u64::MAX] {
                let needed = (count[0] as usize + 1) * (count[1] as usize + 1) * 4;
                for budget in [0, needed - 1, needed] {
                    let mut remaining = budget;
                    let prepared = PreparedPeriodic::new(count, seed, &mut remaining);
                    assert_eq!(prepared.values.is_some(), budget >= needed);
                    assert_eq!(remaining + prepared.bytes(), budget);
                    for &u in &coordinates {
                        for &v in &coordinates {
                            assert_eq!(
                                original_noise(u, v, count[0], count[1], seed).to_bits(),
                                prepared.sample(u, v).to_bits(),
                                "count {count:?} seed {seed} budget {budget} uv {:?}",
                                [u, v]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn periodic_halo_counts_every_wrapped_border_byte_and_checks_overflow() {
        let count = [3, 5];
        let needed = 4 * 6 * std::mem::size_of::<f32>();
        let mut remaining = needed;
        let prepared = PreparedPeriodic::new(count, 202, &mut remaining);
        assert_eq!(remaining, 0);
        assert_eq!(prepared.bytes(), needed);
        let values = prepared.values.unwrap();
        for y in 0..=count[1] {
            for x in 0..=count[0] {
                assert_eq!(
                    values[(y * (count[0] + 1) + x) as usize].to_bits(),
                    hash(x % count[0], y % count[1], 202).to_bits()
                );
            }
        }
        for count in [
            [193, 192],
            [u32::MAX, 1],
            [1, u32::MAX],
            [u32::MAX, 0],
            [0, u32::MAX],
        ] {
            let mut remaining = 256 * 1024;
            let prepared = PreparedPeriodic::new(count, 202, &mut remaining);
            assert!(prepared.values.is_none());
            assert_eq!(prepared.bytes(), 0);
            assert_eq!(remaining, 256 * 1024);
        }
    }
}
