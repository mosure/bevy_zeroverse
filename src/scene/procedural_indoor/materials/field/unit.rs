//! The remainder by one is the input itself throughout the unit interval.
//! Retain the original operation elsewhere, including rounded negative seams,
//! signed zero, infinities and every NaN payload.

#[inline]
pub(in super::super) fn periodic_unit(u: f32) -> f32 {
    if (0.0..1.0).contains(&u) {
        u
    } else {
        u.rem_euclid(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(bits: u32) {
        let u = std::hint::black_box(f32::from_bits(bits));
        assert_eq!(
            periodic_unit(u).to_bits(),
            u.rem_euclid(1.0).to_bits(),
            "unit remainder changed at {bits:#010x}"
        );
    }

    #[test]
    fn periodic_unit_preserves_dense_seams_subnormals_and_all_exponents() {
        // Exhaust the nearby representable values, rather than comparing only
        // decimal inputs: negative subnormals can wrap to exactly 1.0.
        for bits in 0..=65_536 {
            compare(bits);
            compare(bits | 0x8000_0000);
        }
        for center in [0.5f32, 1., 2.] {
            for bits in center.to_bits() - 4096..=center.to_bits() + 4096 {
                compare(bits);
                compare(bits | 0x8000_0000);
            }
        }
        // Cover both signs and every normal/nonfinite exponent with stratified
        // mantissas, including noncanonical quiet and signaling NaN payloads.
        for exponent in 0..=255 {
            for mantissa in (0..=0x007f_ffffu32).step_by(8191) {
                let bits = (exponent << 23) | mantissa;
                compare(bits);
                compare(bits | 0x8000_0000);
            }
        }
        for bits in [
            0x7f7f_ffff,
            0x7f80_0000,
            0x7f80_0001,
            0x7fc1_2345,
            0x7fff_ffff,
        ] {
            compare(bits);
            compare(bits | 0x8000_0000);
        }
        assert_eq!(periodic_unit(-0.).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(periodic_unit(-f32::from_bits(1)), 1.);
    }

    #[test]
    fn periodic_unit_matches_original_over_metric_coordinate_grid() {
        // An exhaustive fixed-point grid spanning [-2, 2], with a resolution
        // finer than any current atlas, exercises both fast and wrapped paths.
        for i in -524_288..=524_288 {
            compare((i as f32 / 262_144.).to_bits());
        }
    }

    #[test]
    #[ignore = "CPU diagnostic; run alone outside capture/qualification timings"]
    fn periodic_unit_cpu_benchmark() {
        use std::{hint::black_box, time::Instant};
        let inputs: Vec<_> = (0..131_072)
            .map(|i| ((i % 512) as f32 / 512., i % 7 == 0))
            .collect();
        for mixed in [false, true] {
            for round in 0..8 {
                for fast in [round % 2 == 0, round % 2 != 0] {
                    let started = Instant::now();
                    let mut sum = 0.;
                    for _ in 0..8 {
                        for &(u, wrapped) in &inputs {
                            let u = black_box(if mixed && wrapped { u - 1. } else { u });
                            sum += if fast {
                                periodic_unit(u)
                            } else {
                                u.rem_euclid(1.)
                            };
                        }
                    }
                    let elapsed = started.elapsed().as_secs_f64();
                    black_box(sum);
                    println!(
                        "{{\"kernel\":\"unit_remainder\",\"mixed\":{mixed},\"fast\":{fast},\"round\":{round},\"samples\":{},\"seconds\":{elapsed}}}",
                        inputs.len() * 8
                    );
                }
            }
        }
    }
}
