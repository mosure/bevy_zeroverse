//! Exact replay oracle retained from the pre-cache environment convolution.
use super::*;
use rand::{Rng, SeedableRng};

struct ReferenceRadianceMips {
    source: RadianceMips,
}
impl ReferenceRadianceMips {
    fn sample(&self, direction: Vec3, lod: f32) -> Vec3 {
        let lod = lod.clamp(0., (self.source.0.len() - 1) as f32);
        let i = lod.floor() as usize;
        let (size, pixels) = &self.source.0[i];
        let a = sample(pixels, *size, direction);
        let (size, pixels) = &self.source.0[(i + 1).min(self.source.0.len() - 1)];
        a.lerp(sample(pixels, *size, direction), lod.fract())
    }
}
fn reference_convolve(pixels: &[Vec3], size: u32) -> [Image; 2] {
    let source = ReferenceRadianceMips {
        source: RadianceMips::new(pixels, size),
    };
    let texel_angle = 4. * PI / (6 * size * size) as f32;
    let levels = size.ilog2() + 1;
    let mut specular = Vec::new();
    let mut diffuse = Vec::new();
    // Explicit layer-major ordering matches Bevy/wgpu Image's upload contract.
    for face in 0..6 {
        for level in 0..levels {
            let n = size >> level;
            let roughness = level as f32 / (levels - 1) as f32;
            for y in 0..n {
                for x in 0..n {
                    let normal = direction(face, x as f32, y as f32, n);
                    let (t, b) = basis(normal);
                    let mut value = Vec3::ZERO;
                    let mut weight = 0.;
                    if level == 0 {
                        value = pixels[(face * size * size + y * size + x) as usize];
                    } else {
                        let alpha2 = roughness.powi(4);
                        for i in 0..SAMPLES {
                            let xi = hammersley(i);
                            let cos = ((1. - xi.x) / (1. + (alpha2 - 1.) * xi.x)).sqrt();
                            let sin = (1. - cos * cos).sqrt();
                            let phi = TAU * xi.y;
                            let h = t * (sin * phi.cos()) + b * (sin * phi.sin()) + normal * cos;
                            let l = h * (2. * cos) - normal;
                            let w = normal.dot(l).max(0.);
                            // Sample the source footprint implied by the GGX PDF.
                            // Reading only level zero made small HDR emitters turn
                            // into isolated noisy dots in rough reflections.
                            let denominator = cos * cos * (alpha2 - 1.) + 1.;
                            let pdf = alpha2 / (4. * PI * denominator * denominator);
                            let lod = 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2();
                            value += source.sample(l, lod) * w;
                            weight += w;
                        }
                        value /= weight.max(1e-6);
                    }
                    encode(&mut specular, value);
                }
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let n = direction(face, x as f32, y as f32, 16);
                let (t, b) = basis(n);
                let mut value = Vec3::ZERO;
                for i in 0..SAMPLES {
                    let xi = hammersley(i);
                    let r = xi.x.sqrt();
                    let phi = TAU * xi.y;
                    let l = t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1. - xi.x).sqrt();
                    let pdf = (1. - xi.x).sqrt() / PI;
                    let lod = 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2();
                    value += source.sample(l, lod);
                }
                encode(&mut diffuse, value / SAMPLES as f32);
            }
        }
    }
    [cube(diffuse, 16, 1), cube(specular, size, levels)]
}
fn seeded_hdr(size: u32, seed: u64) -> Vec<Vec3> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..6 * size * size)
        .map(|i| {
            let value = Vec3::new(
                rng.random_range(0.001..10.),
                rng.random_range(0.001..35.),
                rng.random_range(0.001..4.),
            );
            if i % (size * size / 3).max(1) == 0 {
                value + Vec3::new(3800., 1300., 590.)
            } else {
                value
            }
        })
        .collect()
}

#[test]
fn cached_convolution_matches_original_hdr_bytes_at_every_mip() {
    for size in [8, SIZE] {
        for seed in [0, 81, 202, 1_013_005, 43_084_584, 42_430_575] {
            let pixels = seeded_hdr(size, seed);
            let reference = reference_convolve(&pixels, size);
            let cached = convolve(&pixels, size);
            let round4 = super::round4_replay::round4_convolve(&pixels, size);
            for (a, b) in round4.iter().zip(&cached) {
                assert_eq!(a.data, b.data, "round4 size={size}, seed={seed}");
            }
            for (a, b) in reference.iter().zip(&cached) {
                assert_eq!(a.texture_descriptor, b.texture_descriptor);
                assert_eq!(a.data, b.data, "size={size}, seed={seed}");
            }
        }
    }
}

#[test]
fn cached_cube_edge_taps_preserve_original_sampling_bits() {
    let kernel = ConvolutionKernel::production();
    for (level, taps) in kernel.taps.iter().enumerate() {
        let size = SIZE >> level;
        let pixels = seeded_hdr(size, level as u64);
        for face in 0..6 {
            // Include every texel center, seam and corner at fractional offsets.
            for y in -1..=size as i32 {
                for x in -1..=size as i32 {
                    let d = direction(face, x as f32 + 0.25, y as f32 - 0.125, size);
                    let expected = sample(&pixels, size, d).to_array().map(f32::to_bits);
                    let actual = taps.sample(&pixels, d).to_array().map(f32::to_bits);
                    assert_eq!(actual, expected, "size={size}, face={face}, x={x}, y={y}");
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "counterbalanced CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn cached_environment_convolution_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    // Warm the fixed immutable kernel before timing recurring room work.
    ConvolutionKernel::production();
    let mut reference_seconds = Vec::new();
    let mut cached_seconds = Vec::new();
    for seed in 0..12 {
        let pixels = seeded_hdr(SIZE, seed);
        let reference = || {
            let start = Instant::now();
            black_box(reference_convolve(black_box(&pixels), SIZE));
            start.elapsed().as_secs_f64()
        };
        let cached = || {
            let start = Instant::now();
            black_box(convolve(black_box(&pixels), SIZE));
            start.elapsed().as_secs_f64()
        };
        let (a, b) = if seed % 2 == 0 {
            (reference(), cached())
        } else {
            let b = cached();
            (reference(), b)
        };
        reference_seconds.push(a);
        cached_seconds.push(b);
    }
    reference_seconds.sort_by(f64::total_cmp);
    cached_seconds.sort_by(f64::total_cmp);
    let median = |x: &[f64]| (x[5] + x[6]) * 0.5;
    let reference = median(&reference_seconds);
    let cached = median(&cached_seconds);
    eprintln!(
        "{}",
        serde_json::json!({
            "size": SIZE,
            "samples_per_texel": SAMPLES,
            "rooms": 12,
            "reference_median_seconds": reference,
            "cached_median_seconds": cached,
            "kernel_speedup": reference / cached,
            "reference_seconds": reference_seconds,
            "cached_seconds": cached_seconds,
            "scope": "CPU convolution only; counterbalanced recurring-room work; no GPU or end-to-end capture claim"
        })
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "counterbalanced round4-versus-stencil kernel screen; no capture throughput claim"]
fn reflection_stencil_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    let pixels = seeded_hdr(SIZE, 0);
    let cold = Instant::now();
    black_box(convolve(&pixels, SIZE));
    let cold_seconds = cold.elapsed().as_secs_f64();
    let mut original = Vec::new();
    let mut candidate = Vec::new();
    for seed in 0..12 {
        let pixels = seeded_hdr(SIZE, seed);
        let old = || {
            let start = Instant::now();
            black_box(super::round4_replay::round4_convolve(
                black_box(&pixels),
                SIZE,
            ));
            start.elapsed().as_secs_f64()
        };
        let new = || {
            let start = Instant::now();
            black_box(convolve(black_box(&pixels), SIZE));
            start.elapsed().as_secs_f64()
        };
        let (a, b) = if seed % 2 == 0 {
            (old(), new())
        } else {
            let b = new();
            (old(), b)
        };
        original.push(a);
        candidate.push(b);
    }
    original.sort_by(f64::total_cmp);
    candidate.sort_by(f64::total_cmp);
    let median = |x: &[f64]| (x[5] + x[6]) * 0.5;
    eprintln!(
        "{}",
        serde_json::json!({
            "round4_median_seconds": median(&original),
            "candidate_median_seconds": median(&candidate),
            "cold_candidate_seconds": cold_seconds,
            "speedup": median(&original) / median(&candidate),
            "round4_seconds": original,
            "candidate_seconds": candidate,
            "scope": "CPU reflection convolution only, original samples/math, cold lazy cache separately recorded"
        })
    );
}
