//! Display-only depth mapping. CPU capture decoding and the GPU material share
//! the same coefficients; scalar depth formats never pass through this mapping.

// Turbo polynomial coefficients: Anton Mikhailov / Ruofei Du, Google (2019).
// https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
// The polynomial is evaluated independently here using Horner's method.
#[allow(clippy::excessive_precision)]
/// Shared coefficients for the CPU display transform and renderer shader.
pub const TURBO_COEFFICIENTS: [[f32; 3]; 6] = [
    [0.13572138, 0.09140261, 0.10667330],
    [4.61539260, 2.19418839, 12.64194608],
    [-42.66032258, 4.84296658, -60.58204836],
    [132.13108234, -14.18503333, 110.36276771],
    [-152.94239396, 4.27729857, -89.90310912],
    [59.28637943, 2.82956604, 27.34824973],
];

/// Turbo visualization in **linear RGB**, matching the annotation render target.
/// Warm colors are near, cool colors far. log(1 + metres) retains indoor detail
/// without per-image min/max fitting, so a camera's scale stays fixed over time.
/// `far` is the camera's existing far range; this never modifies clipping/depth.
pub fn turbo_depth_rgb(depth: f32, far: f32) -> [f32; 3] {
    let x = 1.0 - ((1.0 + depth.max(0.0)).ln() / (1.0 + far.max(1e-5)).ln()).clamp(0.0, 1.0);
    std::array::from_fn(|channel| {
        let srgb = TURBO_COEFFICIENTS
            .iter()
            .rev()
            .fold(0.0, |a, c| a * x + c[channel])
            .clamp(0.0, 1.0);
        if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(2.4)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_depth_is_bounded_warm_to_cool_and_range_stable() {
        let near = turbo_depth_rgb(0.1, 50.0);
        let middle = turbo_depth_rgb(5.0, 50.0);
        let far = turbo_depth_rgb(30.0, 50.0);
        assert!(near[0] > near[1] && near[0] > near[2]);
        assert!(middle[1] > middle[0] && middle[1] > middle[2]);
        assert!(far[2] > far[0] && far[2] > far[1]);
        for i in 0..=1000 {
            assert!(turbo_depth_rgb(i as f32 * 0.1, 50.0)
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        }
        assert_eq!(turbo_depth_rgb(50.0, 50.0), turbo_depth_rgb(100.0, 50.0));
        assert_eq!(turbo_depth_rgb(-1.0, 50.0), turbo_depth_rgb(0.0, 50.0));
    }
}
