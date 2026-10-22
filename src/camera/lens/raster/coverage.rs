//! Conservative sensor-footprint cropping. Each accepted interval cell contains
//! every possible contributing ray, not just a grid of sampled pixel centers.
//! Keep the existing source texel lattice, filter context and shadow frustum.
use super::{
    density::{angular_bounds, may_reach_sensor, Interval},
    *,
};
use crate::calibration::CameraCalibration;

pub(in crate::camera::lens) fn sensor_regions(
    side: u32,
    scale: f32,
    c: &CameraCalibration,
) -> [URect; 6] {
    refined_regions::<4>(side, scale, c)
}

fn refined_regions<const REFINEMENT: u32>(
    side: u32,
    scale: f32,
    c: &CameraCalibration,
) -> [URect; 6] {
    let lens = c.fisheye.expect("angular source coverage");
    let mut result = super::regions(side, scale, lens.max_angle);
    result[5] = URect::default();
    if lens.max_angle <= std::f32::consts::FRAC_PI_4 {
        result[1..].fill(URect::default());
        return result;
    }
    let cells = 32 * REFINEMENT;
    // Identical context to the existing cone crop. Include complete bilinear
    // footprints and align postprocess mip texel phases to the original image.
    let context = (side as f64 / 16.).ceil().max(32.);
    for (face, region) in result.iter_mut().enumerate().take(5).skip(1) {
        let mut lo = UVec2::splat(cells);
        let mut hi = UVec2::ZERO;
        let reaches = |min: UVec2, size: u32| {
            let axis = |i| {
                Interval::new(
                    (2. * i as f64 / cells as f64 - 1.) / scale as f64,
                    (2. * (i + size) as f64 / cells as f64 - 1.) / scale as f64,
                )
            };
            let (u, v, one) = (axis(min.x), axis(min.y).neg(), Interval::point(1.));
            let d = match face {
                1 => [one, v, u],
                2 => [one.neg(), v, u.neg()],
                3 => [u, one, v],
                _ => [u, one.neg(), v.neg()],
            };
            let (r, theta) = angular_bounds(d);
            may_reach_sensor(c, lens, d, r, theta)
        };
        // Refine only cells that can extend the rectangle. A uniform fine grid
        // costs 16x as many interval evaluations; interior cells cannot change
        // a bounding rectangle and need no evaluation at all. Bounds are still
        // interval proofs over complete cells, never point-sampled visibility.
        for y in (0..cells).step_by(REFINEMENT as usize) {
            for x in (0..cells).step_by(REFINEMENT as usize) {
                extend(UVec2::new(x, y), REFINEMENT, &mut lo, &mut hi, &reaches);
            }
        }
        if hi == UVec2::ZERO {
            *region = URect::default();
            continue;
        }
        let min = lo.to_array().map(|v| {
            ((v as f64 * side as f64 / cells as f64 - context)
                .floor()
                .max(0.) as u32
                / 16)
                * 16
        });
        let max = hi.to_array().map(|v| {
            ((v as f64 * side as f64 / cells as f64 + context).ceil() as u32)
                .div_ceil(16)
                .saturating_mul(16)
                .min(side)
        });
        region.min = region.min.max(UVec2::from_array(min));
        region.max = region.max.min(UVec2::from_array(max));
        if region.min.cmpge(region.max).any() {
            *region = URect::default();
        }
    }
    result
}

fn extend(
    min: UVec2,
    size: u32,
    lo: &mut UVec2,
    hi: &mut UVec2,
    reaches: &impl Fn(UVec2, u32) -> bool,
) {
    let max = min + UVec2::splat(size);
    if (min.cmpge(*lo).all() && max.cmple(*hi).all()) || !reaches(min, size) {
        return;
    }
    if size == 1 {
        *lo = (*lo).min(min);
        *hi = (*hi).max(max);
        return;
    }
    let half = size / 2;
    for offset in [UVec2::ZERO, UVec2::X, UVec2::Y, UVec2::ONE] {
        extend(min + offset * half, half, lo, hi, reaches);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Fisheye, FisheyeFit};

    #[test]
    fn adaptive_rectangle_matches_exhaustive_cell_union() {
        // Independent brute-force oracle for pruning and subdivision, including
        // disjoint lobes and domains crossing the source-square boundary.
        for center in [Vec2::ZERO, Vec2::new(31., 67.), Vec2::new(120., 95.)] {
            for radius in [1., 19., 80., 190.] {
                let reaches = |min: UVec2, size: u32| {
                    let a = min.as_vec2();
                    let b = a + Vec2::splat(size as f32);
                    center.distance(center.clamp(a, b)) <= radius
                        || (center + Vec2::splat(40.))
                            .distance((center + Vec2::splat(40.)).clamp(a, b))
                            <= radius
                };
                let (mut lo, mut hi) = (UVec2::splat(128), UVec2::ZERO);
                for y in (0..128).step_by(4) {
                    for x in (0..128).step_by(4) {
                        extend(UVec2::new(x, y), 4, &mut lo, &mut hi, &reaches);
                    }
                }
                let (mut expected_lo, mut expected_hi) = (UVec2::splat(128), UVec2::ZERO);
                for y in 0..128 {
                    for x in 0..128 {
                        let p = UVec2::new(x, y);
                        if reaches(p, 1) {
                            expected_lo = expected_lo.min(p);
                            expected_hi = expected_hi.max(p + UVec2::ONE);
                        }
                    }
                }
                assert_eq!((lo, hi), (expected_lo, expected_hi));
            }
        }
    }

    #[test]
    fn all_sensor_fits_keep_compositor_reads_inside_source_crops() {
        let rotations = super::super::super::runtime::orientations();
        let mut checked = 0;
        for fit in [
            FisheyeFit::Circular,
            FisheyeFit::Diagonal,
            FisheyeFit::Elliptical,
        ] {
            for degrees in [40_f32, 90., 140., 180., 220.] {
                for size in [[768, 768], [960, 540], [540, 960]] {
                    for offset in [[0.; 2], [0.16, -0.13]] {
                        let lens = Fisheye {
                            coefficients: [-0.03, 0.002, 0.00001, 0.],
                            max_angle: 0.5 * degrees.to_radians(),
                        };
                        let c = lens
                            .calibration(
                                size,
                                fit,
                                offset,
                                if fit == FisheyeFit::Circular { 1. } else { 1.2 },
                            )
                            .unwrap();
                        for side in [64, 256, 1536] {
                            let scale = 0.96;
                            let regions = sensor_regions(side, scale, &c);
                            let blend_start = scale / (1. - 2. / side as f32);
                            assert_eq!(regions[5], URect::default());
                            for y in 0..=32 {
                                for x in 0..=32 {
                                    let Some(ray) = c.ray([
                                        size[0] as f32 * x as f32 / 32.,
                                        size[1] as f32 * y as f32 / 32.,
                                    ]) else {
                                        continue;
                                    };
                                    let ray = Vec3::from_array(ray);
                                    let largest = ray.abs().max_element();
                                    for (i, rotation) in rotations.iter().enumerate().take(5) {
                                        let local = rotation.inverse() * ray;
                                        let sampled = if degrees <= 90. {
                                            i == 0
                                        } else {
                                            -local.z / largest > blend_start
                                        };
                                        if !sampled {
                                            continue;
                                        }
                                        let p =
                                            (local.xy() / -local.z * Vec2::new(0.5, -0.5) * scale
                                                + Vec2::splat(0.5))
                                                * side as f32;
                                        let region = regions[i];
                                        assert!(p.cmpge(region.min.as_vec2() + Vec2::ONE).all()
                                            && p.cmple(region.max.as_vec2() - Vec2::ONE).all(),
                                            "{fit:?} fov={degrees} size={size:?} side={side} face={i}: {p:?} {region:?}");
                                        checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 100_000);
    }

    #[test]
    fn wide_rectangular_sensor_omits_unused_poles() {
        let lens = Fisheye {
            coefficients: [0.; 4],
            max_angle: 70_f32.to_radians(),
        };
        let diagonal = lens
            .calibration([960, 540], FisheyeFit::Diagonal, [0.; 2], 1.)
            .unwrap();
        let circular = lens
            .calibration([960, 540], FisheyeFit::Circular, [0.; 2], 1.)
            .unwrap();
        let count = |c: &CameraCalibration| {
            sensor_regions(1024, 0.96, c)
                .iter()
                .filter(|r| r.width() > 0)
                .count()
        };
        assert_eq!(count(&diagonal), 3);
        assert_eq!(count(&circular), 5);
    }

    #[test]
    fn refined_coverage_reduces_source_pixels_without_changing_lenses() {
        let prior = crate::camera::lens::LensSampling {
            probability: 1.,
            ..default()
        };
        let mut pixels = [0_u64; 2];
        for seed in 0..96 {
            let sensor = prior.sample(seed, 0).unwrap();
            for size in [[512, 512], [960, 540]] {
                let c = sensor.calibration(size, [0.16, -0.13]).unwrap();
                let coarse = refined_regions::<1>(1536, 0.96, &c);
                let refined = sensor_regions(1536, 0.96, &c);
                for (a, b) in coarse.into_iter().zip(refined) {
                    assert!(
                        b.width() == 0 || (b.min.cmpge(a.min).all() && b.max.cmple(a.max).all())
                    );
                    pixels[0] += a.width() as u64 * a.height() as u64;
                    pixels[1] += b.width() as u64 * b.height() as u64;
                }
            }
        }
        assert!(
            pixels[1] < pixels[0],
            "coverage refinement must remove redundant pixels"
        );
        println!(
            "source pixels: coarse={} refined={} saved={:.3}%",
            pixels[0],
            pixels[1],
            100. * (1. - pixels[1] as f64 / pixels[0] as f64)
        );
    }

    #[test]
    fn crops_cover_all_compositor_reads_for_varied_calibrations() {
        let prior = crate::camera::lens::LensSampling {
            probability: 1.,
            ..default()
        };
        for seed in 0..96 {
            let sensor = prior.sample(seed, 0).unwrap();
            let size = if seed % 2 == 0 {
                [512, 512]
            } else {
                [960, 540]
            };
            let c = sensor.calibration(size, [0.16, -0.13]).unwrap();
            let side = 1024;
            let scale = 0.96;
            let regions = sensor_regions(side, scale, &c);
            let rotations = super::super::super::runtime::orientations();
            let start = scale / (1. - 2. / side as f32);
            for y in 0..=64 {
                for x in 0..=64 {
                    let Some(ray) = c.ray([
                        size[0] as f32 * x as f32 / 64.,
                        size[1] as f32 * y as f32 / 64.,
                    ]) else {
                        continue;
                    };
                    let ray = Vec3::from_array(ray);
                    let largest = ray.abs().max_element();
                    for (i, rotation) in rotations.iter().enumerate().take(5) {
                        let local = rotation.inverse() * ray;
                        if -local.z / largest <= start {
                            continue;
                        }
                        let p = (local.xy() / -local.z * Vec2::new(0.5, -0.5) * scale
                            + Vec2::splat(0.5))
                            * side as f32;
                        let region = regions[i];
                        assert!(
                            region.width() > 0 && region.height() > 0,
                            "seed={seed} face={i}"
                        );
                        assert!(
                            p.cmpge(region.min.as_vec2() + Vec2::ONE).all()
                                && p.cmple(region.max.as_vec2() - Vec2::ONE).all(),
                            "seed={seed} face={i} p={p:?} region={region:?}"
                        );
                    }
                }
            }
        }
    }
}
