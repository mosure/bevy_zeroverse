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
    let lens = c.fisheye.expect("angular source coverage");
    let mut result = super::regions(side, scale, lens.max_angle);
    result[5] = URect::default();
    if lens.max_angle <= std::f32::consts::FRAC_PI_4 {
        result[1..].fill(URect::default());
        return result;
    }
    const CELLS: u32 = 32;
    // Identical context to the existing cone crop. Include complete bilinear
    // footprints and align postprocess mip texel phases to the original image.
    let context = (side as f64 / 16.).ceil().max(32.);
    for (face, region) in result.iter_mut().enumerate().take(5).skip(1) {
        let mut lo = UVec2::splat(CELLS);
        let mut hi = UVec2::ZERO;
        for y in 0..CELLS {
            for x in 0..CELLS {
                let axis = |i| {
                    Interval::new(
                        (2. * i as f64 / CELLS as f64 - 1.) / scale as f64,
                        (2. * (i + 1) as f64 / CELLS as f64 - 1.) / scale as f64,
                    )
                };
                let (u, v, one) = (axis(x), axis(y).neg(), Interval::point(1.));
                let d = match face {
                    1 => [one, v, u],
                    2 => [one.neg(), v, u.neg()],
                    3 => [u, one, v],
                    _ => [u, one.neg(), v.neg()],
                };
                let (r, theta) = angular_bounds(d);
                if may_reach_sensor(c, lens, d, r, theta) {
                    lo = lo.min(UVec2::new(x, y));
                    hi = hi.max(UVec2::new(x + 1, y + 1));
                }
            }
        }
        if hi == UVec2::ZERO {
            *region = URect::default();
            continue;
        }
        let min = lo.to_array().map(|v| {
            ((v as f64 * side as f64 / CELLS as f64 - context)
                .floor()
                .max(0.) as u32
                / 16)
                * 16
        });
        let max = hi.to_array().map(|v| {
            ((v as f64 * side as f64 / CELLS as f64 + context).ceil() as u32)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Fisheye, FisheyeFit};

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
