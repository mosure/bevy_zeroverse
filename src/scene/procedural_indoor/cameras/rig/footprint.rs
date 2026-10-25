//! Co-located, axis-aligned sensor coverage, never scene visibility. Cached per
//! immutable slot calibration during search. Invalid circular borders are excluded.
use super::IndoorCamera;
use bevy_zeroverse_capture::calibration::CameraCalibration;

fn calibration(camera: &IndoorCamera, aspect: f32) -> Option<CameraCalibration> {
    if !aspect.is_finite() || aspect <= 0. {
        return None;
    }
    let size = [(1000. * aspect).round() as u32, 1000];
    camera
        .fisheye
        .map_or_else(
            || {
                CameraCalibration::centered_pinhole(
                    size[0],
                    size[1],
                    camera.fov_degrees.to_radians(),
                    aspect,
                )
            },
            |lens| lens.calibration(size, [0.; 2]),
        )
        .ok()
}

pub(in crate::scene::procedural_indoor::cameras) fn overlap(
    a: &IndoorCamera,
    b: &IndoorCamera,
    aspect: f32,
) -> Option<f32> {
    let (a, b) = (calibration(a, aspect)?, calibration(b, aspect)?);
    // Exact for centered rectilinear sensors. This also resolves footprints
    // smaller than one cell of the visibility proposal grid.
    if a.fisheye.is_none() && b.fisheye.is_none() {
        let directed = |a: &CameraCalibration, b: &CameraCalibration| {
            (a.k[0][0] / b.k[0][0]).min(1.) * (a.k[1][1] / b.k[1][1]).min(1.)
        };
        return Some(directed(&a, &b).min(directed(&b, &a)));
    }
    let directed = |source: &CameraCalibration, target: &CameraCalibration| {
        let (mut shared, mut valid) = (0u32, 0u32);
        for y in 0..17 {
            for x in 0..17 {
                let pixel = [
                    (x as f32 + 0.5) / 17. * source.image_size[0] as f32,
                    (y as f32 + 0.5) / 17. * source.image_size[1] as f32,
                ];
                if let Some(ray) = source.ray(pixel) {
                    valid += 1;
                    shared += u32::from(target.project(ray).is_some_and(|p| target.in_image(p)));
                }
            }
        }
        shared as f32 / valid.max(1) as f32
    };
    Some(directed(&a, &b).min(directed(&b, &a)))
}
