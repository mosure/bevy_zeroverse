//! Check the final sensor model, independently of the perspective source views.
use anyhow::{ensure, Context, Result};
use bevy::prelude::{default, PerspectiveProjection};
use bevy_zeroverse::{
    camera::{lens, pinhole},
    sample::View,
    scene::procedural_indoor::layout::IndoorCamera,
};

pub fn validate<'a>(
    view: &'a View,
    camera: &IndoorCamera,
    size: [u32; 2],
) -> Result<&'a lens::CameraCalibration> {
    let expected_fov = camera
        .fisheye
        .map_or(camera.fov_degrees.to_radians(), |s| s.lens.max_angle * 2.);
    ensure!(
        (view.fovy - expected_fov).abs() < 0.00001
            && (view.near - 0.1).abs() < 0.00001
            && (view.far - 50.).abs() < 0.00001,
        "sensor FOV/clipping mismatch: fovy={} expected={expected_fov}, near={} expected=0.1, far={} expected=50",
        view.fovy,
        view.near,
        view.far
    );
    let offset = camera.principal_point_offset.unwrap_or_default();
    let projection = pinhole::projection(
        PerspectiveProjection {
            fov: camera.fov_degrees.to_radians(),
            aspect_ratio: size[0] as f32 / size[1] as f32,
            ..default()
        },
        offset,
    );
    let expected =
        lens::calibration(&projection, camera.fisheye, offset, size).map_err(anyhow::Error::msg)?;
    let actual = view
        .calibration
        .as_ref()
        .context("missing sensor calibration")?;
    actual.validate().map_err(anyhow::Error::msg)?;
    ensure!(
        actual.schema_version == expected.schema_version
            && actual.lens_model == expected.lens_model
            && actual.lens_model_version == expected.lens_model_version
            && actual.pixel_convention == expected.pixel_convention
            && actual.image_size == expected.image_size
            && actual.fisheye == expected.fisheye
            && actual
                .k
                .iter()
                .flatten()
                .zip(expected.k.iter().flatten())
                .all(|(a, b)| calibration_scalar_matches(*a, *b)),
        "sensor calibration mismatch: actual={actual:?}, expected={expected:?}"
    );
    Ok(actual)
}

fn calibration_scalar_matches(actual: f32, expected: f32) -> bool {
    // Matrix extraction and SSAA rescaling may round independently. At a focal
    // length of 900 pixels, two f32 ULPs already exceed the former fixed 1e-4
    // tolerance. Bound float arithmetic error relative to the coefficient;
    // keep schema, sensor dimensions, lens model and distortion checks exact.
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= 4. * f32::EPSILON * actual.abs().max(expected.abs()).max(1.)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::{Vec2, Vec3};

    #[test]
    fn projection_roundoff_is_not_a_calibration_failure() {
        for focal in [256.0_f32, 881.0401, 917.3916, 1944., 8192.] {
            let two_ulps_lower = f32::from_bits(focal.to_bits() - 2);
            assert!(calibration_scalar_matches(two_ulps_lower, focal));
            assert!(!calibration_scalar_matches(focal + 0.01, focal));
        }
        assert!(!calibration_scalar_matches(f32::NAN, 1.));
        assert!(!calibration_scalar_matches(f32::INFINITY, f32::INFINITY));
        assert!(!calibration_scalar_matches(0.00001, 0.));
        assert!(!calibration_scalar_matches(256.5, 256.));
    }

    #[test]
    fn mixed_lens_export_checks_the_sensor_and_rejects_corrupted_labels() {
        for fit in [
            None,
            Some(lens::FisheyeFit::Circular),
            Some(lens::FisheyeFit::Diagonal),
            Some(lens::FisheyeFit::Elliptical),
        ] {
            let camera = IndoorCamera {
                ceiling_mount: None,
                fisheye: fit.map(|fit| lens::FisheyeSensor {
                    lens: lens::Fisheye {
                        max_angle: 1.6175672,
                        coefficients: [-0.015896793, 0.0025026414, 0.00000884685, 0.00000044163585],
                    },
                    fit,
                    anisotropy: if fit == lens::FisheyeFit::Circular {
                        1.
                    } else {
                        0.9939112
                    },
                }),
                principal_point_offset: Some(Vec2::new(0.02, -0.03)),
                start: Vec3::ZERO,
                end: Vec3::ZERO,
                target: Vec3::NEG_Z,
                fov_degrees: 49.427345,
                motion: None,
            };
            let size = [640, 512];
            let mut calibration = if let Some(lens) = camera.fisheye {
                lens.calibration(size, [0.02, -0.03]).unwrap()
            } else {
                let mut c = lens::CameraCalibration::centered_pinhole(
                    640,
                    512,
                    camera.fov_degrees.to_radians(),
                    1.25,
                )
                .unwrap();
                c.k[0][2] += 0.02 * 640.;
                c.k[1][2] -= 0.03 * 512.;
                c
            };
            let view = View {
                fovy: camera
                    .fisheye
                    .map_or(camera.fov_degrees.to_radians(), |s| 2. * s.lens.max_angle),
                near: 0.1,
                far: 50.,
                calibration: Some(calibration.clone()),
                ..default()
            };
            assert_eq!(validate(&view, &camera, size).unwrap(), &calibration);
            assert!(calibration.k[0][0] > 0. && calibration.k[1][1] > 0.);
            for invalid in [
                View {
                    fovy: view.fovy + 0.01,
                    ..view.clone()
                },
                View {
                    near: 0.2,
                    ..view.clone()
                },
                View {
                    far: 25.,
                    ..view.clone()
                },
                View {
                    calibration: None,
                    ..view.clone()
                },
            ] {
                assert!(validate(&invalid, &camera, size).is_err());
            }
            calibration.k[0][2] += 1.;
            assert!(validate(
                &View {
                    calibration: Some(calibration),
                    ..view
                },
                &camera,
                size
            )
            .is_err());
        }
    }
}
