//! Calibrated fisheye wireframes in Bevy camera space (-Z forward, +Y up).
//! A >180 degree rim is behind the sensor. Meridians through the forward axis
//! make its spherical coverage explicit instead of suggesting a reversed cone.
use super::lens::CameraCalibration;
use bevy::prelude::*;

pub(super) fn fisheye_lines(c: &CameraCalibration) -> Vec<[Vec3; 2]> {
    let Some(lens) = c.fisheye else {
        return Vec::new();
    };
    let radial = lens.radius(f64::from(lens.max_angle)) as f32;
    let center = Vec2::new(c.k[0][2], c.k[1][2]);
    let mut rim = Vec::with_capacity(64);
    let mut lines = Vec::with_capacity(64 + 8 * 16);
    for i in 0..64 {
        let phi = std::f32::consts::TAU * i as f32 / 64.;
        let delta = Vec2::new(
            c.k[0][0] * phi.cos() + c.k[0][1] * phi.sin(),
            c.k[1][1] * phi.sin(),
        );
        let mut r = radial;
        for axis in 0..2 {
            if delta[axis].abs() > 1e-6 {
                let boundary = if delta[axis] > 0. {
                    c.image_size[axis] as f32 - 0.5
                } else {
                    0.5
                };
                r = r.min((boundary - center[axis]) / delta[axis]);
            }
        }
        let edge = delta * r.max(0.) * 0.99999;
        let Some(ray) = c.ray((center + edge).to_array()).map(Vec3::from_array) else {
            continue;
        };
        rim.push(ray);
        if i % 8 == 0 {
            let mut previous = Vec3::NEG_Z;
            for step in 1..=16 {
                if let Some(next) = c
                    .ray((center + edge * (step as f32 / 16.)).to_array())
                    .map(Vec3::from_array)
                {
                    lines.push([previous, next]);
                    previous = next;
                }
            }
        }
    }
    if rim.len() > 1 {
        for i in 0..rim.len() {
            lines.push([rim[i], rim[(i + 1) % rim.len()]]);
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::lens::{FisheyeFit, LensSampling};
    #[test]
    fn outlines_follow_calibrated_rays_for_rotated_ceiling_and_room_cameras() {
        for fov in [90., 180., 200., 220.] {
            for fit in [
                FisheyeFit::Circular,
                FisheyeFit::Diagonal,
                FisheyeFit::Elliptical,
            ] {
                for size in [[512, 512], [640, 400], [400, 640]] {
                    let mut lens = LensSampling {
                        probability: 1.,
                        fov_degrees: [fov, fov],
                        ..default()
                    }
                    .sample(3, 0)
                    .unwrap();
                    lens.fit = fit;
                    lens.anisotropy = if fit == FisheyeFit::Circular {
                        1.
                    } else {
                        1.08
                    };
                    let calibration = lens.calibration(size, [0.12, -0.08]).unwrap();
                    let lines = fisheye_lines(&calibration);
                    assert_eq!(lines.len(), 192);
                    for normal in [
                        Vec3::NEG_Z,
                        Vec3::NEG_Y,
                        Vec3::new(0.2, -1., 0.1).normalize(),
                    ] {
                        let pose = Transform::from_xyz(2., 3., -1.)
                            .looking_to(normal, Vec3::X)
                            .with_scale(Vec3::ONE);
                        assert!((pose.rotation * Vec3::NEG_Z).distance(normal) < 1e-6);
                        let mut forward = false;
                        for ray in lines.iter().flatten() {
                            forward |= ray.distance(Vec3::NEG_Z) < 1e-6;
                            assert!((ray.length() - 1.).abs() < 1e-5);
                            let world = pose.transform_point(*ray * 1.5);
                            let recovered = pose.rotation.inverse() * (world - pose.translation);
                            let pixel = calibration.project(recovered.to_array()).unwrap();
                            let oracle = Vec3::from_array(calibration.ray(pixel).unwrap());
                            assert!(oracle.distance(recovered.normalize()) < 1e-5);
                            assert!(pixel[0] >= -0.001 && pixel[0] <= size[0] as f32 + 0.001);
                            assert!(pixel[1] >= -0.001 && pixel[1] <= size[1] as f32 + 0.001);
                        }
                        assert!(forward, "wireframe must include the true forward pole");
                    }
                }
            }
        }
    }
}
