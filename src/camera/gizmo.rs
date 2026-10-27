//! Shared camera presentation for Studio and inference clients. Colors depend
//! on capture ID, never ECS order or camera count. Geometry uses calibrated rays
//! in Bevy camera space (-Z forward, +Y up), bounded to a unit display radius.
mod runtime;
mod style;
use super::lens::CameraCalibration;
use bevy::prelude::*;
pub use runtime::{draw_camera_gizmo, PredictionCameraGizmoConfigGroup};
pub use style::{color, color_index, config, rgb};

pub const MAX_RADIUS_METRES: f32 = 0.35;
pub const MAX_RADIUS_PIXELS: f32 = 24.;

#[cfg(test)]
#[path = "gizmo/tests.rs"]
mod presentation_tests;

/// A compact calibrated wireframe. Scaling every point by one common display
/// factor preserves all optical ray directions and the pinhole image plane.
/// Fisheye keeps its spherical rim, including rays behind >180-degree sensors.
/// Cache the result until calibration changes; it is independent of the pose.
pub fn wireframe(c: &CameraCalibration) -> Result<Vec<[Vec3; 2]>, String> {
    c.validate()?;
    if c.fisheye.is_some() {
        return Ok(fisheye_lines(c));
    }
    let [w, h] = c.image_size.map(|v| v as f32);
    let mut corners = [Vec3::ZERO; 4];
    for (p, pixel) in corners.iter_mut().zip([[0., 0.], [w, 0.], [w, h], [0., h]]) {
        let ray = Vec3::from_array(c.ray(pixel).ok_or("invalid pinhole corner")?);
        *p = ray / -ray.z;
    }
    let radius = corners.iter().map(|p| p.length()).fold(1., f32::max);
    for p in &mut corners {
        *p /= radius;
    }
    let mut lines = Vec::with_capacity(8);
    for i in 0..4 {
        lines.push([Vec3::ZERO, corners[i]]);
        lines.push([corners[i], corners[(i + 1) % 4]]);
    }
    Ok(lines)
}

/// Prediction outline for use with `config(true)`. Keep visible gaps along
/// fisheye curves: Bevy restarts the dash phase per line segment, so short
/// tessellated arcs can otherwise appear solid. Cached geometric gaps follow
/// arc length continuously; straight pinhole edges use Bevy's dashed style.
pub fn prediction_wireframe(c: &CameraCalibration) -> Result<Vec<[Vec3; 2]>, String> {
    let lines = wireframe(c)?;
    if c.fisheye.is_none() {
        return Ok(lines);
    }
    let mut result = Vec::with_capacity(lines.len());
    let mut previous = None;
    let mut distance = 0_f32;
    let (dash, period) = (0.28, 0.46);
    for [a, b] in lines {
        if previous.is_none_or(|p: Vec3| p.distance_squared(a) > 1e-8) {
            distance = 0.;
        }
        previous = Some(b);
        let length = a.distance(b);
        if length <= 1e-6 {
            continue;
        }
        let end = distance + length;
        for cycle in (distance / period).floor() as usize..(end / period).ceil() as usize {
            let from = distance.max(cycle as f32 * period);
            let to = end.min(cycle as f32 * period + dash);
            if to - from > 1e-6 {
                result.push([
                    a.lerp(b, (from - distance) / length),
                    a.lerp(b, (to - distance) / length),
                ]);
            }
        }
        distance = end;
    }
    Ok(result)
}

/// Screen-aware display size, capped in metres as well as pixels. This never
/// changes a capture camera's pose, projection, near/far planes or annotation.
pub fn display_radius(
    projection: &Projection,
    view_from_world: Mat4,
    viewport: Vec2,
    origin: Vec3,
) -> f32 {
    if !viewport.is_finite() || viewport.min_element() <= 0. {
        return MAX_RADIUS_METRES;
    }
    let pixels = (viewport.min_element() * 0.025).clamp(10., MAX_RADIUS_PIXELS);
    let radius = if let Some((p, _)) = super::pinhole::parameters(projection) {
        let view = view_from_world.transform_point3(origin);
        let depth = -view.z;
        if !view.is_finite() || depth <= 0. {
            return 0.;
        }
        let focal = Vec2::new(viewport.x / p.aspect_ratio, viewport.y) / (2. * (p.fov * 0.5).tan());
        // Bound projection of the entire glyph, including its frontmost point
        // and perspective magnification at off-axis editor positions.
        let magnification = (focal * (Vec2::ONE + view.xy().abs() / depth)).max_element();
        pixels * depth / (magnification + pixels)
    } else if let Projection::Orthographic(p) = projection {
        pixels * (p.area.size() / viewport).min_element()
    } else {
        return MAX_RADIUS_METRES;
    };
    if radius.is_finite() {
        radius.clamp(0., MAX_RADIUS_METRES)
    } else {
        0.
    }
}

/// Validate external predicted poses before drawing. Scale, shear, reflections
/// and projective transforms are not valid calibrated camera extrinsics.
pub(crate) fn rigid_pose(matrix: Mat4) -> bool {
    let rotation = Mat3::from_mat4(matrix);
    matrix.is_finite()
        && (rotation.transpose() * rotation - Mat3::IDENTITY)
            .to_cols_array()
            .iter()
            .all(|v| v.abs() <= 0.01)
        && (rotation.determinant() - 1.).abs() <= 0.01
        && matrix.row(3).distance(Vec4::W) <= 0.001
}

pub(super) fn fisheye_lines(c: &CameraCalibration) -> Vec<[Vec3; 2]> {
    let Some(lens) = c.fisheye else {
        return Vec::new();
    };
    let radial = lens.radius(f64::from(lens.max_angle)) as f32;
    let center = Vec2::new(c.k[0][2], c.k[1][2]);
    let mut rim = Vec::with_capacity(64);
    let mut lines = Vec::with_capacity(64 + 4 * 16);
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
        // Four meridians convey angular shape without a dense wire cage at
        // the compact display size. The calibrated rim retains sensor fit.
        if i % 16 == 0 {
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
                    assert_eq!(lines.len(), 128);
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
