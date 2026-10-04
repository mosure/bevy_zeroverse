//! Anatomical retargeting frames. Match both the spine and shoulder axes; a
//! shortest-arc spine rotation alone loses yaw and separates the arm roots.
use bevy::prelude::*;

pub(super) fn cranial_rotation(joints: &[Vec3], yaw: f32) -> Quat {
    frame(joints[2] - joints[0], joints[9] - joints[5]) * Quat::from_rotation_y(yaw)
}

fn frame(up: Vec3, right: Vec3) -> Quat {
    let up = up.normalize_or(Vec3::Y);
    let back = right.normalize_or(Vec3::X).cross(up).normalize_or(Vec3::Z);
    let right = up.cross(back).normalize_or(Vec3::X);
    Quat::from_mat3(&Mat3::from_cols(right, up, back))
}

pub(super) fn torso_segment(
    pelvis: Vec3,
    chest: Vec3,
    shoulders: [Vec3; 2],
    joints: &[Vec3],
    scale: f32,
) -> Mat4 {
    let source = chest - pelvis;
    let target = joints[2] - joints[0];
    let rotation =
        cranial_rotation(joints, 0.0) * frame(source, shoulders[1] - shoulders[0]).inverse();
    aligned_segment(pelvis, source, joints[0], target, scale, rotation)
}

/// Match endpoints while retaining transverse anatomical dimensions.
pub(super) fn segment(a: Vec3, b: Vec3, target_a: Vec3, target_b: Vec3, scale: f32) -> Mat4 {
    let source = b - a;
    let target = target_b - target_a;
    let rotation =
        Quat::from_rotation_arc(source.normalize_or(Vec3::Y), target.normalize_or(Vec3::Y));
    aligned_segment(a, source, target_a, target, scale, rotation)
}

fn aligned_segment(
    a: Vec3,
    source: Vec3,
    target_a: Vec3,
    target: Vec3,
    scale: f32,
    rotation: Quat,
) -> Mat4 {
    let axis = source.normalize_or(Vec3::Y);
    let stretch = target.length() / source.length().max(0.0001);
    let radial = Mat3::IDENTITY * scale;
    let along = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z) * (stretch - scale);
    let linear = Mat3::from_quat(rotation) * (radial + along);
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        (target_a - linear * a).extend(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torso_keeps_shoulder_roots_attached_when_twisting() {
        let shoulders = [Vec3::new(-0.18, 1., 0.), Vec3::new(0.18, 1., 0.)];
        for yaw in [-0.24, -0.12, 0., 0.12, 0.24] {
            let rotation = Quat::from_rotation_y(yaw);
            let mut p = vec![Vec3::ZERO; 21];
            p[0] = Vec3::new(0.02, 0.6, 0.01);
            p[2] = p[0] + Vec3::Y;
            p[5] = p[2] + rotation * Vec3::NEG_X * 0.18;
            p[9] = p[2] + rotation * Vec3::X * 0.18;
            let t = torso_segment(Vec3::ZERO, Vec3::Y, shoulders, &p, 1.);
            assert!(t.transform_point3(Vec3::ZERO).distance(p[0]) < 1e-6);
            assert!(t.transform_point3(Vec3::Y).distance(p[2]) < 1e-6);
            assert!(t.transform_point3(shoulders[0]).distance(p[5]) < 1e-6);
            assert!(t.transform_point3(shoulders[1]).distance(p[9]) < 1e-6);
        }
    }
}
