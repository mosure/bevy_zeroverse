//! Anatomical retargeting frames. Match both the spine and shoulder axes; a
//! shortest-arc spine rotation alone loses yaw and separates the arm roots.
use bevy::prelude::*;

pub(super) fn cranial_rotation(joints: &[Vec3], yaw: f32) -> Quat {
    frame(joints[2] - joints[0], joints[9] - joints[5]) * Quat::from_rotation_y(yaw)
}

/// Skin, hair, eyewear and annotations share this exact head frame.
pub(super) fn head_rotation(joints: &[Vec3], human: &super::IndoorHuman) -> Quat {
    let (pitch, roll) = human
        .pose_program
        .as_ref()
        .map_or((0., 0.), |p| (p.head_pitch, p.head_roll));
    cranial_rotation(joints, human.head_yaw)
        * Quat::from_rotation_x(pitch)
        * Quat::from_rotation_z(roll)
}

/// Bend around the mapped knuckle with the inherited parent transform. Bone
/// lengths and attachment positions are invariant; flexion is toward the palm.
pub(super) fn curl_finger(
    parent: Mat4,
    origin: Vec3,
    direction: Vec3,
    normal: Vec3,
    angle: f32,
) -> Mat4 {
    let pivot = parent.transform_point3(origin);
    let axis = parent
        .transform_vector3(normal)
        .cross(parent.transform_vector3(direction))
        .normalize_or(Vec3::X);
    Mat4::from_translation(pivot)
        * Mat4::from_quat(Quat::from_axis_angle(axis, angle))
        * Mat4::from_translation(-pivot)
        * parent
}

/// Straighten a pointing index without moving its knuckle or changing length.
pub(super) fn point_finger(
    parent: Mat4,
    origin: Vec3,
    direction: Vec3,
    target: Vec3,
    amount: f32,
) -> Mat4 {
    let pivot = parent.transform_point3(origin);
    let from = parent
        .transform_vector3(direction)
        .normalize_or(Vec3::NEG_Z);
    let to = target.normalize_or(from);
    let rotation = Quat::IDENTITY.slerp(Quat::from_rotation_arc(from, to), amount);
    Mat4::from_translation(pivot)
        * Mat4::from_quat(rotation)
        * Mat4::from_translation(-pivot)
        * parent
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

/// Pose programs use approximate joint spans before the body model is loaded.
/// Attach each complete arm chain to the evaluated anatomical shoulder instead
/// of stretching the skin between two disagreeing arm roots. Keep each wrist
/// target and limb length, retaining the sampled elbow swivel. Supported palms
/// are subsequently fitted to the actual skin/cloth thickness by contact IK.
pub(super) fn attach_shoulders(torso: Mat4, shoulders: [Vec3; 2], joints: &mut [Vec3]) {
    for (shoulder, base) in shoulders.into_iter().zip([5, 9]) {
        let root = torso.transform_point3(shoulder);
        let old_wrist = joints[base + 2];
        let pole = joints[base + 1] - (joints[base] + old_wrist) * 0.5;
        let upper = joints[base].distance(joints[base + 1]);
        let lower = joints[base + 1].distance(old_wrist);
        let (elbow, wrist) = super::poses::two_bone(root, old_wrist, pole, upper, lower);
        joints[base] = root;
        joints[base + 1] = elbow;
        joints[base + 2] = wrist;
        joints[base + 3] += wrist - old_wrist;
    }
}

/// Match endpoints while retaining transverse anatomical dimensions.
pub(super) fn segment(a: Vec3, b: Vec3, target_a: Vec3, target_b: Vec3, scale: f32) -> Mat4 {
    let source = b - a;
    let target = target_b - target_a;
    let rotation =
        Quat::from_rotation_arc(source.normalize_or(Vec3::Y), target.normalize_or(Vec3::Y));
    aligned_segment(a, source, target_a, target, scale, rotation)
}

/// Lock the palm plane as well as its long axis. A shortest-arc wrist rotation
/// otherwise leaves supported hands resting on their little-finger edge.
pub(super) fn palm_segment(
    a: Vec3,
    b: Vec3,
    across: Vec3,
    target_a: Vec3,
    target_b: Vec3,
    scale: f32,
) -> Mat4 {
    let source = b - a;
    let target = target_b - target_a;
    let source_axis = source.normalize_or(Vec3::X);
    let normal = source_axis.cross(across).normalize_or(Vec3::Y);
    let target_axis = target.normalize_or(Vec3::NEG_Z);
    let source_frame = Mat3::from_cols(source_axis, normal.cross(source_axis), normal);
    let target_normal = Vec3::Y;
    let target_frame =
        Mat3::from_cols(target_axis, target_normal.cross(target_axis), target_normal);
    let rotation = Quat::from_mat3(&(target_frame * source_frame.transpose()));
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

/// Extend a supported finger in the palm plane, preserving its bone length,
/// inherited roll and its parent's mapped knuckle position.
pub(super) fn supported_finger(wrist: Mat4, a: Vec3, b: Vec3, origin: Vec3) -> Mat4 {
    let direction = wrist.transform_vector3(b - a).normalize_or(Vec3::NEG_Z);
    let flat = direction.with_y(0.).normalize_or(Vec3::NEG_Z);
    let correction = Mat3::from_quat(Quat::from_rotation_arc(direction, flat));
    let linear = correction * Mat3::from_mat4(wrist);
    Mat4::from_cols(
        linear.x_axis.extend(0.),
        linear.y_axis.extend(0.),
        linear.z_axis.extend(0.),
        (origin - linear * a).extend(1.),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finger_flexion_keeps_knuckles_and_lengths_attached() {
        let mut parent = Mat4::from_rotation_y(0.7);
        for (i, angle) in [0.3, 0.6, 0.4].into_iter().enumerate() {
            let origin = Vec3::NEG_Z * i as f32 * 0.03;
            let next = origin + Vec3::NEG_Z * 0.03;
            let bent = curl_finger(parent, origin, Vec3::NEG_Z, Vec3::Y, angle);
            assert!(
                bent.transform_point3(origin)
                    .distance(parent.transform_point3(origin))
                    < 1e-6
            );
            assert!(
                (bent
                    .transform_point3(next)
                    .distance(bent.transform_point3(origin))
                    - 0.03)
                    .abs()
                    < 1e-6
            );
            assert!(bent.transform_point3(next).y < bent.transform_point3(origin).y);
            parent = bent;
        }
    }

    #[test]
    fn pointing_index_matches_hand_axis_without_stretching() {
        let parent = Mat4::from_scale_rotation_translation(
            Vec3::splat(0.8),
            Quat::from_rotation_y(0.6),
            Vec3::new(0.2, 0.9, -0.1),
        );
        let origin = Vec3::new(0.1, 0.2, 0.3);
        let direction = Vec3::new(0.015, 0.01, -0.025);
        let target = Vec3::new(0.5, 0.3, -0.7).normalize();
        let aimed = point_finger(parent, origin, direction, target, 1.);
        assert!(
            aimed
                .transform_point3(origin)
                .distance(parent.transform_point3(origin))
                < 1e-6
        );
        assert!(
            aimed
                .transform_vector3(direction)
                .normalize()
                .distance(target)
                < 1e-6
        );
        assert!(
            (aimed.transform_vector3(direction).length() - direction.length() * 0.8).abs() < 1e-6
        );
    }

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

    #[test]
    fn anatomical_attachment_preserves_limb_lengths_wrist_goals_and_torso_endpoints() {
        let shoulders = [Vec3::new(-0.16, 1., 0.), Vec3::new(0.16, 1., 0.)];
        for (scale, lean, yaw) in [(0.8, -0.2, -0.5), (1.1, 0.3, 0.7), (1.4, 0., 0.)] {
            let rotation = Quat::from_rotation_z(lean) * Quat::from_rotation_y(yaw);
            let mut joints = vec![Vec3::ZERO; 21];
            joints[0] = Vec3::new(0.02, 0.6, 0.01);
            joints[2] = joints[0] + rotation * Vec3::Y * scale;
            for (side, base) in [(-1., 5), (1., 9)] {
                for i in 0..4 {
                    joints[base + i] = joints[2]
                        + rotation
                            * Vec3::new(
                                side * (0.23 + if i == 1 { 0.08 } else { 0. }) * scale,
                                -0.1 * i as f32,
                                -0.08 * i as f32,
                            );
                }
            }
            let before = joints.clone();
            let torso = torso_segment(Vec3::ZERO, Vec3::Y, shoulders, &joints, scale);
            attach_shoulders(torso, shoulders, &mut joints);
            assert_eq!(&joints[..5], &before[..5]);
            assert_eq!(&joints[13..], &before[13..]);
            assert!(torso.transform_point3(Vec3::ZERO).distance(joints[0]) < 1e-6);
            assert!(torso.transform_point3(Vec3::Y).distance(joints[2]) < 1e-6);
            for (shoulder, base) in shoulders.into_iter().zip([5, 9]) {
                assert!(torso.transform_point3(shoulder).distance(joints[base]) < 1e-6);
                assert!(joints[base + 2].distance(before[base + 2]) < 1e-6);
                assert!(joints[base + 3].distance(before[base + 3]) < 1e-6);
                for i in 1..3 {
                    assert!(
                        (joints[base + i].distance(joints[base + i - 1])
                            - before[base + i].distance(before[base + i - 1]))
                        .abs()
                            < 1e-6
                    );
                }
            }
        }
    }
}
