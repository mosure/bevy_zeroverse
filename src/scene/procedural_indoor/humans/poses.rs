//! Continuous end-effector programs, solved with fixed-length two-bone IK.
//! Activity labels describe the support context; they are not pose lookup tables.
use super::*;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseProgram {
    pub lean: Vec2,
    pub torso_twist: f32,
    pub weight_shift: f32,
    pub stance: f32,
    pub stride: f32,
    pub phase: f32,
    pub arm_reach: [f32; 2],
    pub arm_elevation: [f32; 2],
    pub arm_sweep: [f32; 2],
    pub elbow_pole: [f32; 2],
    pub foot_splay: [f32; 2],
}
impl PoseProgram {
    pub fn sample(seed: u64, pose: HumanPoseKind) -> Self {
        let mut rng = stream(seed, 44);
        let seated = pose.seated();
        // Independently sampled arm goals within the activity's ergonomic domain.
        // These are continuous regions, not stored skeletons or pose variants.
        let desk = pose == HumanPoseKind::SeatedWorking;
        let resting = matches!(
            pose,
            HumanPoseKind::StandingRelaxed
                | HumanPoseKind::StandingWalking
                | HumanPoseKind::SeatedListening
        );
        let reading = pose == HumanPoseKind::StandingReading;
        let presenting = pose == HumanPoseKind::StandingPresenting;
        Self {
            lean: Vec2::new(
                rng.random_range(-0.055..0.055),
                rng.random_range(-0.13..0.035),
            ),
            torso_twist: rng.random_range(-0.24..0.24),
            weight_shift: rng.random_range(-0.035..0.035),
            stance: rng.random_range(0.19..0.36),
            stride: if seated {
                rng.random_range(0.0..0.12)
            } else {
                rng.random_range(0.0..0.28)
            },
            phase: rng.random_range(0.0..TAU),
            arm_reach: std::array::from_fn(|_| {
                if resting {
                    rng.random_range(0.90..0.995)
                } else {
                    rng.random_range(0.45..0.88)
                }
            }),
            arm_elevation: std::array::from_fn(|i| {
                if desk {
                    rng.random_range(-0.79..-0.55)
                } else if resting {
                    rng.random_range(-1.49..-1.13)
                } else if reading {
                    rng.random_range(-0.95..-0.55)
                } else if i == 0 {
                    rng.random_range(-1.4..-1.0)
                } else if presenting {
                    rng.random_range(-0.6..0.48)
                } else {
                    rng.random_range(-1.1..-0.15)
                }
            }),
            arm_sweep: std::array::from_fn(|_| {
                if desk || reading {
                    rng.random_range(-0.35..0.06)
                } else {
                    rng.random_range(-0.12..0.38)
                }
            }),
            elbow_pole: std::array::from_fn(|_| rng.random_range(0.10..0.8)),
            foot_splay: std::array::from_fn(|_| rng.random_range(0.01..0.22)),
        }
    }
    pub fn solve(&self, stature: f32, build: f32, shoulder_width: f32, seated: bool) -> Vec<Vec3> {
        let s = stature / 1.75;
        let hip_half_span = 0.092 * build.sqrt() * s;
        let mut pelvis = Vec3::new(self.weight_shift, 0.585, 0.015);
        if !seated {
            let horizontal = [-1.0, 1.0]
                .into_iter()
                .map(|side| {
                    let hip = Vec2::new(self.weight_shift + side * hip_half_span, 0.015);
                    let ankle = Vec2::new(
                        side * self.stance * 0.5,
                        side * self.stride * self.phase.sin(),
                    );
                    hip.distance_squared(ankle)
                })
                .fold(0.0, f32::max);
            pelvis.y = 0.105 + ((0.86 * s * 0.987).powi(2) - horizontal).max(0.1).sqrt();
        }
        let torso = Quat::from_rotation_y(self.torso_twist)
            * Quat::from_rotation_x(self.lean.y)
            * Quat::from_rotation_z(-self.lean.x);
        let waist = pelvis + torso * Vec3::Y * 0.15 * s;
        let chest = pelvis + torso * Vec3::Y * 0.43 * s;
        let neck = pelvis + torso * Vec3::Y * 0.55 * s;
        let head = neck + torso * Vec3::new(0.0, 0.13, -0.006) * s;
        let mut joints = vec![pelvis, waist, chest, neck, head];
        for (i, side) in [-1.0, 1.0].into_iter().enumerate() {
            let shoulder = chest + torso * Vec3::new(side * shoulder_width * 0.5, -0.005, 0.0);
            let elevation = self.arm_elevation[i];
            let sweep = self.arm_sweep[i];
            let direction = Vec3::new(
                side * sweep.sin() * elevation.cos(),
                elevation.sin(),
                -sweep.cos() * elevation.cos(),
            );
            let goal = shoulder + torso * direction * (0.55 * s * self.arm_reach[i]);
            let (elbow, wrist) = two_bone(
                shoulder,
                goal,
                // Gravity-biased elbow swivel keeps the upper arm near the
                // torso. A predominantly lateral pole creates abducted elbows
                // even when the sampled wrist target is a resting gesture.
                torso * Vec3::new(side * 0.25, -1.0, self.elbow_pole[i] * 0.25),
                0.29 * s,
                0.26 * s,
            );
            let hand = wrist + (wrist - elbow).normalize_or(Vec3::NEG_Y) * 0.075 * s;
            joints.extend([shoulder, elbow, wrist, hand]);
        }
        for (i, side) in [-1.0, 1.0].into_iter().enumerate() {
            let hip = pelvis + Vec3::new(side * hip_half_span, 0.0, 0.0);
            let stride = side * self.stride * self.phase.sin();
            let goal = Vec3::new(
                side * self.stance * 0.5,
                0.105,
                if seated { -0.34 * s + stride } else { stride },
            );
            let (knee, ankle) = two_bone(
                hip,
                goal,
                Vec3::new(side * 0.12, 0.0, -1.0),
                0.43 * s,
                0.43 * s,
            );
            let toe = ankle
                + Quat::from_rotation_y(-side * self.foot_splay[i])
                    * Vec3::new(0.0, -0.06, -0.17 * s);
            joints.extend([hip, knee, ankle, toe]);
        }
        joints
    }
}
/// Clamps unreachable goals to the limb's feasible annulus, retaining both bone
/// lengths. Pole projection controls swivel without discontinuous Euler angles.
pub(super) fn two_bone(
    root: Vec3,
    target: Vec3,
    pole: Vec3,
    upper: f32,
    lower: f32,
) -> (Vec3, Vec3) {
    let delta = target - root;
    let direction = delta.normalize_or(Vec3::NEG_Y);
    let distance = delta
        .length()
        .clamp((upper - lower).abs() + 0.0001, upper + lower - 0.0001);
    let endpoint = root + direction * distance;
    let along = (upper * upper + distance * distance - lower * lower) / (2.0 * distance);
    let perpendicular =
        (pole - direction * pole.dot(direction)).normalize_or(direction.any_orthonormal_vector());
    let bend = (upper * upper - along * along).max(0.0).sqrt();
    (root + direction * along + perpendicular * bend, endpoint)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_program_preserves_limb_lengths_and_replays() {
        let mut endpoints = std::collections::HashSet::new();
        for seed in 0..1024 {
            for seated in [false, true] {
                let pose = if seated {
                    HumanPoseKind::SeatedWorking
                } else {
                    HumanPoseKind::StandingRelaxed
                };
                let program = PoseProgram::sample(seed, pose);
                let joints = program.solve(1.75, 1.0, 0.43, seated);
                assert_eq!(joints, program.solve(1.75, 1.0, 0.43, seated));
                for i in [5, 9] {
                    assert!((joints[i].distance(joints[i + 1]) - 0.29).abs() < 1e-5);
                    assert!((joints[i + 1].distance(joints[i + 2]) - 0.26).abs() < 1e-5);
                }
                for i in [13, 17] {
                    assert!((joints[i].distance(joints[i + 1]) - 0.43).abs() < 1e-5);
                    assert!((joints[i + 1].distance(joints[i + 2]) - 0.43).abs() < 1e-5);
                    assert!((joints[i + 2].y - 0.105).abs() < 1e-5);
                }
                endpoints.insert(joints[7].to_array().map(f32::to_bits));
            }
        }
        assert_eq!(endpoints.len(), 2048);
    }
}
