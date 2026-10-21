//! Calibrated pose labels derived from the captured skeleton, never a manifest
//! proxy or RGB inference. Works on CPU samples and GPU samples' CPU metadata.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::sample::Sample;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProjectedJoint {
    pub world_position: [f32; 3],
    /// XYZW. Indoor limbs encode the incoming bone's +Y direction (no twist);
    /// the head encodes its rendered frame. Legacy human rigs use bone frames.
    pub world_rotation: [f32; 4],
    /// Metres; X right, Y up, -Z forward.
    pub view_position: [f32; 3],
    /// X right / Y down, top-left origin, pixel centres at half integers.
    /// None for points behind the camera. Off-image coordinates are retained.
    pub pixel: Option<[f32; 2]>,
    pub axial_depth: f32,
    /// Frustum inclusion only. Joint centres inside skin are not surface-depth
    /// visible points; this flag deliberately does not claim occlusion visibility.
    pub in_frustum: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProjectedHumanPose {
    /// Same ID as Sample::object_obbs for indoor people; None in old samples.
    pub instance_id: Option<i64>,
    pub joints: Vec<ProjectedJoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ViewHumanPoses {
    pub schema_version: u32,
    pub view_index: usize,
    pub camera_index: usize,
    pub timestep_index: usize,
    pub time: f32,
    pub time_seconds: Option<f32>,
    pub trajectory_progress: Option<f32>,
    pub bone_names: Vec<String>,
    pub bone_parents: Vec<i64>,
    pub people: Vec<ProjectedHumanPose>,
}

impl Sample {
    /// Project the captured pose for exactly this view's timestep. Does not read
    /// GPU image bytes, initialize models, infer visibility, or mutate the sample.
    /// Missing calibration and inconsistent IDs/rig/timestep data are errors.
    pub fn human_poses_for_view(&self, view_index: usize) -> Result<ViewHumanPoses, String> {
        let view = self
            .views
            .get(view_index)
            .ok_or("pose view index out of range")?;
        let cameras = self.view_dim as usize;
        if cameras == 0 || !self.views.len().is_multiple_of(cameras) {
            return Err("invalid camera/timestep dimensions".into());
        }
        let step = view_index / cameras;
        let steps = self.views.len() / cameras;
        let poses = if self.human_pose_steps.is_empty() && steps == 1 {
            &self.human_poses
        } else if self.human_pose_steps.len() == steps {
            &self.human_pose_steps[step]
        } else {
            return Err("pose timesteps do not match captured views".into());
        };
        let calibration = view
            .calibration
            .as_ref()
            .ok_or("pose projection requires captured calibration")?;
        calibration.validate()?;
        let world_from_view = Mat4::from_cols_array_2d(&view.world_from_view);
        if !world_from_view.is_finite()
            || world_from_view.determinant().abs() < 1e-8
            || !view.near.is_finite()
            || !view.far.is_finite()
            || view.near <= 0.
            || view.far <= view.near
        {
            return Err("invalid captured pose camera transform or clipping planes".into());
        }
        let view_from_world = world_from_view.inverse();
        let count = self.human_bone_names.len();
        if (!poses.is_empty() && count == 0)
            || self.human_bone_parents.len() != count
            || self
                .human_bone_parents
                .iter()
                .enumerate()
                .any(|(i, &p)| p < -1 || p >= i as i64)
            || (!self.human_instance_ids.is_empty() && self.human_instance_ids.len() != poses.len())
            || self
                .human_instance_ids
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != self.human_instance_ids.len()
        {
            return Err("invalid human identity or skeleton hierarchy".into());
        }
        let people = poses
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                if pose.bone_positions.len() != count || pose.bone_rotations.len() != count {
                    return Err("pose joint arrays do not match skeleton".into());
                }
                let joints = pose
                    .bone_positions
                    .iter()
                    .zip(&pose.bone_rotations)
                    .map(|(&p, &q)| {
                        let p = Vec3::from_array(p);
                        let rotation = Quat::from_array(q);
                        if !p.is_finite()
                            || !rotation.is_finite()
                            || (rotation.length_squared() - 1.).abs() > 0.01
                        {
                            return Err("invalid pose joint position or quaternion".into());
                        }
                        let local = view_from_world.transform_point3(p);
                        let pixel = calibration
                            .project(local.to_array())
                            .filter(|p| p.iter().all(|v| v.is_finite()));
                        let depth = calibration.depth(local.to_array());
                        Ok(ProjectedJoint {
                            world_position: p.to_array(),
                            world_rotation: q,
                            view_position: local.to_array(),
                            pixel,
                            axial_depth: -local.z,
                            in_frustum: depth >= view.near
                                && depth <= view.far
                                && pixel.is_some_and(|p| calibration.in_image(p)),
                        })
                    })
                    .collect::<Result<_, String>>()?;
                Ok(ProjectedHumanPose {
                    instance_id: self.human_instance_ids.get(index).copied(),
                    joints,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(ViewHumanPoses {
            schema_version: 1,
            view_index,
            camera_index: view_index % cameras,
            timestep_index: step,
            time: view.time,
            time_seconds: view.time_seconds,
            trajectory_progress: view.trajectory_progress,
            bone_names: self.human_bone_names.clone(),
            bone_parents: self.human_bone_parents.clone(),
            people,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{HumanPoseSample, View};
    use bevy_zeroverse_capture::calibration::CameraCalibration;

    fn sample() -> Sample {
        let view = View {
            calibration: Some(CameraCalibration::centered_pinhole(640, 400, 1., 1.6).unwrap()),
            world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
            near: 0.1,
            far: 10.,
            ..default()
        };
        let pose = |z| HumanPoseSample {
            bone_positions: vec![[0., 0., z], [0.2, 0.1, z]],
            bone_rotations: vec![Quat::IDENTITY.to_array(); 2],
        };
        Sample {
            views: vec![
                view.clone(),
                view.clone(),
                View {
                    time: 0.5,
                    time_seconds: Some(2.),
                    ..view.clone()
                },
                view,
            ],
            view_dim: 2,
            human_instance_ids: vec![73],
            human_bone_names: vec!["root".into(), "head".into()],
            human_bone_parents: vec![-1, 0],
            human_pose_steps: vec![vec![pose(-2.)], vec![pose(2.)]],
            ..default()
        }
    }

    #[test]
    fn projection_uses_exact_view_step_ids_axes_and_calibration() {
        let sample = sample();
        let a = sample.human_poses_for_view(1).unwrap();
        assert_eq!((a.camera_index, a.timestep_index), (1, 0));
        assert_eq!(a.people[0].instance_id, Some(73));
        let joints = &a.people[0].joints;
        assert_eq!(joints[0].pixel, Some([320., 200.]));
        assert!(joints[0].in_frustum);
        assert!(joints[1].pixel.unwrap()[0] > 320. && joints[1].pixel.unwrap()[1] < 200.);
        let b = sample.human_poses_for_view(2).unwrap();
        assert_eq!(
            (b.camera_index, b.timestep_index, b.time_seconds),
            (0, 1, Some(2.))
        );
        assert!(b.people[0]
            .joints
            .iter()
            .all(|p| p.pixel.is_none() && !p.in_frustum));
        let decoded: ViewHumanPoses =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(a, decoded);
    }

    #[test]
    fn camera_transform_offscreen_clipping_and_invalid_metadata_are_explicit() {
        let mut s = sample();
        s.views[0].world_from_view = Mat4::from_translation(Vec3::X * 5.).to_cols_array_2d();
        let joint = &s.human_poses_for_view(0).unwrap().people[0].joints[0];
        assert!(joint.pixel.unwrap()[0] < 0. && !joint.in_frustum);
        s.views[0].far = 1.;
        assert!(!s.human_poses_for_view(0).unwrap().people[0].joints[0].in_frustum);
        s.human_instance_ids.push(74);
        assert!(s.human_poses_for_view(0).is_err());
        s.human_instance_ids.clear();
        assert_eq!(
            s.human_poses_for_view(0).unwrap().people[0].instance_id,
            None
        );
        s.human_pose_steps.pop();
        assert!(s.human_poses_for_view(2).is_err());
        s.views[0].calibration = None;
        assert!(s.human_poses_for_view(0).is_err());
        assert!(s.human_poses_for_view(4).is_err());
    }
}
