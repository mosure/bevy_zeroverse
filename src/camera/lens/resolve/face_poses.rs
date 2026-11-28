//! Pose agreement between a logical lens and the physical views extracted
//! for this frame. Multiple Bevy cameras may target the same image.
use bevy::{asset::AssetId, prelude::*};
use std::collections::{hash_map::Entry, HashMap};

#[derive(Default)]
pub(super) struct FacePoses(HashMap<AssetId<Image>, (Mat4, Vec<Mat4>)>);

impl FacePoses {
    pub(super) fn rebuild(&mut self, views: impl Iterator<Item = (AssetId<Image>, Mat4)>) {
        self.0.clear();
        for (target, pose) in views {
            match self.0.entry(target) {
                Entry::Vacant(entry) => {
                    entry.insert((pose, Vec::new()));
                }
                Entry::Occupied(mut entry) => entry.get_mut().1.push(pose),
            }
        }
    }

    pub(super) fn matches(&self, target: AssetId<Image>, expected: Mat4) -> bool {
        self.0.get(&target).is_some_and(|(first, others)| {
            first.abs_diff_eq(expected, 1e-5)
                || others.iter().any(|pose| pose.abs_diff_eq(expected, 1e-5))
        })
    }
}

/// Small rigs can scan the few extracted cameras directly. Both lookup paths
/// use identical orientations, pose tolerances and alias handling.
pub(super) fn ready(
    faces: &[Handle<Image>; 6],
    world: Mat4,
    rotations: [Quat; 6],
    mut matches: impl FnMut(AssetId<Image>, Mat4) -> bool,
) -> bool {
    faces
        .iter()
        .zip(rotations)
        .enumerate()
        .all(|(i, (face, rotation))| {
            (i > 0 && face == &faces[0]) || matches(face.id(), world * Mat4::from_quat(rotation))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_current_extracted_targets_with_matching_poses_complete_a_lens() {
        let mut images = Assets::<Image>::default();
        let a = images.add(Image::default());
        let b = images.add(Image::default());
        let faces = [
            a.clone(),
            b.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
        ];
        let rotations = crate::camera::lens::runtime::orientations();
        let world =
            Mat4::from_rotation_translation(Quat::from_rotation_y(0.35), Vec3::new(1., 2., 3.));
        let first = world * Mat4::from_quat(rotations[0]);
        let second = world * Mat4::from_quat(rotations[1]);
        let mut index = FacePoses::default();
        index.rebuild([(a.id(), first), (b.id(), first)].into_iter());
        assert!(!ready(&faces, world, rotations, |id, pose| index.matches(id, pose)));
        for poses in [[first, second], [second, first]] {
            index.rebuild([(a.id(), first), (b.id(), poses[0]), (b.id(), poses[1])].into_iter());
            assert!(ready(&faces, world, rotations, |id, pose| index.matches(id, pose)));
        }
        index.rebuild([(a.id(), first)].into_iter());
        assert!(!ready(&faces, world, rotations, |id, pose| index.matches(id, pose)));
        index.rebuild(std::iter::empty());
        assert!(!ready(&faces, world, rotations, |id, pose| index.matches(id, pose)));
    }
}
