use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bevy::{
    gizmos::config::GizmoConfigStore, mesh::skinning::SkinnedMesh, prelude::*,
    transform::TransformSystems,
};
use bevy_burn_human::{BurnHumanAssets, BurnHumanInput};

use crate::{app::BevyZeroverseConfig, camera::PoseGizmoConfigGroup, scene::SceneAabbNode};

type TrackedPoseQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static SkinnedMesh), (With<PoseTracked>, With<BurnHumanInput>)>;

/// Marker for entities that should emit pose labels.
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct PoseTracked;

#[derive(Component, Debug, Reflect, Clone)]
#[reflect(Component)]
pub struct HumanPose {
    pub bone_positions: Vec<Vec3>,
    pub bone_rotations: Vec<Quat>,
}

pub struct ZeroversePosePlugin;

impl Plugin for ZeroversePosePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<PoseTracked>();
        app.register_type::<HumanPose>();

        app.add_systems(
            PostUpdate,
            (
                compute_human_poses.after(TransformSystems::Propagate),
                draw_human_poses
                    .after(compute_human_poses)
                    .run_if(resource_exists::<GizmoConfigStore>),
            ),
        );
    }
}

fn compute_human_poses(
    mut commands: Commands,
    parents: Query<&ChildOf>,
    scoped: Query<(), With<SceneAabbNode>>,
    humans: TrackedPoseQuery<'_, '_>,
    joint_transforms: Query<&GlobalTransform>,
) {
    for (entity, skinned) in humans.iter() {
        if !has_scene_scope(entity, &parents, &scoped) {
            continue;
        }

        let mut bone_positions = Vec::with_capacity(skinned.joints.len());
        let mut bone_rotations = Vec::with_capacity(skinned.joints.len());

        for joint in skinned.joints.iter().copied() {
            if let Ok(global) = joint_transforms.get(joint) {
                let (_scale, rotation, translation) = global.to_scale_rotation_translation();
                bone_positions.push(translation);
                bone_rotations.push(rotation);
            } else {
                bone_positions.push(Vec3::ZERO);
                bone_rotations.push(Quat::IDENTITY);
            }
        }

        commands.entity(entity).insert(HumanPose {
            bone_positions,
            bone_rotations,
        });
    }
}

fn has_scene_scope(
    mut entity: Entity,
    parents: &Query<&ChildOf>,
    scoped: &Query<(), With<SceneAabbNode>>,
) -> bool {
    // Traverse upwards until a SceneAabbNode is found.
    loop {
        if scoped.get(entity).is_ok() {
            return true;
        }

        let Ok(parent) = parents.get(entity) else {
            return false;
        };
        entity = parent.parent();
    }
}

fn draw_human_poses(
    args: Res<BevyZeroverseConfig>,
    assets: Option<Res<BurnHumanAssets>>,
    poses: Query<&HumanPose>,
    mut gizmos: Gizmos<PoseGizmoConfigGroup>,
) {
    if !args.gizmos || !args.draw_pose_gizmos {
        return;
    }

    let Some(assets) = assets.as_ref() else {
        return;
    };

    let bone_labels = &assets.body.metadata().metadata.bone_labels;
    let bone_parents = &assets.body.metadata().metadata.bone_parents;

    for pose in poses.iter() {
        if pose.bone_positions.is_empty() {
            continue;
        }

        for (idx, position) in pose.bone_positions.iter().enumerate() {
            let label = bone_labels
                .get(idx)
                .map(|s| s.as_str())
                .unwrap_or("bone");
            let color = bone_color(label, args.gizmos_alpha);
            gizmos.sphere(*position, 0.02, color);

            let parent = bone_parents.get(idx).copied().unwrap_or(-1);
            if parent < 0 {
                continue;
            }
            let parent_idx = parent as usize;
            if let Some(parent_pos) = pose.bone_positions.get(parent_idx) {
                gizmos.line(*parent_pos, *position, color);
            }
        }
    }
}

fn bone_color(bone_name: &str, alpha: f32) -> Color {
    let mut hasher = DefaultHasher::new();
    bone_name.hash(&mut hasher);
    let hash = hasher.finish();

    let r = 0.2 + ((hash & 0xFF) as f32 / 255.0) * 0.8;
    let g = 0.2 + (((hash >> 8) & 0xFF) as f32 / 255.0) * 0.8;
    let b = 0.2 + (((hash >> 16) & 0xFF) as f32 / 255.0) * 0.8;

    Color::srgba(r, g, b, alpha)
}
