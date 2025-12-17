use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bevy::{
    camera::primitives::Aabb, gizmos::config::GizmoConfigStore, prelude::*,
    transform::TransformSystems,
};

use crate::{app::BevyZeroverseConfig, camera::EditorCameraGizmoConfigGroup, scene::SceneAabbNode};

type TrackedObbQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static Aabb,
        &'static GlobalTransform,
        Option<&'static Name>,
        Option<&'static ObbClass>,
    ),
    With<ObbTracked>,
>;

/// Marker for entities that should emit an oriented bounding box.
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct ObbTracked;

/// Optional override for the class name used when exporting or drawing an OBB.
#[derive(Component, Debug, Reflect)]
#[reflect(Component)]
pub struct ObbClass(pub String);

#[derive(Component, Debug, Reflect, Clone)]
#[reflect(Component)]
pub struct ObjectObb {
    pub center: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
    pub class_name: String,
}

pub struct ZeroverseObbPlugin;

impl Plugin for ZeroverseObbPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ObbTracked>();
        app.register_type::<ObbClass>();
        app.register_type::<ObjectObb>();

        app.add_systems(
            PostUpdate,
            (
                compute_object_obbs.after(TransformSystems::Propagate),
                draw_object_obbs
                    .after(compute_object_obbs)
                    .run_if(resource_exists::<GizmoConfigStore>),
            ),
        );
    }
}

fn compute_object_obbs(
    mut commands: Commands,
    parents: Query<&ChildOf>,
    scoped: Query<(), With<SceneAabbNode>>,
    tracked: TrackedObbQuery<'_, '_>,
) {
    for (entity, aabb, global_transform, name, class_override) in tracked.iter() {
        if !has_scene_scope(entity, &parents, &scoped) {
            continue;
        }

        let (min, max) = (Vec3::from(aabb.min()), Vec3::from(aabb.max()));
        let local_center = (min + max) * 0.5;
        let local_size = max - min;

        let (scale, rotation, _translation) = global_transform.to_scale_rotation_translation();
        let world_center = global_transform.transform_point(local_center);
        let world_scale = Vec3::new(
            scale.x.abs() * local_size.x,
            scale.y.abs() * local_size.y,
            scale.z.abs() * local_size.z,
        );

        let class_name = class_override
            .map(|c| c.0.clone())
            .or_else(|| name.map(|n| n.as_str().to_owned()))
            .unwrap_or_else(|| "unknown".to_string());

        commands.entity(entity).insert(ObjectObb {
            center: world_center,
            scale: world_scale,
            rotation,
            class_name,
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

fn draw_object_obbs(
    args: Res<BevyZeroverseConfig>,
    obbs: Query<&ObjectObb>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
) {
    if !args.gizmos || !args.draw_obb_gizmo {
        return;
    }

    for obb in obbs.iter() {
        let color = class_color(&obb.class_name, args.gizmos_alpha);
        let transform = Transform::from_translation(obb.center)
            .with_rotation(obb.rotation)
            .with_scale(obb.scale);
        gizmos.cuboid(transform, color);
    }
}

fn class_color(class_name: &str, alpha: f32) -> Color {
    let mut hasher = DefaultHasher::new();
    class_name.hash(&mut hasher);
    let hash = hasher.finish();

    // Map hash to a stable but bright-ish color.
    let r = 0.2 + ((hash & 0xFF) as f32 / 255.0) * 0.8;
    let g = 0.2 + (((hash >> 8) & 0xFF) as f32 / 255.0) * 0.8;
    let b = 0.2 + (((hash >> 16) & 0xFF) as f32 / 255.0) * 0.8;

    Color::srgba(r, g, b, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{ecs::system::RunSystemOnce, transform::TransformPlugin, MinimalPlugins};

    #[test]
    fn obb_respects_rotation_and_scale() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, ZeroverseObbPlugin));
        app.insert_resource(BevyZeroverseConfig::default());

        let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let scale = Vec3::new(1.0, 2.0, 3.0);

        let scope = app
            .world_mut()
            .spawn((
                SceneAabbNode,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        app.world_mut().spawn((
            ObbTracked,
            Aabb::from_min_max(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
            Transform::from_rotation(rotation).with_scale(scale),
            GlobalTransform::default(),
            ChildOf(scope),
        ));

        app.update(); // propagate transforms
        let _ = app.world_mut().run_system_once(super::compute_object_obbs);

        let obb = {
            let world = app.world_mut();
            world.query::<&ObjectObb>().single(&world).cloned().unwrap()
        };

        // rotation should be preserved
        assert!(
            (obb.rotation.to_euler(EulerRot::YXZ).0 - std::f32::consts::FRAC_PI_4).abs() < 1e-3
        );
        // scale should reflect local extents (unit cube scaled by provided scale)
        assert_eq!(obb.scale, scale);
    }

    #[test]
    fn obb_ignored_outside_scene_scope() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, TransformPlugin, ZeroverseObbPlugin));
        app.insert_resource(BevyZeroverseConfig::default());

        // Entity without SceneAabbNode ancestry should be ignored.
        app.world_mut().spawn((
            ObbTracked,
            Aabb::from_min_max(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            Transform::IDENTITY,
            GlobalTransform::default(),
        ));

        app.update();
        let _ = app.world_mut().run_system_once(super::compute_object_obbs);

        let has_obb = {
            let world = app.world_mut();
            world.query::<&ObjectObb>().iter(&world).next().is_some()
        };

        assert!(
            !has_obb,
            "Object outside SceneAabbNode should not produce an OBB"
        );
    }
}
