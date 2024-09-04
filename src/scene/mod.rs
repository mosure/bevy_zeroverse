use bevy::{
    prelude::*,
    render::primitives::Aabb,
};
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{
    app::BevyZeroverseConfig,
    camera::EditorCameraGizmoConfigGroup,
};

// TODO: cornell box room scene
pub mod cornell_cube;
pub mod lighting;
pub mod object;
pub mod room;


pub struct ZeroverseScenePlugin;

impl Plugin for ZeroverseScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<RegenerateSceneEvent>();
        app.add_event::<SceneLoadedEvent>();

        app.init_resource::<ZeroverseSceneSettings>();
        app.register_type::<ZeroverseSceneSettings>();
        app.register_type::<SceneAabb>();

        app.add_plugins((
            cornell_cube::ZeroverseCornellCubePlugin,
            lighting::ZeroverseLightingPlugin,
            object::ZeroverseObjectPlugin,
            room::ZeroverseRoomPlugin,
        ));

        app.add_systems(PreUpdate, create_scene_aabb);
        app.add_systems(Update, draw_scene_aabb);
    }
}


#[derive(Component, Debug, Reflect)]
pub struct ZeroverseScene;

#[derive(Component, Debug, Reflect)]
pub struct ZeroverseSceneRoot;


#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    Reflect,
    ValueEnum,
)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum ZeroverseSceneType {
    CornellCube,
    #[default]
    Object,
    Room,
}

#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct ZeroverseSceneSettings {
    pub num_cameras: usize,
    pub scene_type: ZeroverseSceneType,
}


#[derive(Event)]
pub struct RegenerateSceneEvent;


#[derive(Event)]
pub struct SceneLoadedEvent;



#[derive(Component, Debug, Reflect)]
pub struct SceneAabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl SceneAabb {
    fn new(center: Vec3) -> Self {
        Self {
            min: center,
            max: center,
        }
    }

    fn merge_aabb(&mut self, aabb: &Aabb, global_transform: &GlobalTransform) {
        let min = aabb.min();
        let max = aabb.max();
        let corners = [
            global_transform.transform_point(Vec3::new(max.x, max.y, max.z)),
            global_transform.transform_point(Vec3::new(min.x, max.y, max.z)),
            global_transform.transform_point(Vec3::new(min.x, max.y, min.z)),
            global_transform.transform_point(Vec3::new(max.x, max.y, min.z)),
            global_transform.transform_point(Vec3::new(max.x, min.y, max.z)),
            global_transform.transform_point(Vec3::new(min.x, min.y, max.z)),
            global_transform.transform_point(Vec3::new(min.x, min.y, min.z)),
            global_transform.transform_point(Vec3::new(max.x, min.y, min.z)),
        ];

        for corner in corners {
            let gt = corner.cmpgt(self.max);
            let lt = corner.cmplt(self.min);

            if gt.x {
                self.max.x = corner.x;
            } else if lt.x {
                self.min.x = corner.x;
            }

            if gt.y {
                self.max.y = corner.y;
            } else if lt.y {
                self.min.y = corner.y;
            }

            if gt.z {
                self.max.z = corner.z;
            } else if lt.z {
                self.min.z = corner.z;
            }
        }
    }
}

impl From<&SceneAabb> for Transform {
    fn from(scene_aabb: &SceneAabb) -> Transform {
        let min = scene_aabb.min;
        let max = scene_aabb.max;

        let center = (min + max) / 2.0;
        let size = max - min;

        Transform::from_translation(center)
            * Transform::from_scale(size)
    }
}


fn create_scene_aabb(
    mut commands: Commands,
    scene_instances: Query<
        (
            Entity,
            &ZeroverseSceneRoot,
            &GlobalTransform,
        ),
        Without<SceneAabb>,
    >,
    children: Query<&Children>,
    bounding_boxes: Query<(&Aabb, &GlobalTransform)>,
) {
    for (entity, _instance, global_transform) in scene_instances.iter() {
        let mut scene_aabb = SceneAabb::new(global_transform.translation());

        if children.iter_descendants(entity).count() == 0 {
            continue;
        }

        for child in children.iter_descendants(entity) {
            let Ok((bb, transform)) = bounding_boxes.get(child) else { continue };
            scene_aabb.merge_aabb(bb, transform);
        }

        commands.entity(entity).insert(scene_aabb);
    }
}


fn draw_scene_aabb(
    args: Res<BevyZeroverseConfig>,
    scene_instances: Query<&SceneAabb>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
) {
    if !args.editor {
        return;
    }

    for aabb in &scene_instances {
        let color = Color::srgb(0.0, 1.0, 1.0);
        gizmos.cuboid(Transform::from(aabb), color);
    }
}
