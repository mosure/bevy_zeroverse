use bevy::{
    prelude::*,
    render::primitives::Aabb,
};
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};
use rand::Rng;

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

        app.init_resource::<GlobalRotationAugment>();
        app.register_type::<GlobalRotationAugment>();

        app.init_resource::<ZeroverseSceneSettings>();
        app.register_type::<ZeroverseSceneSettings>();

        app.register_type::<SceneAabb>();

        app.add_plugins((
            cornell_cube::ZeroverseCornellCubePlugin,
            lighting::ZeroverseLightingPlugin,
            object::ZeroverseObjectPlugin,
            room::ZeroverseRoomPlugin,
        ));

        app.add_systems(
            Update,
            (
                create_scene_aabb,
                draw_scene_aabb,
                regenerate_rotation_augment,
            ),
        );
        app.add_systems(PostUpdate, rotation_augment);
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
    pub rotation_augmentation: bool,
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
    if !args.gizmos {
        return;
    }

    for aabb in &scene_instances {
        let color = Color::srgb(0.0, 1.0, 1.0);
        gizmos.cuboid(Transform::from(aabb), color);
    }
}


#[derive(Debug, Component, Clone, Reflect)]
pub struct RotationAugment;

#[derive(Debug, Component, Clone, Reflect)]
pub struct RotationAugmented;

#[derive(Resource, Debug, Default, Reflect)]
pub struct GlobalRotationAugment {
    pub rotation: Quat,
}

#[allow(clippy::type_complexity)]
fn rotation_augment(
    mut commands: Commands,
    mut to_augment: Query<
        (
            Entity,
            &mut Transform,
        ),
        (
            With<RotationAugment>,
            Without<RotationAugmented>
        ),
    >,
    global_rotation: Res<GlobalRotationAugment>,
) {
    for (entity, mut transform) in to_augment.iter_mut() {
        commands.entity(entity)
            .remove::<RotationAugment>()
            .insert(RotationAugmented);

        transform.rotation *= global_rotation.rotation;
    }
}

fn regenerate_rotation_augment(
    mut global_rotation: ResMut<GlobalRotationAugment>,
    mut regenerate_events: EventReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    if scene_settings.rotation_augmentation {
        let mut rng = rand::thread_rng();
        let random_rotation = Quat::from_rotation_y(rng.gen_range(0.0..std::f32::consts::PI * 2.0));
        global_rotation.rotation = random_rotation;
    } else {
        global_rotation.rotation = Quat::IDENTITY;
    }
}
