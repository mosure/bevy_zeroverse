use bevy::{camera::primitives::Aabb, prelude::*};
use bevy_args::{Deserialize, Serialize, ValueEnum};
use rand::Rng;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{app::BevyZeroverseConfig, camera::EditorCameraGizmoConfigGroup, ovoxel::OvoxelRegion};

// TODO: cornell box room scene
pub mod cornell_cube;
pub mod human;
pub mod lighting;
pub mod object;
pub mod procedural_indoor;
pub mod room;
pub mod semantic_room;

pub struct ZeroverseScenePlugin;

impl Plugin for ZeroverseScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<RegenerateSceneEvent>();
        app.add_message::<SceneLoadedEvent>();

        app.init_resource::<GlobalRotationAugment>();
        app.register_type::<GlobalRotationAugment>();

        app.init_resource::<ZeroverseSceneSettings>();
        app.register_type::<ZeroverseSceneSettings>();

        app.register_type::<SceneAabb>();
        app.register_type::<SceneAabbNode>();

        app.add_plugins((
            semantic_room::ZeroverseSemanticRoomPlugin,
            cornell_cube::ZeroverseCornellCubePlugin,
            human::ZeroverseHumanPlugin,
            lighting::ZeroverseLightingPlugin,
            object::ZeroverseObjectPlugin,
            room::ZeroverseRoomPlugin,
            procedural_indoor::ProceduralIndoorPlugin,
        ));

        app.add_systems(Update, regenerate_rotation_augment);
        app.add_systems(
            PostUpdate,
            (create_scene_aabb, draw_scene_aabb)
                .chain()
                .after(bevy::transform::TransformSystems::Propagate)
                .after(bevy::camera::visibility::VisibilitySystems::CalculateBounds)
                .before(crate::render::position::apply_position_material),
        );
        app.add_systems(
            PostUpdate,
            rotation_augment.before(bevy::transform::TransformSystems::Propagate),
        );
    }
}

#[derive(Component, Debug, Reflect)]
pub struct ZeroverseScene;

#[derive(Component, Debug, Reflect)]
#[require(Transform, Visibility)]
pub struct ZeroverseSceneRoot;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, Reflect, ValueEnum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum ZeroverseSceneType {
    CornellCube,
    Custom,
    Human,
    #[default]
    Object,
    SemanticRoom,
    Room,
    /// Metric furnished interiors with reproducible generation and AnnyBody people.
    #[value(alias = "procedural_indoor")]
    ProceduralIndoor,
}

#[derive(Resource, Debug, Default, Reflect, Clone)]
#[reflect(Resource)]
pub struct ZeroverseSceneSettings {
    pub num_cameras: usize,
    pub rotation_augmentation: bool,
    pub scene_type: ZeroverseSceneType,
    pub max_camera_radius: f32,
}

#[derive(Event, Message)]
pub struct RegenerateSceneEvent;

#[derive(Event, Message)]
pub struct SceneLoadedEvent;

#[derive(Component, Debug, Reflect)]
pub struct SceneAabbNode;

#[derive(Component, Debug, Reflect)]
/// World-space reconstruction bounds. Indoor scenes use their primary-room
/// O-voxel region, including its structural shell, even when voxel export is off.
pub struct SceneAabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl SceneAabb {
    fn new() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
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

        Transform::from_translation(center) * Transform::from_scale(size)
    }
}

pub(crate) fn create_scene_aabb(
    mut commands: Commands,
    scene_instances: Query<(Entity, &GlobalTransform, Option<&OvoxelRegion>), With<SceneAabbNode>>,
    children: Query<&Children>,
    bounding_boxes: Query<(&Aabb, &GlobalTransform)>,
) {
    for (entity, transform, region) in &scene_instances {
        if let Some(region) = region {
            // Use exactly the voxelizer's transformed crop, not the union of
            // render meshes: shared slabs and exterior context extend beyond it.
            let [min, max] = region.world_bounds(transform.affine());
            commands.entity(entity).insert(SceneAabb {
                min: min.into(),
                max: max.into(),
            });
            continue;
        }
        let mut scene_aabb = SceneAabb::new();

        let mut merged_any = false;
        for (bb, tf) in children
            .iter_descendants(entity)
            .filter_map(|child| bounding_boxes.get(child).ok())
        {
            scene_aabb.merge_aabb(bb, tf);
            merged_any = true;
        }

        if merged_any {
            commands.entity(entity).insert(scene_aabb);
        }
    }
}

fn draw_scene_aabb(
    args: Res<BevyZeroverseConfig>,
    scene_instances: Query<&SceneAabb>,
    mut gizmos: Gizmos<EditorCameraGizmoConfigGroup>,
) {
    if !args.draw_obb_gizmo {
        return;
    }

    let color = Color::srgba(0.0, 1.0, 1.0, args.gizmos_alpha);

    for aabb in &scene_instances {
        gizmos.cube(Transform::from(aabb), color);
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
        (Entity, &mut Transform),
        (With<RotationAugment>, Without<RotationAugmented>),
    >,
    global_rotation: Res<GlobalRotationAugment>,
) {
    for (entity, mut transform) in to_augment.iter_mut() {
        commands
            .entity(entity)
            .remove::<RotationAugment>()
            .insert(RotationAugmented);

        transform.rotation *= global_rotation.rotation;
    }
}

fn regenerate_rotation_augment(
    mut global_rotation: ResMut<GlobalRotationAugment>,
    mut regenerate_events: MessageReader<RegenerateSceneEvent>,
    scene_settings: Res<ZeroverseSceneSettings>,
) {
    if regenerate_events.is_empty() {
        return;
    }
    regenerate_events.clear();

    if scene_settings.rotation_augmentation {
        let mut rng = rand::rng();
        let random_rotation =
            Quat::from_rotation_y(rng.random_range(0.0..std::f32::consts::PI * 2.0));
        global_rotation.rotation = random_rotation;
    } else {
        global_rotation.rotation = Quat::IDENTITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{ecs::system::RunSystemOnce, MinimalPlugins};

    #[test]
    fn reconstruction_aabb_ignores_context_and_tracks_transformed_region_without_voxel_export() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        let root = app
            .world_mut()
            .spawn((
                SceneAabbNode,
                GlobalTransform::from(Transform {
                    translation: Vec3::new(10., 2., -7.),
                    rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                    scale: Vec3::new(1.5, 2., 0.5),
                }),
                OvoxelRegion {
                    min: Vec3::new(-2., -0.2, -4.),
                    max: Vec3::new(3., 3.1, 5.),
                },
            ))
            .id();
        // A shared architecture slab and a remote backdrop would both enlarge
        // a union of descendant mesh bounds. Neither can define the crop.
        for (half_size, offset) in [(20., Vec3::ZERO), (5., Vec3::splat(100.))] {
            app.world_mut().spawn((
                Aabb::from_min_max(Vec3::splat(-half_size), Vec3::splat(half_size)),
                GlobalTransform::from_translation(offset),
                ChildOf(root),
            ));
        }
        app.world_mut().run_system_once(create_scene_aabb).unwrap();
        let aabb = app.world().get::<SceneAabb>(root).unwrap();
        assert!(aabb.min.abs_diff_eq(Vec3::new(8., 1.6, -11.5), 1e-5));
        assert!(aabb.max.abs_diff_eq(Vec3::new(12.5, 8.2, -4.), 1e-5));
        assert!(app
            .world()
            .get::<crate::ovoxel::OvoxelExport>(root)
            .is_none());

        // Replacing the crop/transform must not retain bounds from a prior room.
        app.world_mut().entity_mut(root).insert((
            GlobalTransform::IDENTITY,
            OvoxelRegion {
                min: Vec3::splat(-1.),
                max: Vec3::ONE,
            },
        ));
        app.world_mut().run_system_once(create_scene_aabb).unwrap();
        let aabb = app.world().get::<SceneAabb>(root).unwrap();
        assert_eq!(aabb.min, Vec3::splat(-1.));
        assert_eq!(aabb.max, Vec3::ONE);
    }

    #[test]
    fn create_scene_aabb_merges_child_bounds() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);

        let root = app
            .world_mut()
            .spawn((
                ZeroverseSceneRoot,
                SceneAabbNode,
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
            ))
            .id();

        let child = app
            .world_mut()
            .spawn((
                Aabb::from_min_max(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(4.0, 5.0, 6.0)),
                GlobalTransform::from_translation(Vec3::new(2.0, 0.0, -1.0)),
            ))
            .id();

        app.world_mut().entity_mut(root).add_child(child);

        let _ = app.world_mut().run_system_once(super::create_scene_aabb);

        let scene_aabb = app
            .world()
            .entity(root)
            .get::<SceneAabb>()
            .expect("SceneAabb should be computed for root");

        assert_eq!(scene_aabb.min, Vec3::new(1.0, -2.0, -4.0));
        assert_eq!(scene_aabb.max, Vec3::new(6.0, 5.0, 5.0));
    }
}
