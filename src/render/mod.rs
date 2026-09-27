use bevy::{
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    post_process::bloom::Bloom,
    prelude::*,
    render::{render_resource::Face, view::Msaa},
};
use bevy_args::{Deserialize, Serialize, ValueEnum};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::primitive::process_primitives;

mod annotation_material;
pub mod color;
pub mod depth;
#[cfg(not(target_arch = "wasm32"))]
pub mod ground_truth;
pub mod normal;
pub mod optical_flow;
pub mod position;
pub mod residency;
pub mod semantic;

/// Upload through queue staging instead of CPU-writing a mapped device-local
/// allocation. Large mapped-at-creation copies can stall on discrete GPUs.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn upload_buffer(
    device: &bevy::render::renderer::RenderDevice,
    queue: &bevy::render::renderer::RenderQueue,
    descriptor: &bevy::render::render_resource::BufferInitDescriptor<'_>,
) -> bevy::render::render_resource::Buffer {
    use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
    // All callers upload aligned shader structs, vertices or u32 indices.
    assert_eq!(descriptor.contents.len() % 4, 0);
    let buffer = device.create_buffer(&BufferDescriptor {
        label: descriptor.label,
        size: descriptor.contents.len().max(4) as u64,
        usage: descriptor.usage | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !descriptor.contents.is_empty() {
        queue.write_buffer(&buffer, 0, descriptor.contents);
    }
    buffer
}

#[derive(
    Debug, Default, Clone, PartialEq, Serialize, Deserialize, Reflect, Resource, ValueEnum,
)]
#[reflect(Resource)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum RenderMode {
    #[default]
    Color,
    Depth,
    MotionVectors,
    Normal,
    OpticalFlow,
    Position,
    Semantic,
}

impl RenderMode {
    pub fn is_flow(&self) -> bool {
        matches!(self, Self::OpticalFlow | Self::MotionVectors)
    }

    pub fn bloom(&self) -> Option<Bloom> {
        match self {
            RenderMode::Color => Bloom::default().into(),
            _ => None,
        }
    }

    pub fn dither(&self) -> DebandDither {
        match self {
            RenderMode::Color => DebandDither::default(),
            _ => DebandDither::Disabled,
        }
    }

    pub fn msaa(&self) -> Msaa {
        Msaa::Off
    }

    pub fn tonemapping(&self) -> Tonemapping {
        match self {
            RenderMode::Color => Tonemapping::TonyMcMapface,
            _ => Tonemapping::None,
        }
    }
}

#[derive(Debug, Default)]
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderMode>();
        app.register_type::<RenderMode>();

        app.add_plugins(depth::DepthPlugin);
        app.add_plugins(normal::NormalPlugin);
        app.add_plugins(optical_flow::OpticalFlowPlugin);
        app.add_plugins(position::PositionPlugin);
        app.add_plugins(semantic::SemanticPlugin);
        app.add_plugins(residency::RenderResidencyPlugin);
        #[cfg(not(target_arch = "wasm32"))]
        app.add_plugins(ground_truth::GroundTruthPlugin);

        // TODO: add wireframe depth, pbr disable, normals
        app.add_systems(
            Update,
            (
                apply_render_modes,
                auto_disable_pbr_material::<depth::Depth>,
                auto_disable_pbr_material::<normal::Normal>,
                auto_disable_pbr_material::<optical_flow::OpticalFlow>,
                auto_disable_pbr_material::<position::Position>,
                auto_disable_pbr_material::<semantic::Semantic>,
                enable_pbr_material,
            )
                .chain()
                .after(process_primitives),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish(&self, app: &mut App) {
        // GPU clustering appends lights with atomics, so floating point lighting
        // accumulation can vary between identical captures. Indoor datasets have
        // few lights; CPU clustering preserves exact replay without changing the
        // GPU shading, shadow, GI, or asynchronous readback paths.
        if !app
            .world()
            .get_resource::<crate::app::BevyZeroverseConfig>()
            .is_some_and(|config| config.image_copiers)
        {
            return;
        }
        if let Some(mut settings) = app
            .world_mut()
            .get_resource_mut::<bevy::light::cluster::GlobalClusterSettings>()
        {
            settings.gpu_clustering = None;
        }
    }
}

#[derive(Component, Default, Debug, Reflect)]
pub struct DisabledPbrMaterial {
    #[reflect(ignore)]
    pub cull_mode: Option<Face>,
    pub double_sided: bool,
    pub material: Handle<StandardMaterial>,
}

impl DisabledPbrMaterial {
    pub(crate) fn annotation_key(&self) -> Vec<u32> {
        vec![
            match self.cull_mode {
                None => 0,
                Some(Face::Front) => 1,
                Some(Face::Back) => 2,
            },
            self.double_sided as u32,
        ]
    }
}

#[derive(Component, Default, Debug, Reflect)]
pub struct EnablePbrMaterial;

#[allow(clippy::type_complexity)]
pub fn auto_disable_pbr_material<T: Component>(
    mut commands: Commands,
    mut disabled_materials: Query<
        (Entity, &MeshMaterial3d<StandardMaterial>),
        (With<T>, Without<DisabledPbrMaterial>),
    >,
    standard_materials: Res<Assets<StandardMaterial>>,
) {
    for (entity, disabled_material_handle) in disabled_materials.iter_mut() {
        let disabled_material = standard_materials.get(&disabled_material_handle.0).unwrap();

        commands
            .entity(entity)
            .insert(DisabledPbrMaterial {
                cull_mode: disabled_material.cull_mode,
                double_sided: disabled_material.double_sided,
                material: disabled_material_handle.0.clone(),
            })
            .remove::<EnablePbrMaterial>()
            .remove::<MeshMaterial3d<StandardMaterial>>();
    }
}

pub(crate) fn enable_pbr_material(
    mut commands: Commands,
    mut enabled_materials: Query<(Entity, &DisabledPbrMaterial), With<EnablePbrMaterial>>,
) {
    for (entity, disabled_material) in enabled_materials.iter_mut() {
        commands
            .entity(entity)
            .insert(MeshMaterial3d(disabled_material.material.clone()))
            .remove::<DisabledPbrMaterial>()
            .remove::<EnablePbrMaterial>();
    }
}

pub(crate) fn apply_render_modes(
    mut commands: Commands,
    render_mode: Res<RenderMode>,
    meshes: Query<Entity, With<Mesh3d>>,
    new_meshes: Query<Entity, Added<Mesh3d>>,
) {
    let insert_render_mode_flag = |commands: &mut Commands, entity: Entity| match *render_mode {
        RenderMode::Color => {
            commands.entity(entity).insert(EnablePbrMaterial);
        }
        RenderMode::Depth => {
            commands.entity(entity).insert(depth::Depth);
        }
        RenderMode::Normal => {
            commands.entity(entity).insert(normal::Normal);
        }
        RenderMode::OpticalFlow | RenderMode::MotionVectors => {
            commands.entity(entity).insert(optical_flow::OpticalFlow);
        }
        RenderMode::Position => {
            commands.entity(entity).insert(position::Position);
        }
        RenderMode::Semantic => {
            commands.entity(entity).insert(semantic::Semantic);
        }
    };

    if render_mode.is_changed() {
        for entity in meshes.iter() {
            commands
                .entity(entity)
                .remove::<depth::Depth>()
                .remove::<normal::Normal>()
                .remove::<optical_flow::OpticalFlow>()
                .remove::<position::Position>()
                .remove::<semantic::Semantic>();

            insert_render_mode_flag(&mut commands, entity);
        }
    }

    for entity in new_meshes.iter() {
        insert_render_mode_flag(&mut commands, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{math::primitives::Sphere, MinimalPlugins};

    #[test]
    fn annotation_materials_are_specialized_in_the_frame_they_are_created() {
        use bevy::{asset::AssetPlugin, pbr::EntitiesNeedingSpecialization, shader::Shader};
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()));
        app.init_asset::<Mesh>();
        app.init_asset::<StandardMaterial>();
        app.init_asset::<Shader>();
        app.insert_resource(crate::app::BevyZeroverseConfig::default());
        app.add_plugins((
            depth::DepthPlugin,
            normal::NormalPlugin,
            position::PositionPlugin,
            semantic::SemanticPlugin,
            optical_flow::OpticalFlowPlugin,
        ));
        app.world_mut().spawn(crate::scene::SceneAabb {
            min: Vec3::splat(-1.0),
            max: Vec3::ONE,
        });
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let pbr = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let mut previous = Vec::new();
        for _ in 0..3 {
            for entity in previous.drain(..) {
                app.world_mut().despawn(entity);
            }
            let mut spawn = || {
                app.world_mut()
                    .spawn((
                        Mesh3d(mesh.clone()),
                        DisabledPbrMaterial {
                            cull_mode: None,
                            double_sided: true,
                            material: pbr.clone(),
                        },
                    ))
                    .id()
            };
            let entities = [spawn(), spawn(), spawn(), spawn(), spawn()];
            app.world_mut().entity_mut(entities[0]).insert(depth::Depth);
            app.world_mut()
                .entity_mut(entities[1])
                .insert(normal::Normal);
            app.world_mut()
                .entity_mut(entities[2])
                .insert(position::Position);
            app.world_mut()
                .entity_mut(entities[3])
                .insert((semantic::Semantic, semantic::SemanticLabel::Chair));
            app.world_mut()
                .entity_mut(entities[4])
                .insert(optical_flow::OpticalFlow);
            app.update();
            macro_rules! tracked {
                ($material:ty, $index:expr) => {
                    assert!(
                        app.world()
                            .resource::<EntitiesNeedingSpecialization<$material>>()
                            .changed
                            .contains(&entities[$index]),
                        "annotation material reached extraction before specialization tracking"
                    );
                };
            }
            tracked!(depth::DepthMaterial, 0);
            tracked!(normal::NormalMaterial, 1);
            tracked!(position::PositionMaterial, 2);
            tracked!(semantic::SemanticMaterial, 3);
            tracked!(optical_flow::OpticalFlowMaterial, 4);
            // Regeneration must reuse these five equivalent materials rather
            // than allocating one per mesh, which destroys instancing.
            assert_eq!(
                app.world().resource::<Assets<depth::DepthMaterial>>().len(),
                1
            );
            assert_eq!(
                app.world()
                    .resource::<Assets<normal::NormalMaterial>>()
                    .len(),
                1
            );
            assert_eq!(
                app.world()
                    .resource::<Assets<position::PositionMaterial>>()
                    .len(),
                1
            );
            assert_eq!(
                app.world()
                    .resource::<Assets<semantic::SemanticMaterial>>()
                    .len(),
                1
            );
            assert_eq!(
                app.world()
                    .resource::<Assets<optical_flow::OpticalFlowMaterial>>()
                    .len(),
                1
            );
            previous.extend(entities);
        }
    }

    #[test]
    fn color_to_normal_back_to_color_restores_materials_same_frame() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);

        app.insert_resource(RenderMode::Color);
        app.insert_resource(Assets::<Mesh>::default());
        app.insert_resource(Assets::<StandardMaterial>::default());
        app.insert_resource(Assets::<normal::NormalMaterial>::default());

        app.add_systems(
            Update,
            (
                apply_render_modes,
                auto_disable_pbr_material::<normal::Normal>,
                enable_pbr_material,
            )
                .chain(),
        );
        app.add_systems(PostUpdate, normal::apply_normal_material);

        let mesh_handle = {
            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
            meshes.add(Mesh::from(Sphere::default()))
        };
        let material_handle = {
            let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            materials.add(StandardMaterial::default())
        };

        let entity = app
            .world_mut()
            .spawn((Mesh3d(mesh_handle), MeshMaterial3d(material_handle)))
            .id();

        // Enter normal render mode.
        {
            let mut mode = app.world_mut().resource_mut::<RenderMode>();
            *mode = RenderMode::Normal;
        }
        app.update();

        {
            let world = app.world();
            let entity_ref = world.entity(entity);
            assert!(entity_ref.contains::<normal::Normal>());
            assert!(entity_ref.contains::<DisabledPbrMaterial>());
            assert!(entity_ref.contains::<MeshMaterial3d<normal::NormalMaterial>>());
            assert!(!entity_ref.contains::<MeshMaterial3d<StandardMaterial>>());
        }

        // Return to color render mode and ensure the standard material comes back immediately.
        {
            let mut mode = app.world_mut().resource_mut::<RenderMode>();
            *mode = RenderMode::Color;
        }
        app.update();

        let world = app.world();
        let entity_ref = world.entity(entity);
        assert!(!entity_ref.contains::<normal::Normal>());
        assert!(!entity_ref.contains::<DisabledPbrMaterial>());
        assert!(entity_ref.contains::<MeshMaterial3d<StandardMaterial>>());
        assert!(!entity_ref.contains::<MeshMaterial3d<normal::NormalMaterial>>());
    }
}
