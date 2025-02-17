use bevy::{
    prelude::*,
    render::render_resource::Face,
};
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::primitive::process_primitives;

pub mod depth;
pub mod normal;
pub mod optical_flow;
pub mod semantic;


#[derive(
    Debug,
    Default,
    Clone,
    PartialEq,
    Serialize,
    Deserialize,
    Reflect,
    Resource,
    ValueEnum,
)]
#[reflect(Resource)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum RenderMode {
    #[default]
    Color,
    Depth,
    Normal,
    OpticalFlow,
    Semantic,
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
        app.add_plugins(semantic::SemanticPlugin);

        // TODO: add wireframe depth, pbr disable, normals
        app.add_systems(
            Update,
            apply_render_modes.after(process_primitives),
        );
        app.add_systems(
            Update,
            (
                    auto_disable_pbr_material::<depth::Depth>,
                    auto_disable_pbr_material::<normal::Normal>,
                    auto_disable_pbr_material::<optical_flow::OpticalFlow>,
                    auto_disable_pbr_material::<semantic::Semantic>,
                    enable_pbr_material,
                )
                .after(apply_render_modes)
                .after(process_primitives)
        );
    }
}


#[derive(Component, Default, Debug, Reflect)]
pub struct DisabledPbrMaterial {
    #[reflect(ignore)]
    pub cull_mode: Option<Face>,
    pub double_sided: bool,
    pub material: Handle<StandardMaterial>,
}

#[derive(Component, Default, Debug, Reflect)]
pub struct EnablePbrMaterial;


#[allow(clippy::type_complexity)]
pub fn auto_disable_pbr_material<T: Component>(
    mut commands: Commands,
    mut disabled_materials: Query<
        (
            Entity,
            &MeshMaterial3d<StandardMaterial>,
        ),
        (With<T>, Without<DisabledPbrMaterial>),
    >,
    standard_materials: Res<Assets<StandardMaterial>>,
) {
    for (
        entity,
        disabled_material_handle,
    ) in disabled_materials.iter_mut() {
        let disabled_material = standard_materials.get(&disabled_material_handle.0).unwrap();

        commands.entity(entity)
            .insert(DisabledPbrMaterial {
                cull_mode: disabled_material.cull_mode,
                double_sided: disabled_material.double_sided,
                material: disabled_material_handle.0.clone(),
            })
            .remove::<EnablePbrMaterial>()
            .remove::<MeshMaterial3d<StandardMaterial>>();
    }
}

fn enable_pbr_material(
    mut commands: Commands,
    mut enabled_materials: Query<
        (
            Entity,
            &DisabledPbrMaterial,
        ),
        With<EnablePbrMaterial>,
    >,
) {
    for (entity, disabled_material) in enabled_materials.iter_mut() {
        commands.entity(entity)
            .insert(MeshMaterial3d(disabled_material.material.clone()))
            .remove::<DisabledPbrMaterial>()
            .remove::<EnablePbrMaterial>();
    }
}


fn apply_render_modes(
    mut commands: Commands,
    render_mode: Res<RenderMode>,
    meshes: Query<Entity, With<Mesh3d>>,
    new_meshes: Query<Entity, Added<Mesh3d>>,
) {
    let insert_render_mode_flag = |commands: &mut Commands, entity: Entity| {
        match *render_mode {
            RenderMode::Color => {
                commands.entity(entity)
                    .insert(EnablePbrMaterial);
            }
            RenderMode::Depth => {
                commands.entity(entity)
                    .insert(depth::Depth);
            }
            RenderMode::Normal => {
                commands.entity(entity)
                    .insert(normal::Normal);
            }
            RenderMode::OpticalFlow => {
                commands.entity(entity)
                    .insert(optical_flow::OpticalFlow);
            }
            RenderMode::Semantic => {
                commands.entity(entity)
                    .insert(semantic::Semantic);
            }
        }
    };

    if render_mode.is_changed() {
        for entity in meshes.iter() {
            commands.entity(entity)
                .remove::<depth::Depth>()
                .remove::<normal::Normal>()
                .remove::<optical_flow::OpticalFlow>()
                .remove::<semantic::Semantic>();

            insert_render_mode_flag(&mut commands, entity);
        }
    }

    for entity in new_meshes.iter() {
        insert_render_mode_flag(&mut commands, entity);
    }
}
