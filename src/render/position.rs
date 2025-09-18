#![allow(dead_code)] // ShaderType derives emit unused check helpers

use bevy::{
    prelude::*,
    asset::{
        load_internal_asset,
        weak_handle,
    },
    pbr::{
        ExtendedMaterial,
        MaterialExtension,
    },
    render::render_resource::*,
};

use crate::{
    render::DisabledPbrMaterial,
    scene::SceneAabb,
};


pub const POSITION_SHADER_HANDLE: Handle<Shader> = weak_handle!("ade99152-1098-445a-9174-e9f60406a582");

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Position;


#[derive(Debug, Default)]
pub struct PositionPlugin;
impl Plugin for PositionPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            POSITION_SHADER_HANDLE,
            "position.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<Position>();

        app.add_plugins(MaterialPlugin::<PositionMaterial>::default());

        app.add_systems(Update, apply_position_material);
    }
}


#[allow(clippy::type_complexity)]
fn apply_position_material(
    mut commands: Commands,
    positions: Query<
        (
            Entity,
            &DisabledPbrMaterial,
        ),
        (With<Position>, Without<MeshMaterial3d<PositionMaterial>>),
    >,
    aabb: Query<&SceneAabb>,
    mut removed_positions: RemovedComponents<Position>,
    mut materials: ResMut<Assets<PositionMaterial>>,
) {
    for e in removed_positions.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<PositionMaterial>>();
        }
    }

    let Ok(aabb) = aabb.single() else { return };

    for (
        e,
        pbr_material,
    ) in &positions {
        let position_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: PositionExtension {
                    settings: PositionSettings {
                        min: aabb.min.extend(1.0),
                        max: aabb.max.extend(1.0),
                    },
                },
            },
        );

        commands.entity(e).insert(MeshMaterial3d(position_material));
    }
}


pub type PositionMaterial = ExtendedMaterial<StandardMaterial, PositionExtension>;


#[derive(Default, ShaderType, Debug, Clone)]
pub struct PositionSettings {
    pub min: Vec4,
    pub max: Vec4,
}

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct PositionExtension {
    #[uniform(102)]
    pub settings: PositionSettings,
}

impl MaterialExtension for PositionExtension {
    fn fragment_shader() -> ShaderRef {
        POSITION_SHADER_HANDLE.into()
    }
}
