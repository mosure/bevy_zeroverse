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

use crate::render::DisabledPbrMaterial;


pub const NORMAL_SHADER_HANDLE: Handle<Shader> = weak_handle!("93f535c9-7353-4160-920b-9493b0a2c1ee");

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Normal;


#[derive(Debug, Default)]
pub struct NormalPlugin;
impl Plugin for NormalPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            NORMAL_SHADER_HANDLE,
            "normal.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<Normal>();

        app.add_plugins(MaterialPlugin::<NormalMaterial>::default());

        app.add_systems(Update, apply_normal_material);
    }
}


#[allow(clippy::type_complexity)]
fn apply_normal_material(
    mut commands: Commands,
    normals: Query<
        (
            Entity,
            &DisabledPbrMaterial,
        ),
        (With<Normal>, Without<MeshMaterial3d<NormalMaterial>>),
    >,
    mut removed_normals: RemovedComponents<Normal>,
    mut materials: ResMut<Assets<NormalMaterial>>,
) {
    for e in removed_normals.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<NormalMaterial>>();
        }
    }

    for (e, pbr_material) in &normals {
        let normal_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: NormalExtension::default(),
            },
        );

        commands.entity(e).insert(MeshMaterial3d(normal_material));
    }
}


pub type NormalMaterial = ExtendedMaterial<StandardMaterial, NormalExtension>;


#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct NormalExtension { }

impl MaterialExtension for NormalExtension {
    fn fragment_shader() -> ShaderRef {
        NORMAL_SHADER_HANDLE.into()
    }
}
