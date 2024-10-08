use bevy::{
    prelude::*,
    asset::load_internal_asset,
    pbr::{
        ExtendedMaterial,
        MaterialExtension,
    },
    render::render_resource::*,
};

use crate::render::DisabledPbrMaterial;


pub const DEPTH_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(63456234534534);

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Depth;


#[derive(Debug, Default)]
pub struct DepthPlugin;
impl Plugin for DepthPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DEPTH_SHADER_HANDLE,
            "depth.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<Depth>();

        app.add_plugins(MaterialPlugin::<DepthMaterial>::default());

        app.add_systems(Update, apply_depth_material);
    }
}


#[allow(clippy::type_complexity)]
fn apply_depth_material(
    mut commands: Commands,
    depths: Query<
        (
            Entity,
            &DisabledPbrMaterial,
        ),
        (With<Depth>, Without<Handle<DepthMaterial>>),
    >,
    mut removed_depths: RemovedComponents<Depth>,
    mut materials: ResMut<Assets<DepthMaterial>>,
) {
    for e in removed_depths.read() {
        if let Some(mut commands) = commands.get_entity(e) {
            commands.remove::<Handle<DepthMaterial>>();
        }
    }

    for (e, pbr_material) in &depths {
        let depth_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: DepthExtension::default(),
            },
        );


        commands.entity(e).insert(depth_material);
    }
}


pub type DepthMaterial = ExtendedMaterial<StandardMaterial, DepthExtension>;

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct DepthExtension { }

impl MaterialExtension for DepthExtension {
    fn fragment_shader() -> ShaderRef {
        DEPTH_SHADER_HANDLE.into()
    }
}
