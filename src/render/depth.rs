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
        (With<Depth>, Without<MeshMaterial3d<DepthMaterial>>),
    >,
    mut removed_depths: RemovedComponents<Depth>,
    mut materials: ResMut<Assets<DepthMaterial>>,
) {
    for e in removed_depths.read() {
        if let Some(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<DepthMaterial>>();
        }
    }

    for (
        e,
        pbr_material,
    ) in &depths {
        // TODO: support this config at runtime
        #[cfg(feature = "viewer")]
        let linear_depth = Vec4::new(0.0, 0.0, 0.0, 0.0);

        #[cfg(not(feature = "viewer"))]
        let linear_depth = Vec4::new(1.0, 0.0, 0.0, 0.0);

        let depth_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: DepthExtension {
                    settings: DepthSettings {
                        linear_depth,
                    },
                },
            },
        );

        commands.entity(e).insert(MeshMaterial3d(depth_material));
    }
}


pub type DepthMaterial = ExtendedMaterial<StandardMaterial, DepthExtension>;


#[derive(Default, ShaderType, Debug, Clone)]
pub struct DepthSettings {
    pub linear_depth: Vec4,
}

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct DepthExtension {
    #[uniform(102)]
    pub settings: DepthSettings,
}

impl MaterialExtension for DepthExtension {
    fn fragment_shader() -> ShaderRef {
        DEPTH_SHADER_HANDLE.into()
    }
}
