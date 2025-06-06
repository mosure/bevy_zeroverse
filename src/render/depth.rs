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
use bevy_args::{
    Deserialize,
    Serialize,
    ValueEnum,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{
    app::BevyZeroverseConfig,
    render::DisabledPbrMaterial,
};



#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    Deserialize,
    Reflect,
    ValueEnum,
)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum DepthFormat {
    #[default]
    Colorized,
    Linear,
    Normalized,
}


pub const DEPTH_SHADER_HANDLE: Handle<Shader> = weak_handle!("2e8b51ab-6faf-4b40-b342-0250d5b75323");

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
    args: Res<BevyZeroverseConfig>,
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
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<DepthMaterial>>();
        }
    }

    for (
        e,
        pbr_material,
    ) in &depths {
        let depth_material = materials.add(
            ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    ..default()
                },
                extension: DepthExtension {
                    format: args.depth_format,
                    z_depth: args.z_depth,
                },
            },
        );

        commands
            .entity(e)
            .insert(MeshMaterial3d(depth_material));
    }
}


pub type DepthMaterial = ExtendedMaterial<StandardMaterial, DepthExtension>;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DepthSettingsData {
    pub format: DepthFormat,
    pub z_depth: bool,
}

impl From<&DepthExtension> for DepthSettingsData {
    fn from(extension: &DepthExtension) -> Self {
        DepthSettingsData {
            format: extension.format,
            z_depth: extension.z_depth,
        }
    }
}

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
#[bind_group_data(DepthSettingsData)]
pub struct DepthExtension {
    #[data]
    pub format: DepthFormat,

    #[data]
    pub z_depth: bool,
}

impl MaterialExtension for DepthExtension {
    fn fragment_shader() -> ShaderRef {
        DEPTH_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> std::result::Result<(), SpecializedMeshPipelineError> {
        let depth_format = match key.bind_group_data.format {
            DepthFormat::Colorized => "COLORIZED_DEPTH",
            DepthFormat::Linear => "LINEAR_DEPTH",
            DepthFormat::Normalized => "NORMALIZED_DEPTH",
        };

        let depth_type = if key.bind_group_data.z_depth {
            "Z_DEPTH"
        } else {
            "RAY_DEPTH"
        };

        if let Some(fragment) = &mut descriptor.fragment {
            fragment
                .shader_defs
                .extend([
                    depth_format.into(),
                    depth_type.into(),
                ]);
        }

        Ok(())
    }
}
