use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::*,
    shader::ShaderRef,
};
use bevy_args::{Deserialize, Serialize, ValueEnum};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{app::BevyZeroverseConfig, render::DisabledPbrMaterial};

mod palette;
pub use bevy_zeroverse_capture::depth::turbo_depth_rgb;

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Serialize, Deserialize, Reflect, ValueEnum,
)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum DepthFormat {
    #[default]
    Colorized,
    Linear,
    Normalized,
    /// Display RGB: warm near, cool far, logarithmic metric distance.
    Turbo,
}

pub const DEPTH_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("2e8b51ab-6faf-4b40-b342-0250d5b75323");

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
            "depth.wesl",
            |source: &str, path| { Shader::from_wesl(palette::shader(source), path) }
        );

        app.register_type::<Depth>();

        app.add_plugins(MaterialPlugin::<DepthMaterial>::default());

        // Material insertion must precede Bevy's specialization bookkeeping.
        // Otherwise annotation-mode scene regeneration can extract a material
        // without its specialization tick (a renderer panic, notably on Wasm).
        app.add_systems(
            PostUpdate,
            apply_depth_material
                .before(bevy::pbr::check_entities_needing_specialization::<DepthMaterial>),
        );
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn apply_depth_material(
    args: Res<BevyZeroverseConfig>,
    mut commands: Commands,
    depths: Query<(Entity, &DisabledPbrMaterial), With<Depth>>,
    new_depths: Query<
        (Entity, &DisabledPbrMaterial),
        (With<Depth>, Without<MeshMaterial3d<DepthMaterial>>),
    >,
    mut removed_depths: RemovedComponents<Depth>,
    mut materials: ResMut<Assets<DepthMaterial>>,
    mut cache: Local<super::annotation_material::AnnotationMaterialCache<DepthMaterial>>,
    mut previous: Local<Option<DepthSettingsData>>,
) {
    for e in removed_depths.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<DepthMaterial>>();
        }
    }

    let settings = DepthSettingsData {
        format: args.depth_format,
        z_depth: args.z_depth,
    };
    let changed = *previous != Some(settings);
    *previous = Some(settings);
    let mut apply = |(e, pbr_material): (Entity, &DisabledPbrMaterial)| {
        let depth_material = cache.get(
            &mut materials,
            {
                let mut key = pbr_material.annotation_key();
                key.extend([args.depth_format as u32, args.z_depth as u32]);
                key
            },
            || ExtendedMaterial {
                base: StandardMaterial {
                    double_sided: pbr_material.double_sided,
                    cull_mode: pbr_material.cull_mode,
                    unlit: true,
                    ..default()
                },
                extension: DepthExtension {
                    format: args.depth_format,
                    z_depth: args.z_depth,
                },
            },
        );

        commands.entity(e).insert(MeshMaterial3d(depth_material));
    };
    // Rebind on an actual depth-setting change, without regenerating geometry.
    // Steady frames keep the original added-material-only query and cache.
    if changed {
        depths.iter().for_each(&mut apply);
    } else {
        new_depths.iter().for_each(apply);
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
    fn enable_shadows() -> bool {
        false
    }
    fn enable_prepass() -> bool {
        false
    }

    fn fragment_shader() -> ShaderRef {
        DEPTH_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialExtensionKey<Self>,
    ) -> std::result::Result<(), SpecializedMeshPipelineError> {
        let depth_format = match key.bind_group_data.format {
            DepthFormat::Colorized => "COLORIZED_DEPTH",
            DepthFormat::Linear => "LINEAR_DEPTH",
            DepthFormat::Normalized => "NORMALIZED_DEPTH",
            DepthFormat::Turbo => "TURBO_DEPTH",
        };

        let depth_type = if key.bind_group_data.z_depth {
            "Z_DEPTH"
        } else {
            "RAY_DEPTH"
        };

        if let Some(fragment) = &mut descriptor.fragment {
            fragment
                .shader_defs
                .extend([depth_format.into(), depth_type.into()]);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_settings_rebind_existing_materials_and_reuse_the_cache() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::asset::AssetPlugin::default()));
        app.init_asset::<Mesh>();
        app.init_asset::<StandardMaterial>();
        app.init_asset::<Shader>();
        app.init_resource::<BevyZeroverseConfig>();
        app.add_plugins(DepthPlugin);
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::default());
        let entity = app
            .world_mut()
            .spawn((Depth, Mesh3d(mesh), DisabledPbrMaterial::default()))
            .id();
        app.update();
        let original = app
            .world()
            .get::<MeshMaterial3d<DepthMaterial>>(entity)
            .unwrap()
            .0
            .clone();
        for format in [
            DepthFormat::Turbo,
            DepthFormat::Linear,
            DepthFormat::Normalized,
        ] {
            app.world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .depth_format = format;
            app.update();
            let handle = &app
                .world()
                .get::<MeshMaterial3d<DepthMaterial>>(entity)
                .unwrap()
                .0;
            assert_eq!(
                app.world()
                    .resource::<Assets<DepthMaterial>>()
                    .get(handle)
                    .unwrap()
                    .extension
                    .format,
                format
            );
            if format == DepthFormat::Normalized {
                assert_eq!(handle, &original);
            }
        }
        assert_eq!(app.world().resource::<Assets<DepthMaterial>>().len(), 3);
    }
}
