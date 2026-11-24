//! Upstream installs a global deferred renderer and builds BLAS even for unused
//! meshes. Select both explicitly so dormant, staged and transparent surfaces
//! do not enter the active room's acceleration structure.
use super::{PreviewLighting, SelectPreviewLighting, SolariPreviewStatus};
use crate::{app::BevyZeroverseConfig, render::RenderMode, scene::ZeroverseSceneType};
use bevy::{
    asset::AssetEventSystems,
    camera::visibility::VisibilitySystems,
    pbr::DefaultOpaqueRendererMethod,
    prelude::*,
    render::{renderer::RenderDevice, RenderApp},
};
use bevy_solari::{realtime::SolariLightingPlugin, scene::RaytracingScenePlugin, SolariPlugins};

mod meshes;
#[cfg(test)]
mod tests;
mod views;

/// Native Studio and explicit offscreen comparisons. The archive and Burn
/// workers do not install this plugin.
pub struct SolariPreviewPlugin;
impl Plugin for SolariPreviewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SolariPreviewStatus>();
    }
    fn finish(&self, app: &mut App) {
        let Some(device) = app
            .get_sub_app(RenderApp)
            .and_then(|a| a.world().get_resource::<RenderDevice>())
        else {
            return;
        };
        let limits = device.limits();
        let missing = SolariPlugins::required_wgpu_features().difference(device.features());
        // Bevy 0.20 binds two 500-slab arrays, up to 5,000 textures and 5,000
        // samplers. Its per-view world cache is one 84 MiB + 4 KiB binding.
        if !missing.is_empty()
            || limits.max_binding_array_elements_per_shader_stage < 6000
            || limits.max_binding_array_sampler_elements_per_shader_stage < 5000
            || limits.max_storage_buffer_binding_size < 84 * 1024 * 1024 + 4096 + 16
        {
            warn!("Solari preview unavailable on this device (missing {missing:?}, binding/sampler arrays {}/{}). Using raster PBR.", limits.max_binding_array_elements_per_shader_stage, limits.max_binding_array_sampler_elements_per_shader_stage);
            app.add_systems(PreUpdate, reject_unsupported);
            return;
        }

        // Bevy exposes the device in finish. Install the two upstream plugins'
        // shaders/schedules only after checking limits, before RenderStartup.
        // Solari's build selects global deferred; we own only the supported
        // StandardMaterials. All other renderer methods keep their usual path.
        let scene = RaytracingScenePlugin;
        let lighting = SolariLightingPlugin;
        let previous = app
            .world_mut()
            .remove_resource::<DefaultOpaqueRendererMethod>();
        scene.build(app);
        lighting.build(app);
        app.insert_resource(previous.unwrap_or_else(DefaultOpaqueRendererMethod::forward));
        scene.finish(app);
        lighting.finish(app);
        let mut status = app.world_mut().resource_mut::<SolariPreviewStatus>();
        status.supported = true;
        status.max_reservoir_pixels = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            / 48;
        app.init_resource::<meshes::TracedAssets>();
        app.add_systems(
            PostUpdate,
            select
                .in_set(SelectPreviewLighting)
                .before(meshes::synchronize)
                .before(views::release)
                .before(crate::render::anti_alias::ConfigurePreviewAa),
        );
        app.add_systems(
            PostUpdate,
            (
                meshes::synchronize
                    .after(VisibilitySystems::VisibilityPropagate)
                    .before(AssetEventSystems),
                meshes::filter_new_blas.after(AssetEventSystems),
                views::release
                    .before(crate::render::anti_alias::ConfigurePreviewAa)
                    .before(crate::camera::lens::LensPhotometry),
                views::synchronize
                    .after(crate::camera::lens::LensPhotometry)
                    .after(bevy::camera::CameraUpdateSystems),
            ),
        );
    }
}

fn reject_unsupported(
    mut config: ResMut<BevyZeroverseConfig>,
    #[cfg(feature = "viewer")] editor: Option<ResMut<crate::app::editor::model::EditorState>>,
) {
    if config.preview_lighting != PreviewLighting::Raster {
        warn!("This device cannot run Solari. Restoring Raster PBR.");
        config.preview_lighting = PreviewLighting::Raster;
        #[cfg(feature = "viewer")]
        if let Some(mut editor) = editor {
            editor.applied["preview_lighting"] = serde_json::json!(PreviewLighting::Raster);
            editor.draft["preview_lighting"] = serde_json::json!(PreviewLighting::Raster);
            editor.error =
                Some("This render device cannot run Solari. Raster PBR is active.".into());
        }
    }
}

fn select(
    config: Res<BevyZeroverseConfig>,
    mode: Res<RenderMode>,
    mut status: ResMut<SolariPreviewStatus>,
) {
    let enabled = config.preview_lighting != PreviewLighting::Raster
        && config.scene_type == ZeroverseSceneType::ProceduralIndoor
        && !config.room_schematic
        && *mode == RenderMode::Color;
    let temporal_reuse = enabled && config.preview_lighting == PreviewLighting::SolariRestir;
    if enabled != status.active || temporal_reuse != status.temporal_reuse {
        status.active = enabled;
        status.temporal_reuse = temporal_reuse;
    }
}
