use bevy::{
    asset::{load_internal_asset, uuid_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::*,
    shader::ShaderRef,
};

use crate::render::DisabledPbrMaterial;

/// Viewer velocity preview over a fixed reference interval. Dataset flow is
/// instead the exact displacement between captured timesteps (see metadata).
#[derive(Resource, Reflect, Clone, Copy)]
#[reflect(Resource)]
pub struct FlowPreviewSettings {
    pub interval_seconds: f32,
    pub full_scale_pixels: f32,
}
impl Default for FlowPreviewSettings {
    fn default() -> Self {
        Self {
            interval_seconds: 0.05,
            full_scale_pixels: 32.0,
        }
    }
}

#[derive(Default)]
struct PreviewHistory {
    previous: Option<(crate::camera::PlaybackMode, f32)>,
    settle: u8,
}

fn update_preview(
    mode: Res<super::RenderMode>,
    settings: Res<FlowPreviewSettings>,
    playback: Res<crate::camera::Playback>,
    mut events: MessageReader<crate::scene::RegenerateSceneEvent>,
    mut history: Local<PreviewHistory>,
    mut materials: ResMut<Assets<OpticalFlowMaterial>>,
) {
    let regenerated = !events.is_empty();
    events.clear();
    if !mode.is_flow() {
        history.previous = None;
        return;
    }
    if regenerated
        || history
            .previous
            .is_none_or(|(mode, p)| mode != playback.mode || (p - playback.progress).abs() > 0.25)
    {
        history.settle = 2;
    }
    history.previous = Some((playback.mode, playback.progress));
    let valid = history.settle == 0;
    history.settle = history.settle.saturating_sub(1);
    let preview = Vec4::new(
        settings.interval_seconds.clamp(0.001, 1.0),
        settings.full_scale_pixels.clamp(1.0, 1024.0),
        u32::from(valid) as f32,
        0.0,
    );
    // Changing material uniforms only when settings/reset change avoids a
    // bind-group upload on every frame. Delta seconds comes from Bevy's globals.
    let ids: Vec<_> = materials
        .iter()
        .filter(|(_, m)| m.extension.preview != preview)
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        materials.get_mut(id).unwrap().extension.preview = preview;
    }
}

/// Shared by lossless dataset writers. Viewer colors are a separate preview.
pub fn annotation_metadata() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "direction": "forward; source timestep t to target t+1, same camera index",
        "grid": "source image; pixel centers at (x+0.5,y+0.5)",
        "axes": "positive x right, positive y down",
        "optical_flow_units": "pixels per captured interval; not pixels per second",
        "motion_vectors_units": "normalized image displacement; dx/width, dy/height",
        "valid": "source surface retains vertex correspondence and target lies between near/far planes; off-screen targets remain valid",
        "visible": "valid target lies in image and passes target surface depth test",
        "visibility_tolerance_m": "point-to-target-tangent-plane distance <= max(0.001, 0.0001 * target_view_depth)",
        "terminal": "last timestep has zero vectors and both masks zero",
        "surface": "first geometric surface including opaque glass; excludes reflection/refraction and shading motion",
        "precision": "float32 raster attachments; no color conversion or tonemapping",
        "rgba_layout": ["dx", "dy", "valid", "visible"]
    })
}

pub const OPTICAL_FLOW_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3f3f9390-7b0d-483e-b197-a8b79123205d");

#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct OpticalFlow;

#[derive(Debug, Default)]
pub struct OpticalFlowPlugin;
impl Plugin for OpticalFlowPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            OPTICAL_FLOW_SHADER_HANDLE,
            "optical_flow.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<OpticalFlow>();
        app.init_resource::<super::RenderMode>()
            .init_resource::<crate::camera::Playback>()
            .add_message::<crate::scene::RegenerateSceneEvent>();
        app.init_resource::<FlowPreviewSettings>()
            .register_type::<FlowPreviewSettings>();

        app.add_plugins(MaterialPlugin::<OpticalFlowMaterial>::default());

        // Material insertion must precede Bevy's specialization bookkeeping.
        // Otherwise annotation-mode scene regeneration can extract a material
        // without its specialization tick (a renderer panic, notably on Wasm).
        app.add_systems(
            PostUpdate,
            apply_optical_flow_material
                .before(bevy::pbr::check_entities_needing_specialization::<OpticalFlowMaterial>),
        );
        app.add_systems(
            PostUpdate,
            update_preview.after(apply_optical_flow_material),
        );
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn apply_optical_flow_material(
    mut commands: Commands,
    optical_flows: Query<
        (Entity, &DisabledPbrMaterial),
        (
            With<OpticalFlow>,
            Without<MeshMaterial3d<OpticalFlowMaterial>>,
        ),
    >,
    mut removed_optical_flows: RemovedComponents<OpticalFlow>,
    mut materials: ResMut<Assets<OpticalFlowMaterial>>,
    mut cache: Local<super::annotation_material::AnnotationMaterialCache<OpticalFlowMaterial>>,
) {
    for e in removed_optical_flows.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<OpticalFlowMaterial>>();
        }
    }

    for (e, pbr_material) in &optical_flows {
        let optical_flow_material =
            cache.get(&mut materials, pbr_material.annotation_key(), || {
                ExtendedMaterial {
                    base: StandardMaterial {
                        double_sided: pbr_material.double_sided,
                        cull_mode: pbr_material.cull_mode,
                        unlit: true,
                        ..default()
                    },
                    extension: OpticalFlowExtension::default(),
                }
            });

        commands
            .entity(e)
            .insert(MeshMaterial3d(optical_flow_material));
    }
}

pub type OpticalFlowMaterial = ExtendedMaterial<StandardMaterial, OpticalFlowExtension>;

#[derive(AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct OpticalFlowExtension {
    #[uniform(100)]
    pub preview: Vec4,
}
impl Default for OpticalFlowExtension {
    fn default() -> Self {
        Self {
            preview: Vec4::new(0.05, 32.0, 0.0, 0.0),
        }
    }
}

impl MaterialExtension for OpticalFlowExtension {
    fn enable_shadows() -> bool {
        false
    }

    fn fragment_shader() -> ShaderRef {
        OPTICAL_FLOW_SHADER_HANDLE.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_playback_mode_discards_preview_history() {
        let mut app = App::new();
        app.insert_resource(super::super::RenderMode::OpticalFlow)
            .init_resource::<FlowPreviewSettings>()
            .init_resource::<crate::camera::Playback>()
            .init_resource::<Assets<OpticalFlowMaterial>>()
            .add_message::<crate::scene::RegenerateSceneEvent>()
            .add_systems(Update, update_preview);
        let material = app
            .world_mut()
            .resource_mut::<Assets<OpticalFlowMaterial>>()
            .add(OpticalFlowMaterial::default());
        let valid = |app: &App| {
            app.world()
                .resource::<Assets<OpticalFlowMaterial>>()
                .get(&material)
                .unwrap()
                .extension
                .preview
                .z
                > 0.5
        };
        for _ in 0..3 {
            app.update();
        }
        assert!(valid(&app));
        app.world_mut()
            .resource_mut::<crate::camera::Playback>()
            .mode = crate::camera::PlaybackMode::Once;
        app.update();
        assert!(!valid(&app), "mode changes must not display stale motion");
        app.update();
        assert!(!valid(&app));
        app.update();
        assert!(valid(&app), "preview must resume after history settles");
    }
}
