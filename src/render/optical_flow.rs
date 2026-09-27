use bevy::{
    asset::{load_internal_asset, uuid_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::*,
    shader::ShaderRef,
};

use crate::render::DisabledPbrMaterial;

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

        app.add_plugins(MaterialPlugin::<OpticalFlowMaterial>::default());

        // Material insertion must precede Bevy's specialization bookkeeping.
        // Otherwise annotation-mode scene regeneration can extract a material
        // without its specialization tick (a renderer panic, notably on Wasm).
        app.add_systems(
            PostUpdate,
            apply_optical_flow_material
                .before(bevy::pbr::check_entities_needing_specialization::<OpticalFlowMaterial>),
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

#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct OpticalFlowExtension {}

impl MaterialExtension for OpticalFlowExtension {
    fn enable_shadows() -> bool {
        false
    }

    fn fragment_shader() -> ShaderRef {
        OPTICAL_FLOW_SHADER_HANDLE.into()
    }
}
