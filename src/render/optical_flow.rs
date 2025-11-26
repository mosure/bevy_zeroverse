use bevy::{
    asset::{load_internal_asset, uuid_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::*,
    shader::ShaderRef,
};

use crate::render::DisabledPbrMaterial;

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

        app.add_systems(Update, apply_optical_flow_material);
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
) {
    for e in removed_optical_flows.read() {
        if let Ok(mut commands) = commands.get_entity(e) {
            commands.remove::<MeshMaterial3d<OpticalFlowMaterial>>();
        }
    }

    for (e, pbr_material) in &optical_flows {
        let optical_flow_material = materials.add(ExtendedMaterial {
            base: StandardMaterial {
                double_sided: pbr_material.double_sided,
                cull_mode: pbr_material.cull_mode,
                unlit: true,
                ..default()
            },
            extension: OpticalFlowExtension::default(),
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
    fn fragment_shader() -> ShaderRef {
        OPTICAL_FLOW_SHADER_HANDLE.into()
    }
}
