use bevy::{
    prelude::*,
    asset::load_internal_asset,
    pbr::{
        MaterialPipeline,
        MaterialPipelineKey,
    },
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::*,
    },
};

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

        app.add_systems(Startup, setup_global_depth_material);
        app.add_systems(Update, apply_depth_material);
    }
}

fn setup_global_depth_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<DepthMaterial>>,
) {
    let depth_material = materials.add(DepthMaterial::default());
    commands.insert_resource(GlobalDepthMaterial(depth_material));
}

fn apply_depth_material(
    mut commands: Commands,
    depths: Query<
        Entity,
        (With<Depth>, Without<Handle<DepthMaterial>>),
    >,
    mut removed_depths: RemovedComponents<Depth>,
    global_material: Res<GlobalDepthMaterial>,
) {
    for e in removed_depths.read() {
        if let Some(mut commands) = commands.get_entity(e) {
            commands.remove::<Handle<DepthMaterial>>();
        }
    }

    for e in &depths {
        commands.entity(e).insert(global_material.0.clone());
    }
}


#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct DepthMaterial { }

#[derive(Default, Resource, Debug, Clone)]
pub struct GlobalDepthMaterial(pub Handle<DepthMaterial>);

impl Material for DepthMaterial {
    fn fragment_shader() -> ShaderRef {
        DEPTH_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
