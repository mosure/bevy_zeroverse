use bevy::{
    prelude::*,
    asset::load_internal_asset,
    pbr::{
        MaterialPipeline,
        MaterialPipelineKey,
    },
    render::{
        mesh::MeshVertexBufferLayout,
        render_resource::*,
    },
};


pub const NORMAL_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(234253434561);

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

        app.add_systems(Startup, setup_global_normal_material);
        app.add_systems(Update, apply_normal_material);
    }
}

fn setup_global_normal_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<NormalMaterial>>,
) {
    let normal_material = materials.add(NormalMaterial::default());
    commands.insert_resource(GlobalNormalMaterial(normal_material));
}

fn apply_normal_material(
    mut commands: Commands,
    normals: Query<
        Entity,
        (With<Normal>, Without<Handle<NormalMaterial>>),
    >,
    mut removed_normals: RemovedComponents<Normal>,
    global_material: Res<GlobalNormalMaterial>,
) {
    for e in removed_normals.read() {
        if let Some(mut commands) = commands.get_entity(e) {
            commands.remove::<Handle<NormalMaterial>>();
        }
    }

    for e in &normals {
        commands.entity(e).insert(global_material.0.clone());
    }
}


#[derive(Default, AsBindGroup, TypePath, Debug, Clone, Asset)]
pub struct NormalMaterial { }

#[derive(Default, Resource, Debug, Clone)]
pub struct GlobalNormalMaterial(pub Handle<NormalMaterial>);

impl Material for NormalMaterial {
    fn fragment_shader() -> ShaderRef {
        NORMAL_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
