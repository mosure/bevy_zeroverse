use super::*;
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    render::{
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems},
        texture::GpuImage,
        Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup,
        RenderSystems,
    },
    shader::ShaderCacheError,
};
const SHADER: Handle<Shader> = uuid_handle!("0c865497-a20a-49c6-8867-f1048901cbda");
pub(crate) struct ReflectionBakePlugin;
impl Plugin for ReflectionBakePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "convolve.wgsl", Shader::from_wgsl);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<Extracted>()
                .init_resource::<Scratch>();
        }
    }
    fn finish(&self, app: &mut App) {
        let available = app
            .get_sub_app(RenderApp)
            .and_then(|r| r.world().get_resource::<RenderDevice>())
            .is_some_and(|d| {
                d.limits().max_storage_buffers_per_shader_stage >= 4
                    && d.limits().max_storage_buffer_binding_size >= 32 * 1024 * 1024
            });
        if !available {
            return;
        }
        app.sub_app_mut(RenderApp)
            .add_systems(ExtractSchedule, extract)
            .init_gpu_resource::<Pipeline>()
            .add_systems(RenderStartup, reset)
            .add_systems(Render, queue.in_set(RenderSystems::PrepareResources))
            .add_systems(
                RenderGraph,
                bake.before(bevy::core_pipeline::schedule::camera_driver)
                    .in_set(RenderGraphSystems::Render),
            );
    }
}
#[derive(Resource, Default)]
struct Extracted(Vec<Request>);
fn extract(
    mut out: ResMut<Extracted>,
    current: Extract<Option<Res<super::super::super::gpu::Requests>>>,
    future: Extract<Res<crate::scene::procedural_indoor::preparation::residency::FutureAssets>>,
) {
    out.0.clear();
    if let Some(current) = current.as_deref() {
        out.0.extend(current.1.iter().cloned());
    }
    out.0.extend(future.material_bakes.1.iter().cloned());
}
#[derive(Resource)]
struct Pipeline {
    layout: BindGroupLayout,
    descriptor: BindGroupLayoutDescriptor,
    id: Option<CachedComputePipelineId>,
}
impl FromWorld for Pipeline {
    fn from_world(world: &mut World) -> Self {
        let descriptor = BindGroupLayoutDescriptor::new("IBL layout", &layout_entries());
        let layout = world
            .resource::<PipelineCache>()
            .get_bind_group_layout(&descriptor);
        Self {
            layout,
            descriptor,
            id: None,
        }
    }
}
fn queue(requests: Res<Extracted>, mut pipeline: ResMut<Pipeline>, cache: Res<PipelineCache>) {
    if pipeline.id.is_none() && !requests.0.is_empty() {
        pipeline.id = Some(cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("indoor IBL convolution".into()),
            layout: vec![pipeline.descriptor.clone()],
            shader: SHADER,
            entry_point: Some("convolve".into()),
            ..default()
        }));
    }
}
#[derive(Resource, Default)]
struct Scratch(Mutex<Option<Workspace>>);
fn reset(mut scratch: ResMut<Scratch>, requests: Res<Extracted>) {
    *scratch.0.get_mut().unwrap() = None;
    for r in &requests.0 {
        r.status.encoded.store(false, Ordering::Release);
        *r.status.error.lock().unwrap() = None;
    }
}
fn bake(world: &World, mut context: RenderContext) {
    let requests = world.resource::<Extracted>();
    if requests.0.iter().all(Request::ready) {
        return;
    }
    let pipeline = world.resource::<Pipeline>();
    let cache = world.resource::<PipelineCache>();
    let Some(id) = pipeline.id else {
        return;
    };
    if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(id) {
        if !matches!(
            error,
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable
        ) {
            for r in &requests.0 {
                *r.status.error.lock().unwrap() = Some(format!("IBL compute pipeline: {error}"));
            }
        }
        return;
    }
    let Some(compute) = cache.get_compute_pipeline(id) else {
        return;
    };
    let device = world.resource::<RenderDevice>().wgpu_device();
    let images = world.resource::<RenderAssets<GpuImage>>();
    let mut scratch = world.resource::<Scratch>().0.lock().unwrap();
    let workspace = scratch.get_or_insert_with(|| Workspace::new(device));
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "indoor_reflection_convolution");
    for r in &requests.0 {
        if r.ready() {
            continue;
        }
        let [Some(diffuse), Some(specular)] = r.images.each_ref().map(|h| images.get(h)) else {
            continue;
        };
        workspace.encode(
            device,
            context.command_encoder(),
            compute,
            &pipeline.layout,
            &r.source,
            [&diffuse.texture, &specular.texture],
        );
        r.status.encoded.store(true, Ordering::Release);
    }
    span.end(context.command_encoder());
}
