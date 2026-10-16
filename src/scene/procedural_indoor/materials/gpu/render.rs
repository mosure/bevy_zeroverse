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
const SHADER: Handle<Shader> = uuid_handle!("d910d869-42aa-49af-9513-4e8bd4b13811");
pub(crate) struct MaterialBakePlugin;
impl Plugin for MaterialBakePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MaterialComputeAvailable>();
        app.add_plugins(super::super::environment::gpu::ReflectionBakePlugin);
        load_internal_asset!(app, SHADER, "synthesis.wgsl", |source, path| {
            Shader::from_wgsl([source, include_str!("paint.wgsl")].join("\n"), path)
        });

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
                let l = d.limits();
                l.max_compute_workgroups_per_dimension >= 64
                    && l.max_storage_buffers_per_shader_stage >= 4
                    && l.max_storage_buffer_binding_size
                        >= kernel::plane_bytes(kernel::MAX_SIZE) * 3
                    && l.max_storage_buffer_binding_size >= 32 * 1024 * 1024
            });
        app.world_mut().resource_mut::<MaterialComputeAvailable>().0 = available;
        if !available {
            return;
        }
        let render = app.sub_app_mut(RenderApp);
        render
            .add_systems(ExtractSchedule, extract)
            .init_gpu_resource::<Pipeline>()
            .add_systems(RenderStartup, reset_device)
            .add_systems(Render, queue.in_set(RenderSystems::PrepareResources));
        render.add_systems(
            RenderGraph,
            bake.before(bevy::core_pipeline::schedule::camera_driver)
                .in_set(RenderGraphSystems::Render),
        );
    }
}
#[derive(Resource, Default)]
struct Extracted {
    current: Requests,
    future: Requests,
}
fn extract(
    mut out: ResMut<Extracted>,
    current: Extract<Option<Res<Requests>>>,
    future: Extract<Res<crate::scene::procedural_indoor::preparation::residency::FutureAssets>>,
) {
    out.current = current.as_deref().cloned().unwrap_or_default();
    out.future = future.material_bakes.clone();
}
#[derive(Resource)]
struct Pipeline {
    layout: BindGroupLayout,
    descriptor: BindGroupLayoutDescriptor,
    ids: Option<[CachedComputePipelineId; 8]>,
}
impl FromWorld for Pipeline {
    fn from_world(world: &mut World) -> Self {
        let descriptor = BindGroupLayoutDescriptor::new(
            "indoor material recipe layout",
            &kernel::layout_entries(),
        );
        let cache = world.resource::<PipelineCache>();
        let layout = cache.get_bind_group_layout(&descriptor);
        Self {
            layout,
            descriptor,
            ids: None,
        }
    }
}
fn queue(requests: Res<Extracted>, mut pipeline: ResMut<Pipeline>, cache: Res<PipelineCache>) {
    if pipeline.ids.is_some() || requests.current.0.is_empty() && requests.future.0.is_empty() {
        return;
    }
    // Compile only when a room actually requests GPU maps. CPU-only callers do
    // not pay shader compilation or scratch allocation costs.
    pipeline.ids = Some(kernel::ENTRIES.map(|entry| {
        cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("indoor material {entry}").into()),
            layout: vec![pipeline.descriptor.clone()],
            shader: SHADER,
            entry_point: Some(entry.into()),
            ..default()
        })
    }));
}
fn reset_device(mut scratch: ResMut<Scratch>, requests: Res<Extracted>) {
    *scratch.0.get_mut().unwrap() = None;
    for r in requests.current.0.iter().chain(&requests.future.0) {
        r.status.encoded.store(false, Ordering::Release);
        *r.status.error.lock().unwrap() = None;
    }
}

#[derive(Resource, Default)]
struct Scratch(Mutex<Option<kernel::Workspace>>);
fn bake(world: &World, mut context: RenderContext) {
    let requests = world.resource::<Extracted>();
    if requests.current.ready() && requests.future.ready() {
        return;
    }
    let pipeline = world.resource::<Pipeline>();
    let cache = world.resource::<PipelineCache>();
    let Some(ids) = pipeline.ids else {
        return;
    };
    for id in ids {
        if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(id) {
            if matches!(
                error,
                ShaderCacheError::ShaderNotLoaded(_)
                    | ShaderCacheError::ShaderImportNotYetAvailable
            ) {
                return;
            }
            for r in requests.current.0.iter().chain(&requests.future.0) {
                *r.status.error.lock().unwrap() =
                    Some(format!("indoor material compute pipeline: {error}"));
            }
            return;
        }
    }
    let pipelines = ids.map(|id| cache.get_compute_pipeline(id));
    if pipelines.iter().any(Option::is_none) {
        return;
    }
    let pipelines = pipelines.map(Option::unwrap);
    let device = world.resource::<RenderDevice>().wgpu_device();
    let images = world.resource::<RenderAssets<GpuImage>>();
    let mut scratch = world.resource::<Scratch>().0.lock().unwrap();
    let workspace = scratch.get_or_insert_with(|| kernel::Workspace::new(device));
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "indoor_material_bake");
    let mut encoders = Vec::new();
    // Current first; future requests exist only for the single bounded resident successor.
    for r in requests.current.0.iter().chain(&requests.future.0) {
        if r.status.encoded.load(Ordering::Acquire) {
            continue;
        }
        let [Some(color), normal, Some(data)] = r.images.each_ref().map(|h| images.get(h)) else {
            continue;
        };
        let normal: Option<&wgpu::Texture> = if r.normal_output {
            let Some(normal) = normal else {
                continue;
            };
            Some(&normal.texture)
        } else {
            None
        };
        let bindings = workspace.bindings(device, &pipeline.layout, &r.recipe);
        // Keep one recipe per encoder. WGPU recycles native command pools across
        // unrelated encoders; a room-sized bake would eventually inflate every
        // pool to hundreds of passes, retaining gigabytes of driver memory.
        // These buffers still share one ordered submission and the workspace.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("indoor material recipe"),
        });
        workspace.encode(
            &mut encoder,
            [
                pipelines[r.recipe.meta[0] as usize],
                pipelines[kernel::NORMALS],
                pipelines[kernel::MIPS],
            ],
            &bindings,
            r.recipe.meta[1],
            [Some(&color.texture), normal, Some(&data.texture)],
        );
        encoders.push(encoder);
        // All maps and mip copies precede the camera pass in this queue. This is
        // a submission dependency, not a host fence or a claim of GPU completion.
        r.status.encoded.store(true, Ordering::Release);
    }
    finish_recipes(&mut context, encoders);
    span.end(context.command_encoder());
}

fn finish_recipes(context: &mut RenderContext, encoders: Vec<wgpu::CommandEncoder>) {
    // Finish in parallel just as Bevy does for its render systems, then restore
    // recipe order: every recipe reads/writes the same scratch buffers.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut buffers = bevy::tasks::ComputeTaskPool::get().scope(|scope| {
            for (index, encoder) in encoders.into_iter().enumerate() {
                scope.spawn(async move { (index, encoder.finish()) });
            }
        });
        buffers.sort_unstable_by_key(|(index, _)| *index);
        for (_, buffer) in buffers {
            context.add_command_buffer(buffer);
        }
    }
    #[cfg(target_arch = "wasm32")]
    for encoder in encoders {
        context.add_command_buffer(encoder.finish());
    }
}
