//! One geometry raster per camera, then one GPU dispatch for all directed
//! visibility relations. Atlas allocations/bindings are reused until cameras change.
use super::*;
use bevy::{
    asset::{load_internal_asset, uuid_handle, AssetId},
    render::{
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::binding_types::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        view::ExtractedView,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::num::NonZeroU64;

const SHADER: Handle<Shader> = uuid_handle!("b1b67f08-8c99-47b2-93bc-eebd2bc6e1c6");
const PREVIEW: Handle<Shader> = uuid_handle!("06568d06-2bda-474a-aa06-6bfe74e0ccf5");
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub(crate) struct CoVisibilityLabel;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraData {
    clip: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    world: [[f32; 4]; 4],
    limits: [f32; 4],
    rgb: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    cameras: [CameraData; MAX_CAMERAS],
    info: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Palette {
    colors: [[f32; 4]; MAX_CAMERAS],
    count: [u32; 4],
}

#[derive(Resource, Default)]
struct Pipeline(Option<Pipelines>);
struct Pipelines {
    compute: CachedComputePipelineId,
    preview: CachedRenderPipelineId,
    layout: BindGroupLayout,
    preview_layout: BindGroupLayout,
}
impl Pipelines {
    fn new(cache: &PipelineCache) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "co_visibility",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer_sized(
                        false,
                        NonZeroU64::new(std::mem::size_of::<Uniform>() as u64),
                    ),
                    texture_2d_array(TextureSampleType::Float { filterable: false }),
                    texture_2d_array(TextureSampleType::Float { filterable: false }),
                    texture_storage_2d_array(
                        TextureFormat::Rgba32Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
            ),
        );
        let preview_layout = BindGroupLayoutDescriptor::new(
            "co_visibility_preview",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer_sized(
                        false,
                        NonZeroU64::new(std::mem::size_of::<Palette>() as u64),
                    ),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );
        let compute = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("co_visibility_membership".into()),
            layout: vec![layout.clone()],
            shader: SHADER,
            entry_point: Some("visibility".into()),
            ..default()
        });
        let preview = cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("co_visibility_additive_rgb".into()),
            layout: vec![preview_layout.clone()],
            vertex: VertexState {
                shader: PREVIEW,
                entry_point: Some("vertex".into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: PREVIEW,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        });
        Self {
            compute,
            preview,
            layout: cache.get_bind_group_layout(&layout),
            preview_layout: cache.get_bind_group_layout(&preview_layout),
        }
    }
}

#[derive(Resource, Default)]
struct Prepared(Option<Atlas>);
#[derive(Resource, Default)]
struct RetainCaptureStorage(bool);

fn extract_retention(
    mut retained: ResMut<RetainCaptureStorage>,
    args: Extract<Option<Res<crate::app::BevyZeroverseConfig>>>,
) {
    retained.0 = cfg!(not(target_arch = "wasm32"))
        && args.as_ref().is_some_and(|args| {
            args.headless
                && args.image_copiers
                && args
                    .render_modes
                    .iter()
                    .chain(std::iter::once(&args.render_mode))
                    .any(|m| *m == RenderMode::CoVisibility)
        });
}
#[derive(PartialEq, Eq)]
struct Key {
    images: Vec<(AssetId<Image>, TextureId)>,
    size: UVec3,
}
struct Atlas {
    key: Key,
    world: Texture,
    normals: Texture,
    output: Texture,
    uniform: Buffer,
    _palette: Buffer,
    bindings: BindGroup,
    previews: Vec<Option<BindGroup>>,
}

pub(super) fn install(app: &mut App) {
    load_internal_asset!(app, SHADER, "visibility.wgsl", Shader::from_wgsl);
    load_internal_asset!(app, PREVIEW, "preview.wgsl", Shader::from_wgsl);
    let diagnostics = app.world().resource::<CoVisibilityDiagnostics>().clone();
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.insert_resource(diagnostics);
    render
        .init_resource::<Pipeline>()
        .init_resource::<RetainCaptureStorage>()
        .init_resource::<Prepared>();
    render.add_systems(ExtractSchedule, extract_retention);
    render.add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
    render.add_systems(
        RenderGraph,
        render_visibility
            .after(crate::render::ground_truth::GroundTruthLabel)
            .in_set(CoVisibilityLabel)
            .in_set(RenderGraphSystems::Render),
    );
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    cameras: Query<(&GroundTruthCamera, Option<&ExtractedView>)>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    mut pipeline: ResMut<Pipeline>,
    mut prepared: ResMut<Prepared>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    diagnostics: Res<CoVisibilityDiagnostics>,
    retained: Res<RetainCaptureStorage>,
) {
    let mut cameras: Vec<_> = cameras
        .iter()
        .filter(|(c, _)| c.co_visibility.is_some())
        .collect();
    if cameras.is_empty() {
        // Camera entities disappear for a frame during room replacement. The
        // headless pool retains their attachments, so keep this single atlas
        // too. Disabling capture/visibility immediately releases the allocation.
        if !retained.0 {
            prepared.0 = None;
            diagnostics.0.lock().unwrap().atlas_bytes = 0;
        }
        return;
    }
    cameras.sort_by_key(|(c, _)| c.co_visibility.as_ref().unwrap().slot);
    if cameras.len() > MAX_CAMERAS
        || cameras
            .iter()
            .enumerate()
            .any(|(i, (c, _))| c.co_visibility.as_ref().unwrap().slot != i)
    {
        for (c, _) in cameras {
            c.fail("co-visibility requires at most 16 densely ordered capture cameras".into());
        }
        return;
    }
    if cameras
        .iter()
        .all(|(c, _)| c.rendered_frame() == Some(c.frame_id))
    {
        return;
    }
    let count = cameras.len();
    let mut uniform: Uniform = bytemuck::Zeroable::zeroed();
    uniform.info[0] = count as u32;
    let mut key = Key {
        images: vec![],
        size: UVec3::new(0, 0, count as u32),
    };
    for (i, (c, view)) in cameras.iter().enumerate() {
        let Some(view) = view else { return };
        let output = c.co_visibility.as_ref().unwrap();
        for handle in [&c.world_depth, &c.normal_semantic, &output.image]
            .into_iter()
            .chain(output.preview.iter())
        {
            let Some(image) = images.get(handle) else {
                return;
            };
            key.images.push((handle.id(), image.texture.id()));
        }
        let image = images.get(&c.world_depth).unwrap();
        let size = image.texture_descriptor.size;
        key.size.x = key.size.x.max(size.width);
        key.size.y = key.size.y.max(size.height);
        let world = view.world_from_view.to_matrix();
        let camera_view = world.inverse();
        let rgb = camera_color(i, count).map(|v| v as f32 / 255.0);
        uniform.cameras[i] = CameraData {
            clip: view
                .clip_from_world
                .unwrap_or(view.clip_from_view * camera_view)
                .to_cols_array_2d(),
            view: camera_view.to_cols_array_2d(),
            world: world.to_cols_array_2d(),
            limits: [c.near, c.far, size.width as f32, size.height as f32],
            rgb: [rgb[0], rgb[1], rgb[2], 0.0],
        };
    }
    let pipeline = pipeline.0.get_or_insert_with(|| {
        diagnostics.0.lock().unwrap().pipeline_initializations += 1;
        Pipelines::new(&cache)
    });
    if prepared.0.as_ref().is_none_or(|p| p.key != key) {
        let mut stats = diagnostics.0.lock().unwrap();
        stats.atlas_allocations += 1;
        stats.atlas_bytes =
            key.size.x as usize * key.size.y as usize * key.size.z as usize * 16 * 3;
        drop(stats);
        let texture = |label, usage| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: key.size.x,
                    height: key.size.y,
                    depth_or_array_layers: key.size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage,
                view_formats: &[],
            })
        };
        let world = texture(
            "co_visibility_world_atlas",
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        );
        let normals = texture(
            "co_visibility_normals_atlas",
            TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        );
        let output = texture(
            "co_visibility_mask_atlas",
            TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        );
        let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("co_visibility_cameras"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let palette_data = Palette {
            colors: uniform.cameras.map(|c| c.rgb),
            count: uniform.info,
        };
        let palette = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("co_visibility_palette"),
            contents: bytemuck::bytes_of(&palette_data),
            usage: BufferUsages::UNIFORM,
        });
        let view = |t: &Texture| {
            t.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..default()
            })
        };
        let bindings = device.create_bind_group(
            "co_visibility",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                buffer.as_entire_binding(),
                &view(&world),
                &view(&normals),
                &view(&output),
            )),
        );
        let previews = cameras
            .iter()
            .map(|(c, _)| {
                let o = c.co_visibility.as_ref().unwrap();
                o.preview.as_ref().map(|_| {
                    device.create_bind_group(
                        "co_visibility_preview",
                        &pipeline.preview_layout,
                        &BindGroupEntries::sequential((
                            palette.as_entire_binding(),
                            &images.get(&o.image).unwrap().texture_view,
                        )),
                    )
                })
            })
            .collect();
        prepared.0 = Some(Atlas {
            key,
            world,
            normals,
            output,
            uniform: buffer,
            _palette: palette,
            bindings,
            previews,
        });
    } else {
        queue.write_buffer(
            &prepared.0.as_ref().unwrap().uniform,
            0,
            bytemuck::bytes_of(&uniform),
        );
    }
}

fn render_visibility(
    world: &World,
    mut context: RenderContext,
    cameras: Query<&GroundTruthCamera>,
) {
    let Some(atlas) = world.resource::<Prepared>().0.as_ref() else {
        return;
    };
    let Some(pipeline) = world.resource::<Pipeline>().0.as_ref() else {
        return;
    };
    let mut cameras: Vec<_> = cameras
        .iter()
        .filter(|c| c.co_visibility.is_some())
        .collect();
    cameras.sort_by_key(|c| c.co_visibility.as_ref().unwrap().slot);
    if cameras.len() != atlas.key.size.z as usize || cameras.is_empty() {
        return;
    }
    let stamp = cameras[0].frame_id;
    if stamp == 0
        || cameras
            .iter()
            .any(|c| c.frame_id != stamp || c.geometry_frame() != Some(stamp))
        || cameras
            .iter()
            .all(|c| c.co_visibility.as_ref().unwrap().rendered_frame() == Some(stamp))
    {
        return;
    }
    let cache = world.resource::<PipelineCache>();
    if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(pipeline.compute) {
        for c in cameras {
            c.fail(format!("co-visibility shader pipeline failed: {error}"));
        }
        return;
    }
    let Some(compute) = cache.get_compute_pipeline(pipeline.compute) else {
        return;
    };
    let preview = cache.get_render_pipeline(pipeline.preview);
    if atlas.previews.iter().any(Option::is_some) {
        if let CachedPipelineState::Err(error) = cache.get_render_pipeline_state(pipeline.preview) {
            for c in cameras {
                c.fail(format!("co-visibility preview pipeline failed: {error}"));
            }
            return;
        }
    }
    if atlas.previews.iter().any(Option::is_some) && preview.is_none() {
        return;
    }
    let images = world.resource::<RenderAssets<GpuImage>>();
    if cameras.iter().any(|c| {
        let o = c.co_visibility.as_ref().unwrap();
        [&c.world_depth, &c.normal_semantic, &o.image]
            .into_iter()
            .chain(o.preview.iter())
            .any(|h| images.get(h).is_none())
    }) {
        return;
    }
    // A retained atlas must never acknowledge newly sized/replaced attachments
    // until prepare has rebound their exact GPU textures and camera uniforms.
    let current_images = cameras
        .iter()
        .flat_map(|c| {
            let o = c.co_visibility.as_ref().unwrap();
            [&c.world_depth, &c.normal_semantic, &o.image]
                .into_iter()
                .chain(o.preview.iter())
        })
        .map(|h| (h.id(), images.get(h).unwrap().texture.id()));
    if !current_images.eq(atlas.key.images.iter().copied()) {
        return;
    }
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), "co_visibility");
    for (i, c) in cameras.iter().enumerate() {
        for (handle, target) in [
            (&c.world_depth, &atlas.world),
            (&c.normal_semantic, &atlas.normals),
        ] {
            let image = images.get(handle).unwrap();
            let mut target = target.as_image_copy();
            target.origin.z = i as u32;
            context.command_encoder().copy_texture_to_texture(
                image.texture.as_image_copy(),
                target,
                image.texture_descriptor.size,
            );
        }
    }
    {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("co_visibility"),
                timestamp_writes: None,
            });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, &atlas.bindings, &[]);
        pass.dispatch_workgroups(
            atlas.key.size.x.div_ceil(8),
            atlas.key.size.y.div_ceil(8),
            atlas.key.size.z,
        );
    }
    for (i, c) in cameras.iter().enumerate() {
        let o = c.co_visibility.as_ref().unwrap();
        let image = images.get(&o.image).unwrap();
        let mut source = atlas.output.as_image_copy();
        source.origin.z = i as u32;
        context.command_encoder().copy_texture_to_texture(
            source,
            image.texture.as_image_copy(),
            image.texture_descriptor.size,
        );
        if let (Some(target), Some(bindings), Some(pipeline)) =
            (&o.preview, &atlas.previews[i], preview)
        {
            let target = images.get(target).unwrap();
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("co_visibility_preview"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &target.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(wgpu::Color::BLACK),
                            store: StoreOp::Store,
                        },
                    })],
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindings, &[]);
            pass.draw(0..3, 0..1);
        }
        o.status.frame.store(stamp, Ordering::Release);
        o.status.valid.store(true, Ordering::Release);
    }
    span.end(context.command_encoder());
    let mut stats = world
        .resource::<CoVisibilityDiagnostics>()
        .0
        .lock()
        .unwrap();
    stats.dispatches += 1;
    stats.views += cameras.len() as u64;
    stats.pixels += cameras
        .iter()
        .map(|c| {
            let size = images.get(&c.world_depth).unwrap().texture_descriptor.size;
            size.width as u64 * size.height as u64
        })
        .sum::<u64>();
}
