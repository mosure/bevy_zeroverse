//! Capture-time surface correspondence. History advances on capture stamps, never
//! renderer frames. One geometry pair is shared by all synchronized cameras.
use super::*;
use std::collections::HashMap;

pub(super) const SHADER: Handle<Shader> = uuid_handle!("9a1ba40c-5538-41dd-b989-853bed674d74");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlowVertex {
    source: [f32; 3],
    target: [f32; 3],
    valid: f32,
}

struct Snapshot {
    positions: Vec<Vec3>,
    topology: Vec<ObjectTopology>,
    indices: Vec<u32>,
    batches: Vec<Batch>,
}

impl Snapshot {
    fn new(geometry: &ExtractedGeometry) -> Self {
        Self {
            positions: geometry
                .vertices
                .iter()
                .map(|v| {
                    Mat4::from_cols_array_2d(
                        &geometry.instances[v.instance as usize].world_from_local,
                    )
                    .transform_point3(Vec3::from_array(v.position))
                })
                .collect(),
            topology: geometry.topology.clone(),
            indices: geometry.indices.clone(),
            batches: geometry.batches.clone(),
        }
    }

    fn correspondence(&self, target: &Self) -> Vec<FlowVertex> {
        let targets: HashMap<_, _> = target.topology.iter().map(|t| (t.entity, t)).collect();
        let mut vertices: Vec<_> = self
            .positions
            .iter()
            .map(|p| FlowVertex {
                source: p.to_array(),
                target: p.to_array(),
                valid: 0.0,
            })
            .collect();
        for source in &self.topology {
            let Some(next) = targets.get(&source.entity).filter(|next| {
                source.mesh == next.mesh
                    && source.vertices.len() == next.vertices.len()
                    && source.indices_hash == next.indices_hash
            }) else {
                continue;
            };
            for (vertex, position) in vertices[source.vertices.clone()]
                .iter_mut()
                .zip(&target.positions[next.vertices.clone()])
            {
                vertex.target = position.to_array();
                vertex.valid = 1.0;
            }
        }
        vertices
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlowUniform {
    source_clip: [[f32; 4]; 4],
    target_clip: [[f32; 4]; 4],
    source_view: [[f32; 4]; 4],
    target_view: [[f32; 4]; 4],
    source_limits: [f32; 4],
    target_limits: [f32; 4],
}

#[derive(Clone)]
struct ViewSnapshot {
    clip: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    limits: [f32; 4],
}

#[derive(Resource, Default)]
pub(super) struct FlowHistory {
    stamp: (u64, u64),
    snapshot: Option<Snapshot>,
    views: HashMap<AssetId<Image>, ViewSnapshot>,
    prepared: HashMap<AssetId<Image>, (Buffer, BindGroup)>,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    batches: Vec<Batch>,
}

#[derive(Resource, Default)]
pub(super) struct FlowPipeline(Option<Pipeline>);

struct Pipeline {
    layout: BindGroupLayout,
    pipelines: [CachedRenderPipelineId; 3],
}

impl Pipeline {
    fn new(cache: &PipelineCache) -> Self {
        let descriptor = BindGroupLayoutDescriptor::new(
            "forward_flow_camera",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX_FRAGMENT,
                (
                    uniform_buffer_sized(
                        false,
                        NonZeroU64::new(std::mem::size_of::<FlowUniform>() as u64),
                    ),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );
        let pipelines = [None, Some(Face::Back), Some(Face::Front)].map(|cull_mode| {
            cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("forward_flow_float32".into()),
                layout: vec![descriptor.clone()],
                vertex: VertexState {
                    shader: SHADER,
                    entry_point: Some("vertex".into()),
                    buffers: vec![VertexBufferLayout {
                        array_stride: std::mem::size_of::<FlowVertex>() as u64,
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            VertexAttribute {
                                format: VertexFormat::Float32x3,
                                offset: 12,
                                shader_location: 1,
                            },
                            VertexAttribute {
                                format: VertexFormat::Float32,
                                offset: 24,
                                shader_location: 2,
                            },
                        ],
                    }],
                    ..default()
                },
                fragment: Some(FragmentState {
                    shader: SHADER,
                    entry_point: Some("fragment".into()),
                    targets: vec![Some(ColorTargetState {
                        format: TextureFormat::Rgba32Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                primitive: PrimitiveState {
                    cull_mode,
                    ..default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(CompareFunction::GreaterEqual),
                    stencil: default(),
                    bias: default(),
                }),
                ..default()
            })
        });
        Self {
            layout: cache.get_bind_group_layout(&descriptor),
            pipelines,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    cameras: Query<(&GroundTruthCamera, Option<&ExtractedView>)>,
    geometry: Res<ExtractedGeometry>,
    images: Res<RenderAssets<GpuImage>>,
    mut pipeline: ResMut<FlowPipeline>,
    cache: Res<PipelineCache>,
    mut history: ResMut<FlowHistory>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let cameras: Vec<_> = cameras.iter().filter(|(c, _)| c.flow.is_some()).collect();
    let Some((first, _)) = cameras.first() else {
        if history.snapshot.is_some() {
            *history = default();
        }
        return;
    };
    let stamp = (first.flow_sequence, first.frame_id);
    if stamp.1 == 0
        || stamp == history.stamp
        || geometry.failure.is_some()
        || cameras
            .iter()
            .all(|(camera, _)| camera.rendered_frame() == Some(camera.frame_id))
    {
        return;
    }
    if cameras
        .iter()
        .any(|(c, _)| (c.flow_sequence, c.frame_id) != stamp)
    {
        for (c, _) in &cameras {
            *c.status.failure.lock().unwrap() =
                Some("flow cameras must use synchronized capture stamps".into());
        }
        return;
    }
    if cameras
        .iter()
        .any(|(c, _)| images.get(&c.world_depth).is_none())
    {
        return;
    }
    // Headless readback disables cameras between requests. Bevy removes their
    // ExtractedView while retaining GroundTruthCamera: that is an idle interval,
    // not the end of a sequence. Wait for every synchronized view to return.
    let Some(cameras) = cameras
        .into_iter()
        .map(|(camera, view)| view.map(|view| (camera, view)))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    // No pipeline compilation or temporal buffers until flow is actually requested.
    let pipeline = pipeline.0.get_or_insert_with(|| Pipeline::new(&cache));
    let current = Snapshot::new(&geometry);
    if stamp.0 != history.stamp.0 {
        *history = default();
    }
    history.vertices = None;
    history.indices = None;
    history.batches.clear();
    if let Some(previous) = &history.snapshot {
        let vertices = previous.correspondence(&current);
        if !vertices.is_empty() && !previous.indices.is_empty() {
            let vertex_buffer = super::super::upload_buffer(
                &device,
                &queue,
                &BufferInitDescriptor {
                    label: Some("flow_vertex_pair"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: BufferUsages::VERTEX,
                },
            );
            let index_buffer = super::super::upload_buffer(
                &device,
                &queue,
                &BufferInitDescriptor {
                    label: Some("flow_source_indices"),
                    contents: bytemuck::cast_slice(&previous.indices),
                    usage: BufferUsages::INDEX,
                },
            );
            history.batches = previous.batches.clone();
            history.vertices = Some(vertex_buffer);
            history.indices = Some(index_buffer);
        }
    }
    let mut views = HashMap::new();
    let mut prepared = HashMap::new();
    for (camera, view) in cameras {
        let target = images.get(&camera.world_depth).unwrap();
        let view_matrix = view.world_from_view.to_matrix().inverse();
        let current_view = ViewSnapshot {
            clip: view
                .clip_from_world
                .unwrap_or(view.clip_from_view * view_matrix)
                .to_cols_array_2d(),
            view: view_matrix.to_cols_array_2d(),
            limits: [
                camera.near,
                camera.far,
                target.texture_descriptor.size.width as f32,
                target.texture_descriptor.size.height as f32,
            ],
        };
        let id = camera.flow.as_ref().unwrap().id();
        if let Some(source) = history.views.get(&id) {
            let uniform = FlowUniform {
                source_clip: source.clip,
                target_clip: current_view.clip,
                source_view: source.view,
                target_view: current_view.view,
                source_limits: source.limits,
                target_limits: current_view.limits,
            };
            let buffer = super::super::upload_buffer(
                &device,
                &queue,
                &BufferInitDescriptor {
                    label: Some("forward_flow_camera"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: BufferUsages::UNIFORM,
                },
            );
            let bindings = device.create_bind_group(
                "forward_flow_camera",
                &pipeline.layout,
                &BindGroupEntries::sequential((buffer.as_entire_binding(), &target.texture_view)),
            );
            prepared.insert(id, (buffer, bindings));
        }
        views.insert(id, current_view);
    }
    history.views = views;
    history.prepared = prepared;
    history.snapshot = Some(current);
    history.stamp = stamp;
}

/// Called after the target geometric pass, before its readback completion stamp.
pub(super) fn render(
    world: &World,
    context: &mut RenderContext,
    camera: &GroundTruthCamera,
    depth: &TextureView,
) -> bool {
    let history = world.resource::<FlowHistory>();
    if history.stamp != (camera.flow_sequence, camera.frame_id) {
        return false;
    }
    let output = camera.flow.as_ref().unwrap();
    let images = world.resource::<RenderAssets<GpuImage>>();
    let Some(target) = images.get(output) else {
        return false;
    };
    let Some(pipeline) = world.resource::<FlowPipeline>().0.as_ref() else {
        return false;
    };
    let cache = world.resource::<PipelineCache>();
    for id in &pipeline.pipelines {
        if let CachedPipelineState::Err(error) = cache.get_render_pipeline_state(*id) {
            *camera.status.failure.lock().unwrap() =
                Some(format!("flow shader pipeline failed: {error}"));
            return false;
        }
    }
    let Some(pipelines) = pipeline
        .pipelines
        .iter()
        .map(|id| cache.get_render_pipeline(*id))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let mut pass = context
        .command_encoder()
        .begin_render_pass(&RenderPassDescriptor {
            label: Some("forward_flow_source_raster"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    if let (Some(vertices), Some(indices), Some((_, bindings))) = (
        &history.vertices,
        &history.indices,
        history.prepared.get(&output.id()),
    ) {
        pass.set_vertex_buffer(0, *vertices.slice(..));
        pass.set_index_buffer(*indices.slice(..), IndexFormat::Uint32);
        pass.set_bind_group(0, bindings, &[]);
        for batch in &history.batches {
            if camera.layers.intersects(&batch.layers) {
                pass.set_pipeline(pipelines[batch.cull]);
                pass.draw_indexed(batch.indices.clone(), 0, 0..1);
            }
        }
    }
    true
}
