//! Capture-time surface correspondence. History advances on capture stamps, never
//! renderer frames. One geometry pair is shared by all synchronized cameras.
use super::*;
use std::collections::HashMap;

pub(super) const SHADER: Handle<Shader> = uuid_handle!("9a1ba40c-5538-41dd-b989-853bed674d74");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FlowVertex {
    pub(super) source: [f32; 3],
    pub(super) target: [f32; 3],
    pub(super) valid: f32,
}

pub(super) struct Snapshot {
    generation: u64,
    transforms: Vec<[[f32; 4]; 4]>,
    positions: Vec<Vec3>,
    topology: Vec<ObjectTopology>,
    pub(super) indices: Vec<u32>,
    pub(super) batches: Vec<Batch>,
}

impl Snapshot {
    fn new(geometry: &ExtractedGeometry) -> Self {
        Self {
            generation: geometry.generation,
            transforms: geometry
                .instances
                .iter()
                .map(|i| i.world_from_local)
                .collect(),
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

    /// A stationary scene still has optical flow when cameras move. Reuse its
    /// surface pair only; camera uniforms and target-depth occlusion stay live.
    fn matches(&self, geometry: &ExtractedGeometry) -> bool {
        (self.generation == geometry.generation
            || geometry.deformation.as_ref().is_some_and(|update| {
                update.base_generation == self.generation && update.ranges.is_empty()
            }))
            && self.transforms.len() == geometry.instances.len()
            && self
                .transforms
                .iter()
                .zip(&geometry.instances)
                .all(|(a, b)| bytemuck::bytes_of(a) == bytemuck::bytes_of(&b.world_from_local))
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
pub(super) struct ViewSnapshot {
    pub(super) calibration: Option<bevy_zeroverse_capture::calibration::CameraCalibration>,
    clip: [[f32; 4]; 4],
    pub(super) view: [[f32; 4]; 4],
    pub(super) limits: [f32; 4],
}

#[derive(Resource, Default)]
pub(super) struct FlowHistory {
    angular: super::angular::flow::History,
    stamp: (u64, u64),
    snapshot: Option<Snapshot>,
    views: HashMap<AssetId<Image>, ViewSnapshot>,
    prepared: HashMap<AssetId<Image>, (Buffer, BindGroup)>,
    vertices: Option<Buffer>,
    indices: Option<Buffer>,
    batches: Vec<Batch>,
    /// Both endpoints of the retained pair are the current stationary scene.
    /// A moving -> stationary transition must build a new pair once.
    stationary_pair: bool,
    pair_kinds: [bool; 2],
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
    diagnostics: Res<GroundTruthDiagnostics>,
) {
    let cameras: Vec<_> = cameras.iter().filter(|(c, _)| c.flow.is_some()).collect();
    let Some((first, _)) = cameras.first() else {
        if history.snapshot.is_some() {
            let mut angular = std::mem::take(&mut history.angular);
            angular.reset();
            *history = FlowHistory {
                angular,
                ..default()
            };
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
    // Compile and upload only the projection paths requested by the consumer.
    let has_raster = cameras.iter().any(|(c, _)| !c.is_angular());
    let pipeline = has_raster.then(|| pipeline.0.get_or_insert_with(|| Pipeline::new(&cache)));
    if stamp.0 != history.stamp.0 {
        let mut angular = std::mem::take(&mut history.angular);
        angular.reset();
        *history = FlowHistory {
            angular,
            ..default()
        };
    }
    let has_angular = cameras
        .iter()
        .any(|(c, _)| c.calibration.as_ref().is_some_and(|l| l.fisheye.is_some()));
    let unchanged = history
        .snapshot
        .as_ref()
        .is_some_and(|snapshot| snapshot.matches(&geometry));
    let current = (!unchanged).then(|| Snapshot::new(&geometry));
    let pair_kinds = [has_raster, has_angular];
    let reuse = unchanged && history.stationary_pair && history.pair_kinds == pair_kinds;
    if !reuse {
        history.angular.set_pair(None);
        history.vertices = None;
        history.indices = None;
        history.batches.clear();
        if let Some(previous) = &history.snapshot {
            let vertices = previous.correspondence(current.as_ref().unwrap_or(previous));
            if !vertices.is_empty() && !previous.indices.is_empty() {
                let vertex_buffer = has_raster.then(|| {
                    super::super::upload_buffer(
                        &device,
                        &queue,
                        &BufferInitDescriptor {
                            label: Some("flow_vertex_pair"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: BufferUsages::VERTEX,
                        },
                    )
                });
                let index_buffer = has_raster.then(|| {
                    super::super::upload_buffer(
                        &device,
                        &queue,
                        &BufferInitDescriptor {
                            label: Some("flow_source_indices"),
                            contents: bytemuck::cast_slice(&previous.indices),
                            usage: BufferUsages::INDEX,
                        },
                    )
                });
                let angular_pair = has_angular
                    .then(|| super::angular::flow::pair(previous, &vertices, &device, &queue));
                history.batches = previous.batches.clone();
                history.angular.set_pair(angular_pair);
                history.vertices = vertex_buffer;
                history.indices = index_buffer;
                let mut stats = diagnostics.0.lock().unwrap();
                stats.flow_pair_uploads += 1;
                if has_raster {
                    stats.flow_uploaded_bytes +=
                        (vertices.len() * std::mem::size_of::<FlowVertex>()
                            + history.snapshot.as_ref().unwrap().indices.len()
                                * std::mem::size_of::<u32>()) as u64;
                }
            }
        }
    } else {
        diagnostics.0.lock().unwrap().flow_pair_reuses += 1;
    }
    history.stationary_pair = unchanged;
    history.pair_kinds = pair_kinds;
    history.angular.clear_views();
    let mut views = HashMap::new();
    let mut prepared = HashMap::new();
    for (camera, view) in cameras {
        let target = images.get(&camera.world_depth).unwrap();
        let view_matrix = view.world_from_view.to_matrix().inverse();
        let current_view = ViewSnapshot {
            calibration: camera.calibration.clone(),
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
        if camera
            .calibration
            .as_ref()
            .is_some_and(|l| l.fisheye.is_some())
        {
            let history = &mut *history;
            history.angular.prepare(
                camera,
                history.views.get(&id),
                &current_view,
                &history.batches,
                &images,
                &device,
                &queue,
                &cache,
            );
        } else if let Some(source) = history.views.get(&id) {
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
                &pipeline.as_ref().unwrap().layout,
                &BindGroupEntries::sequential((buffer.as_entire_binding(), &target.texture_view)),
            );
            prepared.insert(id, (buffer, bindings));
        }
        views.insert(id, current_view);
    }
    history.views = views;
    history.prepared = prepared;
    if let Some(current) = current {
        history.snapshot = Some(current);
    } else if let Some(previous) = &mut history.snapshot {
        previous.generation = geometry.generation;
    }
    history.stamp = stamp;
}

#[cfg(test)]
#[path = "flow/tests.rs"]
mod tests;

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
    if camera
        .calibration
        .as_ref()
        .is_some_and(|l| l.fisheye.is_some())
    {
        return history.angular.render(world, context, camera);
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
