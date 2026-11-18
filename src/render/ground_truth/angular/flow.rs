use super::super::flow::{FlowVertex, Snapshot, ViewSnapshot};
use super::*;
use std::collections::HashMap;
const SHADER: Handle<Shader> = uuid_handle!("aa3316ee-d940-4fbd-b138-8854a4818e4f");
#[derive(Default)]
pub(in crate::render::ground_truth) struct History {
    pair: Option<bvh::TreeBuffers>,
    pipeline: Option<(CachedComputePipelineId, BindGroupLayout)>,
    prepared: HashMap<AssetId<Image>, BindGroup>,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    source_world: [[f32; 4]; 4],
    target_view: [[f32; 4]; 4],
    source_lens: [[f32; 4]; 4],
    target_lens: [[f32; 4]; 4],
    limits: [f32; 4],
    counts: [u32; 4],
}
pub(super) fn install(app: &mut App) {
    load_internal_asset!(app, SHADER, "flow.wgsl", |source: &str, path| {
        Shader::from_wgsl(
            crate::camera::lens::shader(source, include_str!("bvh.wgsl")),
            path,
        )
    });
}
pub(in crate::render::ground_truth) fn pair(
    previous: &Snapshot,
    vertices: &[FlowVertex],
    device: &RenderDevice,
    queue: &RenderQueue,
) -> Result<bvh::TreeBuffers, String> {
    let mut tris = Vec::new();
    for (batch_id, batch) in previous.batches.iter().enumerate() {
        for (j, indices) in previous.indices
            [batch.indices.start as usize..batch.indices.end as usize]
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            let v: [_; 3] = std::array::from_fn(|i| &vertices[indices[i] as usize]);
            tris.push(bvh::Triangle {
                p: v.map(|v| [v.source[0], v.source[1], v.source[2], 0.]),
                n: v.map(|v| [v.target[0], v.target[1], v.target[2], v.valid]),
                info: [
                    0,
                    batch.cull as u32,
                    batch_id as u32,
                    batch.indices.start / 3 + j as u32,
                ],
            });
        }
    }
    Ok(bvh::PackedTree::build(tris, &device.limits())?.upload(device, queue))
}
impl History {
    pub(in crate::render::ground_truth) fn reset(&mut self) {
        self.pair = None;
        self.prepared.clear();
    }
    pub(in crate::render::ground_truth) fn set_pair(&mut self, pair: Option<bvh::TreeBuffers>) {
        self.pair = pair;
    }
    pub(in crate::render::ground_truth) fn clear_views(&mut self) {
        self.prepared.clear();
    }
    #[allow(clippy::too_many_arguments)]
    pub(in crate::render::ground_truth) fn prepare(
        &mut self,
        camera: &GroundTruthCamera,
        source: Option<&ViewSnapshot>,
        target: &ViewSnapshot,
        batches: &[Batch],
        images: &RenderAssets<GpuImage>,
        device: &RenderDevice,
        queue: &RenderQueue,
        cache: &PipelineCache,
    ) {
        let Some(lens) = &camera.calibration else {
            return;
        };
        if lens.fisheye.is_none() {
            return;
        }
        if self.pipeline.is_none() {
            let layout = BindGroupLayoutDescriptor::new(
                "angular_flow",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        uniform_buffer_sized(
                            false,
                            NonZeroU64::new(std::mem::size_of::<Uniform>() as u64),
                        ),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        texture_storage_2d(
                            TextureFormat::Rgba32Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            );
            let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("fisheye_temporal_flow".into()),
                layout: vec![layout.clone()],
                shader: SHADER,
                entry_point: Some("main".into()),
                ..default()
            });
            self.pipeline = Some((id, cache.get_bind_group_layout(&layout)));
        }
        let (Some(depth), Some(output)) = (
            images.get(&camera.world_depth),
            camera.flow.as_ref().and_then(|h| images.get(h)),
        ) else {
            return;
        };
        let source = source.unwrap_or(target);
        let source_lens = source.calibration.as_ref().unwrap_or(lens);
        let uniform = Uniform {
            source_world: Mat4::from_cols_array_2d(&source.view)
                .inverse()
                .to_cols_array_2d(),
            target_view: target.view,
            source_lens: source_lens.shader_parameters(),
            target_lens: lens.shader_parameters(),
            limits: [
                source.limits[0],
                source.limits[1],
                target.limits[0],
                target.limits[1],
            ],
            counts: [self.pair.as_ref().map_or(0, |p| p.node_count), 0, 0, 0],
        };
        let buffer = super::super::super::upload_buffer(
            device,
            queue,
            &BufferInitDescriptor {
                label: Some("angular_flow_camera"),
                contents: bytemuck::bytes_of(&uniform),
                usage: BufferUsages::UNIFORM,
            },
        );
        let dummy;
        let (surfaces, nodes, attributes) = if let Some(tree) = &self.pair {
            (&tree.surfaces, &tree.nodes, &tree.attributes)
        } else {
            // All three storage arrays have zero logical records at the first
            // timestep, but each binding must still hold one complete element.
            dummy = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("angular_flow_empty"),
                contents: &[0; 64],
                usage: BufferUsages::STORAGE,
            });
            (&dummy, &dummy, &dummy)
        };
        let layers: Vec<_> = batches
            .iter()
            .map(|b| u32::from(camera.layers.intersects(&b.layers)))
            .chain(std::iter::once(0))
            .collect();
        let layers = super::super::super::upload_buffer(
            device,
            queue,
            &BufferInitDescriptor {
                label: Some("angular_flow_layers"),
                contents: bytemuck::cast_slice(&layers),
                usage: BufferUsages::STORAGE,
            },
        );
        let group = device.create_bind_group(
            "angular_flow",
            &self.pipeline.as_ref().unwrap().1,
            &BindGroupEntries::sequential((
                buffer.as_entire_binding(),
                surfaces.as_entire_binding(),
                nodes.as_entire_binding(),
                layers.as_entire_binding(),
                &depth.texture_view,
                &output.texture_view,
                attributes.as_entire_binding(),
            )),
        );
        self.prepared
            .insert(camera.flow.as_ref().unwrap().id(), group);
    }
    pub(in crate::render::ground_truth) fn render(
        &self,
        world: &World,
        context: &mut RenderContext,
        camera: &GroundTruthCamera,
    ) -> bool {
        let (Some((id, _)), Some(output), Some(calibration)) =
            (&self.pipeline, &camera.flow, &camera.calibration)
        else {
            return false;
        };
        let cache = world.resource::<PipelineCache>();
        if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(*id) {
            camera.fail(format!("fisheye flow pipeline: {error}"));
        }
        let Some(pipeline) = cache.get_compute_pipeline(*id) else {
            return false;
        };
        let Some(group) = self.prepared.get(&output.id()) else {
            return false;
        };
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("fisheye_forward_flow"),
                ..default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(
            calibration.image_size[0].div_ceil(8),
            calibration.image_size[1].div_ceil(8),
            1,
        );
        true
    }
}
