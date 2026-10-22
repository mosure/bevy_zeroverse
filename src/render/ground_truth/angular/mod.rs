//! Nonlinear annotations trace final pixel-center rays. No interpolation of
//! categorical IDs, depth edges or view-space normal images is involved.
use super::*;
mod bvh;
pub(super) mod flow;
const SHADER: Handle<Shader> = uuid_handle!("5c93e9f1-0130-4845-a19a-dbd4c08d4b21");
#[derive(Resource, Default)]
pub(super) struct State {
    pipeline: Option<(CachedComputePipelineId, BindGroupLayout)>,
    stamp: Option<(u64, Vec<u8>)>,
    triangles: Option<Buffer>,
    nodes: Option<Buffer>,
    node_count: u32,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    world: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    lens: [[f32; 4]; 4],
    limits: [f32; 4],
    counts: [u32; 4],
}
#[derive(Component)]
pub(super) struct Prepared {
    uniform: Buffer,
    layers: Buffer,
    group: BindGroup,
    generation: u64,
    targets: [TextureId; 2],
}

fn matches_stamp(
    stamp: &Option<(u64, Vec<u8>)>,
    geometry: &ExtractedGeometry,
    bytes: &[u8],
) -> bool {
    stamp
        .as_ref()
        .is_some_and(|(g, b)| *g == geometry.generation && b == bytes)
}

fn advance_unchanged_skin(
    stamp: &mut Option<(u64, Vec<u8>)>,
    geometry: &ExtractedGeometry,
    instance_bytes: &[u8],
) {
    if let Some((generation, bytes)) = stamp.as_mut() {
        if bytes == instance_bytes
            && geometry.deformation.as_ref().is_some_and(|update| {
                update.base_generation == *generation && update.ranges.is_empty()
            })
        {
            *generation = geometry.generation;
        }
    }
}

pub(super) fn install(app: &mut App) {
    flow::install(app);
    load_internal_asset!(app, SHADER, "trace.wgsl", |source: &str, path| {
        Shader::from_wgsl(
            crate::camera::lens::shader(source, include_str!("bvh.wgsl")),
            path,
        )
    });
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.init_resource::<State>();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    mut commands: Commands,
    cameras: Query<(
        Entity,
        &GroundTruthCamera,
        Option<&ExtractedView>,
        Option<&Prepared>,
    )>,
    geometry: Res<ExtractedGeometry>,
    mut state: ResMut<State>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    diagnostics: Res<GroundTruthDiagnostics>,
) {
    let instance_bytes = bytemuck::cast_slice::<_, u8>(&geometry.instances);
    if !cameras
        .iter()
        .any(|(_, c, _, _)| c.calibration.as_ref().is_some_and(|l| l.fisheye.is_some()))
    {
        state.stamp = None;
        state.triangles = None;
        state.nodes = None;
        return;
    }
    if cameras.iter().all(|(_, _, v, _)| v.is_none()) {
        return;
    }
    if geometry.failure.is_some() {
        return;
    }
    if state.pipeline.is_none() {
        let layout = BindGroupLayoutDescriptor::new(
            "angular_ground_truth",
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
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let id = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fisheye_exact_ray_annotations".into()),
            layout: vec![layout.clone()],
            shader: SHADER,
            entry_point: Some("main".into()),
            ..default()
        });
        state.pipeline = Some((id, cache.get_bind_group_layout(&layout)));
    }
    // A new capture epoch can refresh skinning without changing any vertex.
    // Carry the certified empty update forward, as the raster/flow paths do.
    // Otherwise a static human rebuilds the entire room BVH at every capture.
    advance_unchanged_skin(&mut state.stamp, &geometry, instance_bytes);
    let changed = !matches_stamp(&state.stamp, &geometry, instance_bytes);
    if changed {
        let bytes = geometry.indices.len() / 3 * std::mem::size_of::<bvh::Triangle>();
        if bytes > device.limits().max_storage_buffer_binding_size as usize {
            for (_, camera, _, _) in &cameras {
                camera.fail(format!("fisheye annotation geometry requires {bytes} bytes, above this device's storage binding limit"));
            }
            return;
        }
        let started = bevy::platform::time::Instant::now();
        let (triangles, nodes) = bvh::build(&geometry);
        let seconds = started.elapsed().as_secs_f64();
        {
            let mut stats = diagnostics.0.lock().unwrap();
            stats.angular_bvh_builds += 1;
            stats.angular_bvh_build_seconds += seconds;
            stats.angular_uploaded_bytes += (triangles.len() * std::mem::size_of::<bvh::Triangle>()
                + nodes.len() * std::mem::size_of::<bvh::Node>())
                as u64;
        }
        let buffer = |label, data: &[u8]| {
            super::super::upload_buffer(
                &device,
                &queue,
                &BufferInitDescriptor {
                    label: Some(label),
                    contents: if data.is_empty() { &[0; 112] } else { data },
                    usage: BufferUsages::STORAGE,
                },
            )
        };
        state.triangles = Some(buffer(
            "angular_triangles",
            bytemuck::cast_slice(&triangles),
        ));
        state.nodes = Some(buffer("angular_bvh", bytemuck::cast_slice(&nodes)));
        state.node_count = nodes.len() as u32;
        state.stamp = Some((geometry.generation, instance_bytes.to_vec()));
    }
    let layout = &state.pipeline.as_ref().unwrap().1;
    for (entity, camera, view, prepared) in &cameras {
        let Some(view) = view else {
            continue;
        };
        let Some(lens) = camera.calibration.as_ref().filter(|l| l.fisheye.is_some()) else {
            continue;
        };
        let (Some(a), Some(b)) = (
            images.get(&camera.world_depth),
            images.get(&camera.normal_semantic),
        ) else {
            continue;
        };
        let world = view.world_from_view.to_matrix();
        let uniform = Uniform {
            world: world.to_cols_array_2d(),
            view: world.inverse().to_cols_array_2d(),
            lens: lens.shader_parameters(),
            limits: [camera.near, camera.far, 0., 0.],
            counts: [state.node_count, 0, 0, 0],
        };
        let layers: Vec<u32> = geometry
            .batches
            .iter()
            .map(|b| u32::from(camera.layers.intersects(&b.layers)))
            .chain(std::iter::once(0))
            .collect();
        if let Some(p) = prepared.filter(|p| {
            !changed
                && p.generation == geometry.generation
                && p.targets == [a.texture.id(), b.texture.id()]
        }) {
            queue.write_buffer(&p.uniform, 0, bytemuck::bytes_of(&uniform));
            queue.write_buffer(&p.layers, 0, bytemuck::cast_slice(&layers));
        } else {
            let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("angular_camera"),
                contents: bytemuck::bytes_of(&uniform),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let layers = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("angular_layers"),
                contents: bytemuck::cast_slice(&layers),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            let group = device.create_bind_group(
                "angular_annotation",
                layout,
                &BindGroupEntries::sequential((
                    buffer.as_entire_binding(),
                    state.triangles.as_ref().unwrap().as_entire_binding(),
                    state.nodes.as_ref().unwrap().as_entire_binding(),
                    layers.as_entire_binding(),
                    &a.texture_view,
                    &b.texture_view,
                )),
            );
            commands.entity(entity).insert(Prepared {
                uniform: buffer,
                layers,
                group,
                generation: geometry.generation,
                targets: [a.texture.id(), b.texture.id()],
            });
        }
    }
}

pub(super) fn render(
    world: &World,
    context: &mut RenderContext,
    camera: &GroundTruthCamera,
    prepared: Option<&Prepared>,
) -> bool {
    let state = world.resource::<State>();
    let geometry = world.resource::<ExtractedGeometry>();
    if !matches_stamp(
        &state.stamp,
        geometry,
        bytemuck::cast_slice(&geometry.instances),
    ) {
        return false;
    }
    let (Some((id, _)), Some(prepared), Some(lens)) =
        (&state.pipeline, prepared, &camera.calibration)
    else {
        return false;
    };
    let cache = world.resource::<PipelineCache>();
    if let CachedPipelineState::Err(error) = cache.get_compute_pipeline_state(*id) {
        camera.fail(format!("fisheye annotation pipeline: {error}"));
    }
    let Some(pipeline) = cache.get_compute_pipeline(*id) else {
        return false;
    };
    let mut pass = context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("fisheye_ground_truth"),
            ..default()
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &prepared.group, &[]);
    pass.dispatch_workgroups(
        lens.image_size[0].div_ceil(8),
        lens.image_size[1].div_ceil(8),
        1,
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_an_exact_noop_skin_update_can_reuse_the_acceleration_tree() {
        for (base, ranges, instances, reusable) in [
            (7, vec![], vec![1_u8, 2], true),
            (6, vec![], vec![1, 2], false),
            (7, std::iter::once(0..3).collect(), vec![1, 2], false),
            (7, vec![], vec![1, 3], false),
        ] {
            let mut stamp = Some((7, vec![1, 2]));
            let geometry = ExtractedGeometry {
                generation: 8,
                deformation: Some(super::super::deformation::VertexUpdate {
                    base_generation: base,
                    ranges,
                }),
                ..default()
            };
            advance_unchanged_skin(&mut stamp, &geometry, &instances);
            assert_eq!(stamp.unwrap().0, if reusable { 8 } else { 7 });
        }
        let mut stamp = Some((7, vec![1, 2]));
        advance_unchanged_skin(
            &mut stamp,
            &ExtractedGeometry {
                generation: 8,
                ..default()
            },
            &[1, 2],
        );
        assert_eq!(stamp.unwrap().0, 7, "topology updates cannot reuse a tree");
    }
}
