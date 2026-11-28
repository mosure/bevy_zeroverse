//! One-shot native GPU diffuse transport. No readback is required for rendering.
#[cfg(test)]
mod bake_tests;
mod probes;
mod triangles;
use super::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::shader::ShaderCacheError;
use bevy::{
    asset::{load_internal_asset, uuid_handle, AssetId},
    render::{
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        Extract, ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderSystems,
    },
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

const SHADER: Handle<Shader> = uuid_handle!("d7a4f8bd-fb54-4e6f-85ec-6d962c227cc1");

#[derive(Default)]
struct Status {
    encoded: AtomicBool,
    failure: Mutex<Option<String>>,
}

/// `ready` means GI dispatch was encoded before the camera pass. GPU ordering
/// then guarantees capture observes that dispatch; no blocking device poll.
#[derive(Resource, Clone, Default)]
pub struct GiGpuReadiness(Arc<Status>);
impl GiGpuReadiness {
    #[cfg(test)]
    pub(crate) fn encoded_for_test() -> Self {
        let readiness = Self::default();
        readiness.0.encoded.store(true, Ordering::Release);
        readiness
    }

    pub fn ready(&self) -> bool {
        self.0.encoded.load(Ordering::Acquire)
    }
    pub fn failure(&self) -> Option<String> {
        self.0.failure.lock().unwrap().clone()
    }
}

// Preserve the original portable GPU/CPU routing boundary when compacting the
// private upload. Changing integrators can change the floating-point result.
type DenseTriangle = [[f32; 4]; 5];
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo_offset_u: [f32; 4],
    emission_offset_v: [f32; 4],
    uv_columns: [f32; 4],
    base_map: [u32; 2],
    emission_map: [u32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    direction: [f32; 4],
    position_range: [f32; 4],
    color_candela: [f32; 4],
    spot: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    resolution: [u32; 4],
    table_offsets: [u32; 4],
    budget: [u32; 4],
    sun_direction: [f32; 4],
    sun: [f32; 4],
    sky: [f32; 4],
    rotation: [f32; 4],
}

struct Input {
    params: Params,
    triangles: Vec<[u32; 4]>,
    nodes: Arc<Vec<TransportNode>>,
    materials: Vec<GpuMaterial>,
    texels: Vec<[f32; 4]>,
    lights: Vec<GpuLight>,
    origins: probes::ProbeDispatch,
}

#[derive(Resource, Clone)]
pub struct GpuBakeRequest {
    pub image: Handle<Image>,
    input: Arc<Input>,
    pub readiness: GiGpuReadiness,
}

/// A runtime-sized WGSL array must bind space for at least one element, even
/// when its logical count is zero. For nonempty arrays this borrows the
/// original bytes without allocating or copying them.
fn storage_bytes<T: bytemuck::Pod>(values: &[T]) -> &[u8] {
    const ZERO: [u8; size_of::<GpuMaterial>()] = [0; size_of::<GpuMaterial>()];
    const { assert!(size_of::<T>() <= ZERO.len()) };
    if values.is_empty() {
        &ZERO[..size_of::<T>()]
    } else {
        bytemuck::cast_slice(values)
    }
}

impl Input {
    fn buffers(&self) -> [&[u8]; 7] {
        // The shader starts traversal at node zero. An empty tree needs a
        // zero-triangle leaf there; an all-zero node would point back to itself.
        const EMPTY_ROOT: TransportNode = TransportNode {
            lo: Vec3::ZERO,
            hi: Vec3::ZERO,
            index: 0,
            tag: TransportNode::LEAF,
        };
        let nodes = if self.nodes.is_empty() {
            std::slice::from_ref(&EMPTY_ROOT)
        } else {
            self.nodes.as_slice()
        };
        [
            bytemuck::bytes_of(&self.params),
            storage_bytes(&self.triangles),
            bytemuck::cast_slice(nodes),
            storage_bytes(&self.materials),
            storage_bytes(&self.texels),
            storage_bytes(&self.lights),
            storage_bytes(&self.origins.records),
        ]
    }
}

impl GpuBakeRequest {
    pub(crate) fn gpu_buffer_bytes(&self) -> u64 {
        self.input.buffers().iter().map(|b| b.len() as u64).sum()
    }

    fn moment_bytes(&self) -> u64 {
        // One float4 (RGB and variance) per cardinal lobe, released after resolve.
        (u64::from(self.input.params.resolution[3]) * 6 * 16).max(16)
    }

    pub(crate) fn resident_buffer_bytes(&self) -> u64 {
        self.gpu_buffer_bytes() + self.moment_bytes()
    }

    pub(crate) fn probe_texture_bytes(&self) -> u64 {
        u64::from(self.input.params.resolution[3]) * 48
    }
}

#[cfg(test)]
pub(crate) fn test_request(image: Handle<Image>) -> GpuBakeRequest {
    GpuBakeRequest {
        image,
        input: Arc::new(Input {
            params: bytemuck::Zeroable::zeroed(),
            triangles: Vec::new(),
            nodes: Arc::default(),
            materials: Vec::new(),
            texels: Vec::new(),
            lights: Vec::new(),
            origins: probes::ProbeDispatch::new(&[], &[], &[]),
        }),
        readiness: default(),
    }
}

/// Choose the full CPU integrator before packing a transport scene which exceeds
/// the portable native storage binding budget. Both backends use the requested
/// probe spacing, ray count and bounce depth.
pub(in crate::scene::procedural_indoor) fn fits_portable_bindings(scene: &BakeScene) -> bool {
    let texels = scene
        .materials
        .iter()
        .map(|m| {
            let base = m.base_map.as_ref().map_or(0, |map| map.pixels.len());
            let separate = match &m.emission_map {
                EmissionMap::Separate(map) => map.pixels.len(),
                _ => 0,
            };
            (base + separate) as u64
        })
        .sum::<u64>()
        + scene.sky_field.0.len() as u64;
    let limit = wgpu::Limits::default().max_storage_buffer_binding_size;
    fits_records::<DenseTriangle>(scene.triangles.len() as u64, limit)
        && fits_records::<TransportNode>(scene.gpu_transport().nodes.len() as u64, limit)
        && fits_records::<GpuLight>(scene.lights.len() as u64, limit)
        && fits_records::<GpuMaterial>(scene.materials.len() as u64, limit)
        && fits_records::<[f32; 4]>(texels, limit)
}

fn fits_records<T>(count: u64, limit: u64) -> bool {
    count.max(1) <= limit / size_of::<T>() as u64
}

pub fn prepare(
    scene: &BakeScene,
    settings: BakeSettings,
    seed: u64,
    images: &mut impl super::super::preparation::AssetStore<Image>,
) -> (GpuBakeRequest, Transform, BakeStatistics) {
    assert!(settings.spacing >= 0.2 && settings.spacing.is_finite());
    assert!(
        (16..=65536).contains(&settings.rays_per_probe)
            && (1..=32).contains(&settings.diffuse_bounces)
    );
    let started = Instant::now();
    let resolution = ((scene.bounds_max - scene.bounds_min) / settings.spacing)
        .ceil()
        .as_uvec3()
        .max(UVec3::splat(2));
    let count = (resolution.x * resolution.y * resolution.z) as usize;
    assert!(
        count <= 65535,
        "GPU GI probe budget exceeds a dispatch dimension"
    );
    let (points, valid) = scene.probe_grid(resolution);
    let links = scene.probe_links(resolution, &points, &valid);
    let origins = probes::ProbeDispatch::new(&points, &valid, &links);
    // Closest hits, solid classification and the independent CPU oracle retain
    // their original traversal order. Boolean shadows and GPU transport share
    // the same tighter spatial tree without exposing its reordered indices.
    let cached_transport = scene.transport_tree.get().is_some();
    let transport = scene.gpu_transport();
    let nodes = Arc::clone(&transport.nodes);
    // Sky samples lead the existing radiance buffer. Diffuse material offsets
    // below naturally follow them; isotropic rooms need no extra records.
    let mut texels: Vec<_> = scene
        .sky_field
        .0
        .iter()
        .map(|v| v.extend(1.).to_array())
        .collect();
    let materials: Vec<_> = scene
        .materials
        .iter()
        .map(|m| {
            let mut append = |map: &TransportMap| {
                let offset = texels.len() as u32;
                texels.extend(map.pixels.iter().map(|p| p.extend(1.).to_array()));
                [offset, (map.size << 5) | map.address.gpu_flags()]
            };
            let base_map = m.base_map.as_ref().map_or([0; 2], &mut append);
            let emission_map = match &m.emission_map {
                EmissionMap::Uniform => [0; 2],
                EmissionMap::SharedWithBase => base_map,
                EmissionMap::Separate(map) => append(map),
            };
            GpuMaterial {
                // Pack the affine translation alongside the two RGB values to keep
                // the 64-byte transport record and existing storage bindings.
                albedo_offset_u: m.albedo.extend(m.uv_transform.translation.x).to_array(),
                emission_offset_v: m.emission.extend(m.uv_transform.translation.y).to_array(),
                uv_columns: m.uv_transform.matrix2.to_cols_array(),
                base_map,
                emission_map,
            }
        })
        .collect();
    let (triangles, table_offsets) = triangles::pack(transport, scene, &materials);
    let lights = scene
        .lights
        .iter()
        .map(|l| GpuLight {
            direction: l.direction.extend(0.).to_array(),
            position_range: l.position.extend(l.range).to_array(),
            color_candela: l.color.extend(l.candela).to_array(),
            spot: [
                u32::from(l.spot),
                l.inner_cos.to_bits(),
                l.outer_cos.to_bits(),
                0,
            ],
        })
        .collect();
    // The resolve dispatch writes all six lobes of every grid slot, including
    // relocated probes, before camera readiness. No CPU image upload is needed.
    let mut image = probe_image(resolution);
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    let image = images.add(image);
    let input = Input {
        params: Params {
            resolution: [resolution.x, resolution.y, resolution.z, count as u32],
            table_offsets,
            budget: [
                settings.rays_per_probe,
                settings.diffuse_bounces,
                seed as u32,
                scene.lights.len() as u32,
            ],
            sun_direction: scene.sun_direction.extend(0.0).to_array(),
            sun: scene.sun.extend(0.0).to_array(),
            sky: scene
                .sky
                .extend(if scene.sky_field.0.is_empty() {
                    0.
                } else {
                    super::super::sky::SIZE as f32
                })
                .to_array(),
            rotation: scene.world_rotation.to_array(),
        },
        triangles,
        nodes,
        materials,
        texels,
        lights,
        origins,
    };
    let request = GpuBakeRequest {
        image,
        input: Arc::new(input),
        readiness: default(),
    };
    let stats = BakeStatistics {
        backend: "gpu_bvh_compute".into(),
        triangles: scene.triangles.len(),
        probes: count,
        relocated_probes: request.input.origins.relocated,
        primary_rays: count as u64 * u64::from(settings.rays_per_probe),
        traced_primary_rays: Some(
            u64::from(request.input.origins.sources) * u64::from(settings.rays_per_probe),
        ),
        diffuse_bounces: settings.diffuse_bounces,
        texture_bytes: count * 48,
        preparation_ms: scene.preparation_ms
            + started.elapsed().as_secs_f64() * 1000.0
            + if cached_transport {
                transport.preparation_ms
            } else {
                0.0
            },
        // GPU execution is asynchronous. Zero is not a timing measurement;
        // execution timings come from the explicit GPU validation experiment.
        bake_ms: None,
        transport_bytes: request.gpu_buffer_bytes() as usize,
        ..default()
    };
    (
        request,
        probe_transform(scene.bounds_min, scene.bounds_max),
        stats,
    )
}

pub struct GpuGiPlugin;
impl Plugin for GpuGiPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "gpu.wgsl", Shader::from_wgsl);
        let Some(render) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render
            .init_resource::<Prepared>()
            .init_resource::<FutureRequest>()
            .init_resource::<FuturePrepared>();
        render.add_systems(ExtractSchedule, extract);
        render.add_systems(Render, upload.in_set(RenderSystems::PrepareResources));
        render.init_gpu_resource::<Pipeline>();
        render.add_systems(
            RenderGraph,
            bake_gpu
                .before(bevy::core_pipeline::schedule::camera_driver)
                .in_set(RenderGraphSystems::Render),
        );
    }
}

fn extract(
    mut commands: Commands,
    request: Extract<Option<Res<GpuBakeRequest>>>,
    future: Extract<Res<super::super::preparation::residency::FutureAssets>>,
    mut next: ResMut<FutureRequest>,
) {
    if let Some(request) = request.as_ref() {
        if request.is_changed() {
            commands.insert_resource((**request).clone());
        }
    } else {
        commands.remove_resource::<GpuBakeRequest>();
    }
    let requests: Vec<_> = future.gpu_requests().collect();
    if !next
        .0
        .iter()
        .map(|r| (&r.key, r.request.image.id()))
        .eq(requests.iter().map(|r| (&r.key, r.request.image.id())))
    {
        next.0 = requests.into_iter().cloned().collect();
    }
}

#[derive(Resource, Default)]
struct FutureRequest(Vec<super::super::preparation::residency::FutureGi>);

#[derive(Resource, Default)]
struct FuturePrepared([Prepared; 2]);

#[derive(Resource)]
struct Pipeline {
    layout: BindGroupLayout,
    bake: CachedComputePipelineId,
    resolve: CachedComputePipelineId,
}
impl FromWorld for Pipeline {
    fn from_world(world: &mut World) -> Self {
        let mut entries = vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for binding in 1..7 {
            entries.push(BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(BindGroupLayoutEntry {
            binding: 7,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D3,
            },
            count: None,
        });
        entries.push(BindGroupLayoutEntry {
            binding: 8,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout_descriptor = BindGroupLayoutDescriptor::new("indoor_gi_layout", &entries);
        let layout = world
            .resource::<PipelineCache>()
            .get_bind_group_layout(&layout_descriptor);
        let queue = |entry: &'static str| {
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(format!("indoor_gi_{entry}").into()),
                    layout: vec![layout_descriptor.clone()],
                    shader: SHADER,
                    entry_point: Some(entry.into()),
                    ..default()
                })
        };
        Self {
            layout,
            bake: queue("bake"),
            resolve: queue("resolve"),
        }
    }
}

#[derive(Resource, Default)]
struct Prepared {
    image: Option<AssetId<Image>>,
    bind_group: Option<BindGroup>,
    _buffers: Vec<Buffer>,
}

fn take_prepared(slots: &mut [Prepared], image: AssetId<Image>) -> Option<Prepared> {
    slots
        .iter_mut()
        .find(|slot| slot.image == Some(image))
        .map(std::mem::take)
}

#[allow(clippy::too_many_arguments)]
fn upload(
    request: Option<Res<GpuBakeRequest>>,
    pipeline: Res<Pipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    mut prepared: ResMut<Prepared>,
    future: (Res<FutureRequest>, ResMut<FuturePrepared>),
) {
    let (next, mut next_prepared) = future;
    let failure = [pipeline.bake, pipeline.resolve]
        .into_iter()
        .find_map(|id| pipeline_failure(cache.get_compute_pipeline_state(id)));
    if let Some(request) = request.as_deref() {
        // Promotion reuses independently uploaded bindings, including when the
        // future dispatch has not yet encoded. Never overwrite its probe data.
        if prepared.image != Some(request.image.id()) {
            if let Some(slot) = take_prepared(&mut next_prepared.0, request.image.id()) {
                *prepared = slot;
            }
        }
        upload_request(
            request,
            &pipeline,
            failure.as_deref(),
            &device,
            &queue,
            &images,
            &mut prepared,
        );
    } else {
        *prepared = default();
    }
    // Slot positions change as the contiguous queue promotes its first room.
    // Transfer by immutable probe identity before retiring unused bindings.
    let mut previous = std::mem::take(&mut next_prepared.0);
    for (next, slot) in next.0.iter().zip(&mut next_prepared.0) {
        if let Some(old) = take_prepared(&mut previous, next.request.image.id()) {
            *slot = old;
        }
        upload_request(
            &next.request,
            &pipeline,
            failure.as_deref(),
            &device,
            &queue,
            &images,
            slot,
        );
    }
}

fn upload_request(
    request: &GpuBakeRequest,
    pipeline: &Pipeline,
    failure: Option<&str>,
    device: &RenderDevice,
    queue: &RenderQueue,
    images: &RenderAssets<GpuImage>,
    prepared: &mut Prepared,
) {
    if let Some(error) = failure {
        *request.readiness.0.failure.lock().unwrap() = Some(error.to_owned());
        return;
    }
    if request.readiness.ready() {
        // Encoding holds GPU resource references until submission/completion;
        // completed one-shot input buffers need no room-lifetime residency.
        *prepared = default();
        return;
    }
    if prepared.image == Some(request.image.id()) {
        return;
    }
    let Some(image) = images.get(&request.image) else {
        return;
    };
    let bytes = request.input.buffers();
    let limits = device.limits();
    let mut sizes = bytes
        .iter()
        .enumerate()
        .map(|(i, data)| (i, data.len() as u64))
        .chain(std::iter::once((8, request.moment_bytes())));
    if let Some((binding, length, limit)) = sizes.find_map(|(binding, length)| {
        let limit = limits.max_buffer_size.min(if binding == 0 {
            limits.max_uniform_buffer_binding_size
        } else {
            limits.max_storage_buffer_binding_size
        });
        (length > limit).then_some((binding, length, limit))
    }) {
        *request.readiness.0.failure.lock().unwrap() = Some(format!(
            "indoor GI binding {binding} requires {length} bytes; adapter limit is {limit}"
        ));
        prepared.image = Some(request.image.id());
        prepared.bind_group = None;
        prepared._buffers.clear();
        return;
    }
    *request.readiness.0.failure.lock().unwrap() = None;
    let mut buffers: Vec<_> = bytes
        .iter()
        .enumerate()
        .map(|(i, data)| {
            crate::render::upload_buffer(
                device,
                queue,
                &BufferInitDescriptor {
                    label: Some("indoor_gi_transport"),
                    contents: data,
                    usage: if i == 0 {
                        BufferUsages::UNIFORM
                    } else {
                        BufferUsages::STORAGE
                    },
                },
            )
        })
        .collect();
    // GPU-only scratch, bounded by the 65,535-slot dispatch limit (< 6 MiB).
    let moments = device.create_buffer(&BufferDescriptor {
        label: Some("indoor_gi_lobe_moments"),
        size: request.moment_bytes(),
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    entries.push(BindGroupEntry {
        binding: 7,
        resource: BindingResource::TextureView(&image.texture_view),
    });
    entries.push(BindGroupEntry {
        binding: 8,
        resource: moments.as_entire_binding(),
    });
    let bind_group = device.create_bind_group("indoor_gi_bindings", &pipeline.layout, &entries);
    buffers.push(moments);
    *prepared = Prepared {
        image: Some(request.image.id()),
        bind_group: Some(bind_group),
        _buffers: buffers,
    };
}

fn pipeline_failure(state: &CachedPipelineState) -> Option<String> {
    match state {
        CachedPipelineState::Err(
            ShaderCacheError::ShaderNotLoaded(_) | ShaderCacheError::ShaderImportNotYetAvailable,
        ) => None,
        CachedPipelineState::Err(error) => Some(format!("indoor GI compute pipeline: {error}")),
        _ => None,
    }
}

fn bake_gpu(world: &World, mut context: RenderContext) {
    let pipeline = world.resource::<Pipeline>();
    let cache = world.resource::<PipelineCache>();
    let (Some(bake), Some(resolve)) = (
        cache.get_compute_pipeline(pipeline.bake),
        cache.get_compute_pipeline(pipeline.resolve),
    ) else {
        return;
    };
    if let Some(request) = world.get_resource::<GpuBakeRequest>() {
        encode_request(
            request,
            world.resource::<Prepared>(),
            [bake, resolve],
            &mut context,
            "indoor_diffuse_bake",
        );
        // An unfinished current request always has priority over speculation.
        if !request.readiness.ready() {
            return;
        }
    }
    // Independent seed-ordered dispatches preserve each room's original shader,
    // group IDs, random state and reduction order. There is no cross-room sum.
    for (index, (next, prepared)) in world
        .resource::<FutureRequest>()
        .0
        .iter()
        .zip(&world.resource::<FuturePrepared>().0)
        .enumerate()
    {
        encode_request(
            &next.request,
            prepared,
            [bake, resolve],
            &mut context,
            if index == 0 {
                "indoor_diffuse_bake_future"
            } else {
                "indoor_diffuse_bake_future_second"
            },
        );
    }
}

fn encode_request(
    request: &GpuBakeRequest,
    prepared: &Prepared,
    pipelines: [&ComputePipeline; 2],
    context: &mut RenderContext,
    diagnostic_label: &'static str,
) {
    if request.readiness.ready() {
        return;
    }
    if prepared.image != Some(request.image.id()) {
        return;
    }
    let Some(bind_group) = &prepared.bind_group else {
        return;
    };
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(context.command_encoder(), diagnostic_label);
    let sources = request.input.origins.sources;
    for (pipeline, groups, label) in [
        (pipelines[0], sources, "indoor_gi_bake_once"),
        (pipelines[1], sources.div_ceil(8), "indoor_gi_resolve_once"),
    ] {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    span.end(context.command_encoder());
    request.readiness.0.encoded.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_future_binding_promotes_by_reserved_image_identity_exactly_once() {
        let mut images = Assets::<Image>::default();
        let first = images.add(Image::default());
        let second = images.add(Image::default());
        let unrelated = images.add(Image::default());
        let mut slots = [
            Prepared {
                image: Some(first.id()),
                ..default()
            },
            Prepared {
                image: Some(second.id()),
                ..default()
            },
        ];
        assert!(take_prepared(&mut slots, unrelated.id()).is_none());
        let current = take_prepared(&mut slots, first.id()).unwrap();
        assert_eq!(current.image, Some(first.id()));
        assert!(slots[0].image.is_none());
        assert_eq!(slots[1].image, Some(second.id()));
        // A second-slot room becomes first without allocation or overwriting
        // its independently prepared bindings. Asset generations are retained.
        let mut next = [Prepared::default(), Prepared::default()];
        next[0] = take_prepared(&mut slots, second.id()).unwrap();
        assert_eq!(next[0].image, Some(second.id()));
        assert!(slots.iter().all(|slot| slot.image.is_none()));
        assert!(take_prepared(&mut next, first.id()).is_none());
        assert_eq!(
            take_prepared(&mut next, second.id()).unwrap().image,
            Some(second.id())
        );
        assert!(take_prepared(&mut next, second.id()).is_none());
    }

    #[test]
    fn promotion_shares_encoded_future_status_without_releasing_current() {
        let mut images = Assets::<Image>::default();
        let current = test_request(images.add(Image::default()));
        let future = test_request(images.add(Image::default()));
        let promoted = future.clone();
        future.readiness.0.encoded.store(true, Ordering::Release);
        assert!(promoted.readiness.ready());
        assert!(!current.readiness.ready());
        assert_ne!(current.image.id(), promoted.image.id());
        assert!(Arc::ptr_eq(&future.input, &promoted.input));
        assert!(Arc::ptr_eq(&future.readiness.0, &promoted.readiness.0));
    }

    #[test]
    fn empty_shader_arrays_bind_a_complete_element_and_preserve_logical_counts() {
        let request = test_request(Handle::default());
        let lengths = request.input.buffers().map(<[u8]>::len);
        assert_eq!(lengths, [size_of::<Params>(), 16, 32, 64, 16, 64, 16]);
        assert_eq!(
            request.gpu_buffer_bytes(),
            lengths.iter().sum::<usize>() as u64
        );
        assert_eq!(request.input.params.budget[3], 0);
        assert!(request.input.nodes.is_empty());
        let bindings = request.input.buffers();
        let root = bytemuck::pod_read_unaligned::<TransportNode>(bindings[2]);
        assert_eq!(root.tag, TransportNode::LEAF);
        assert!(bindings[5].iter().all(|&b| b == 0));
        assert!(!fits_records::<GpuLight>(0, 63));
        assert!(fits_records::<GpuLight>(0, 64));
        assert!(fits_records::<GpuLight>(2, 128));
        assert!(!fits_records::<GpuLight>(2, 127));

        let values = [[1.0f32, 2.0, 3.0, 4.0], [5.0; 4]];
        let bytes = storage_bytes(&values);
        assert_eq!(bytes.as_ptr(), values.as_ptr().cast::<u8>());
        assert_eq!(bytes, bytemuck::bytes_of(&values));
    }

    #[test]
    fn accelerated_transport_preserves_probe_placement_at_touching_room_surfaces() {
        let scene = IndoorManifest::generate_with_humans(
            207,
            super::super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.25,
        )
        .unwrap();
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let transport = BakeScene::from_manifest(&scene, &set, &materials, &images);
        let (request, _, stats) =
            prepare(&transport, BakeSettings::default(), scene.seed, &mut images);
        let (reference, _) =
            super::super::bvh::build_transport_scalar_reference(&transport.triangles);
        assert!(Arc::ptr_eq(
            &request.input.nodes,
            &transport.gpu_transport().nodes
        ));
        triangles::tests::assert_surfaces(
            &request.input.triangles,
            request.input.params.table_offsets,
            &reference,
            &request.input.materials,
        );
        assert!(images.get(&request.image).unwrap().data.is_none());
        // Check every packed trace origin against the two floor polygons and
        // the full six-ray solid test, independent of the dispatch planner.
        let settings = BakeSettings::default();
        let resolution = ((transport.bounds_max - transport.bounds_min) / settings.spacing)
            .ceil()
            .as_uvec3()
            .max(UVec3::splat(2));
        let count = resolution.element_product();
        let primary = &scene.envelope.as_ref().unwrap().footprint;
        let neighbor = scene.adjoining_space().footprint(scene.room_size);
        let sources: Vec<_> = (0..count)
            .filter(|&i| {
                let xyz = UVec3::new(
                    i % resolution.x,
                    (i / resolution.x) % resolution.y,
                    i / (resolution.x * resolution.y),
                );
                let p = transport.bounds_min
                    + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                        * (transport.bounds_max - transport.bounds_min);
                let (outline, floor) = if p.z > scene.room_size.z * 0.5 {
                    (&neighbor, 0.)
                } else {
                    (primary, scene.floor_height(p.xz()))
                };
                super::super::super::envelope::polygon::contains(outline, p.xz(), 0.)
                    && p.y > floor + 0.002
                    && p.y < scene.ceiling_height(p.xz()) - 0.002
                    && !super::super::probe_classification::legacy_inside_solid(&transport, p)
            })
            .collect();
        let packed = &request.input.origins.records[count as usize..];
        assert_eq!(request.input.origins.sources as usize, sources.len());
        assert_eq!(&packed.as_flattened()[..sources.len()], sources);
        assert_eq!(stats.triangles, transport.triangles.len());
        assert_eq!(stats.probes, count as usize);
        assert_eq!(stats.relocated_probes, count as usize - sources.len());
    }

    #[test]
    fn full_precision_transport_fits_the_previous_oversize_room() {
        let limit = 128 * 1024 * 1024;
        let triangles = 136_205_184 / 96; // Captured on llvmpipe, seed 90108.
        assert_eq!(std::mem::size_of::<DenseTriangle>(), 80);
        assert_eq!(std::mem::size_of::<TransportNode>(), 32);
        assert_eq!(std::mem::offset_of!(TransportNode, index), 12);
        assert_eq!(std::mem::offset_of!(TransportNode, hi), 16);
        assert_eq!(std::mem::offset_of!(TransportNode, tag), 28);
        assert!(fits_records::<DenseTriangle>(triangles, limit));
        assert!(fits_records::<DenseTriangle>(limit / 80, limit));
        assert!(!fits_records::<DenseTriangle>(limit / 80 + 1, limit));
        let count = (limit / 80) as usize;
        let fully_textured_records =
            3 * count + count.div_ceil(4) + count.div_ceil(128) + (3 * count).div_ceil(2);
        assert!(fully_textured_records * 16 <= limit as usize);
        assert!(!fits_records::<TransportNode>(u64::MAX, limit));
    }

    #[test]
    fn delayed_shader_availability_is_not_a_permanent_capture_failure() {
        assert!(pipeline_failure(&CachedPipelineState::Queued).is_none());
        assert!(pipeline_failure(&CachedPipelineState::Err(
            ShaderCacheError::ShaderNotLoaded(SHADER.id())
        ))
        .is_none());
        assert!(pipeline_failure(&CachedPipelineState::Err(
            ShaderCacheError::ShaderImportNotYetAvailable
        ))
        .is_none());
    }
}
