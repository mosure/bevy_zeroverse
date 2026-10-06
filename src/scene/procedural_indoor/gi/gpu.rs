//! One-shot native GPU diffuse transport. No readback is required for rendering.
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
    pub fn ready(&self) -> bool {
        self.0.encoded.load(Ordering::Acquire)
    }
    pub fn failure(&self) -> Option<String> {
        self.0.failure.lock().unwrap().clone()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    a: [f32; 4],
    ab: [f32; 4],
    ac: [f32; 4],
    uv01: [f32; 4],
    uv2_material: [f32; 4],
    normal: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNode {
    lo: [f32; 4],
    hi: [f32; 4],
    children: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    albedo: [f32; 4],
    emission: [f32; 4],
    uv_scale: [f32; 4],
    texture: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position_range: [f32; 4],
    color_candela: [f32; 4],
    spot: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    resolution: [u32; 4],
    budget: [u32; 4],
    sun_direction: [f32; 4],
    sun: [f32; 4],
    sky: [f32; 4],
    rotation: [f32; 4],
}

struct Input {
    params: Params,
    triangles: Vec<GpuTriangle>,
    nodes: Vec<GpuNode>,
    materials: Vec<GpuMaterial>,
    texels: Vec<[f32; 4]>,
    lights: Vec<GpuLight>,
    origins: Vec<[f32; 4]>,
}

#[derive(Resource, Clone)]
pub struct GpuBakeRequest {
    pub image: Handle<Image>,
    input: Arc<Input>,
    pub readiness: GiGpuReadiness,
}

impl Input {
    fn buffers(&self) -> [&[u8]; 7] {
        [
            bytemuck::bytes_of(&self.params),
            bytemuck::cast_slice(&self.triangles),
            bytemuck::cast_slice(&self.nodes),
            bytemuck::cast_slice(&self.materials),
            bytemuck::cast_slice(&self.texels),
            bytemuck::cast_slice(&self.lights),
            bytemuck::cast_slice(&self.origins),
        ]
    }
}

impl GpuBakeRequest {
    pub(crate) fn gpu_buffer_bytes(&self) -> u64 {
        self.input
            .buffers()
            .iter()
            .map(|b| b.len().max(16) as u64)
            .sum()
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
            nodes: Vec::new(),
            materials: Vec::new(),
            texels: Vec::new(),
            lights: Vec::new(),
            origins: Vec::new(),
        }),
        readiness: default(),
    }
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
    let points: Vec<Vec3> = (0..count as u32)
        .map(|i| {
            let xyz = UVec3::new(
                i % resolution.x,
                (i / resolution.x) % resolution.y,
                i / (resolution.x * resolution.y),
            );
            scene.bounds_min
                + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                    * (scene.bounds_max - scene.bounds_min)
        })
        .collect();
    let valid: Vec<bool> = points.iter().map(|p| !scene.inside_solid(*p)).collect();
    let mut relocated = 0;
    let origins = points
        .iter()
        .enumerate()
        .map(|(i, point)| {
            let index = if valid[i] {
                i
            } else {
                relocated += 1;
                (0..count)
                    .filter(|&j| valid[j])
                    .min_by(|&a, &b| {
                        points[a]
                            .distance_squared(*point)
                            .total_cmp(&points[b].distance_squared(*point))
                    })
                    .unwrap_or(i)
            };
            points[index].extend(index as f32).to_array()
        })
        .collect();
    // Closest hits, solid classification and the independent CPU oracle retain
    // their original traversal order. Boolean shadows and GPU transport share
    // the same tighter spatial tree without exposing its reordered indices.
    let cached_transport = scene.transport_tree.get().is_some();
    let transport = scene.gpu_transport();
    let triangles = transport
        .triangles
        .iter()
        .map(|t| GpuTriangle {
            a: t.a.extend(0.0).to_array(),
            ab: t.ab.extend(0.0).to_array(),
            ac: t.ac.extend(0.0).to_array(),
            uv01: [t.uv[0].x, t.uv[0].y, t.uv[1].x, t.uv[1].y],
            uv2_material: [t.uv[2].x, t.uv[2].y, t.material as f32, 0.0],
            normal: t.normal.extend(0.0).to_array(),
        })
        .collect();
    let nodes: Vec<_> = transport
        .nodes
        .iter()
        .map(|n| GpuNode {
            lo: n.lo.extend(0.0).to_array(),
            hi: n.hi.extend(0.0).to_array(),
            children: [
                n.start as u32,
                n.count as u32,
                n.right as u32,
                n.axis as u32,
            ],
        })
        .collect();
    let mut texels = Vec::new();
    let materials = scene
        .materials
        .iter()
        .map(|m| {
            let offset = texels.len() as u32;
            let size = if let Some((n, pixels)) = &m.texture {
                texels.extend(pixels.iter().map(|p| p.extend(1.0).to_array()));
                *n
            } else {
                0
            };
            GpuMaterial {
                albedo: m.albedo.extend(0.0).to_array(),
                emission: m.emission.extend(0.0).to_array(),
                uv_scale: [m.uv_scale.x, m.uv_scale.y, 0.0, 0.0],
                texture: [offset, size, u32::from(m.textured_emission), 0],
            }
        })
        .collect();
    if texels.is_empty() {
        texels.push([1.0; 4]);
    }
    let lights = scene
        .lights
        .iter()
        .map(|l| GpuLight {
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
    let data = ProbeData {
        resolution,
        bounds_min: scene.bounds_min,
        bounds_max: scene.bounds_max,
        values: vec![[Vec3::ZERO; 6]; count],
        statistics: default(),
    };
    let mut image = data.image();
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC;
    let image = images.add(image);
    let stats = BakeStatistics {
        backend: "gpu_bvh_compute".into(),
        triangles: scene.triangles.len(),
        probes: count,
        relocated_probes: relocated,
        primary_rays: count as u64 * settings.rays_per_probe as u64,
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
        transport_bytes: scene.triangles.len() * std::mem::size_of::<GpuTriangle>()
            + nodes.len() * std::mem::size_of::<GpuNode>()
            + texels.len() * 16
            + scene.materials.len() * 64
            + count * 16,
        ..default()
    };
    let input = Input {
        params: Params {
            resolution: [resolution.x, resolution.y, resolution.z, count as u32],
            budget: [
                settings.rays_per_probe,
                settings.diffuse_bounces,
                seed as u32,
                scene.lights.len() as u32,
            ],
            sun_direction: scene.sun_direction.extend(0.0).to_array(),
            sun: scene.sun.extend(0.0).to_array(),
            sky: scene.sky.extend(0.0).to_array(),
            rotation: scene.world_rotation.to_array(),
        },
        triangles,
        nodes,
        materials,
        texels,
        lights,
        origins,
    };
    (
        GpuBakeRequest {
            image,
            input: Arc::new(input),
            readiness: default(),
        },
        data.transform(),
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
    pipeline: CachedComputePipelineId,
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
        let layout_descriptor = BindGroupLayoutDescriptor::new("indoor_gi_layout", &entries);
        let layout = world
            .resource::<PipelineCache>()
            .get_bind_group_layout(&layout_descriptor);
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("indoor_diffuse_gi".into()),
                    layout: vec![layout_descriptor],
                    shader: SHADER,
                    entry_point: Some("bake".into()),
                    ..default()
                });
        Self { layout, pipeline }
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
    let failure = pipeline_failure(cache.get_compute_pipeline_state(pipeline.pipeline));
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
    *request.readiness.0.failure.lock().unwrap() = failure.map(str::to_owned);
    if failure.is_some() {
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
    let buffers: Vec<_> = bytes
        .iter()
        .enumerate()
        .map(|(i, data)| {
            crate::render::upload_buffer(
                device,
                queue,
                &BufferInitDescriptor {
                    label: Some("indoor_gi_transport"),
                    contents: if data.is_empty() { &[0; 16] } else { data },
                    usage: if i == 0 {
                        BufferUsages::UNIFORM
                    } else {
                        BufferUsages::STORAGE
                    },
                },
            )
        })
        .collect();
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
    let bind_group = device.create_bind_group("indoor_gi_bindings", &pipeline.layout, &entries);
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
    let Some(pipeline) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    if let Some(request) = world.get_resource::<GpuBakeRequest>() {
        encode_request(
            request,
            world.resource::<Prepared>(),
            pipeline,
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
            pipeline,
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
    pipeline: &ComputePipeline,
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
    {
        let mut pass = context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("indoor_gi_bake_once"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(request.input.params.resolution[3], 1, 1);
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
        // Empty storage inputs still allocate sixteen-byte shader bindings.
        assert_eq!(
            future.gpu_buffer_bytes(),
            std::mem::size_of::<Params>() as u64 + 6 * 16
        );
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
        let (_, _, stats) = prepare(&transport, BakeSettings::default(), scene.seed, &mut images);
        // Build the solid-classification oracle before transport reorders its
        // clone. Furniture tessellation may evolve; the classification must not.
        let settings = BakeSettings::default();
        let resolution = ((transport.bounds_max - transport.bounds_min) / settings.spacing)
            .ceil()
            .as_uvec3()
            .max(UVec3::splat(2));
        let count = resolution.element_product();
        let relocated = (0..count)
            .filter(|&i| {
                let xyz = UVec3::new(
                    i % resolution.x,
                    (i / resolution.x) % resolution.y,
                    i / (resolution.x * resolution.y),
                );
                let p = transport.bounds_min
                    + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                        * (transport.bounds_max - transport.bounds_min);
                transport.inside_solid(p)
            })
            .count();
        assert_eq!(stats.triangles, transport.triangles.len());
        assert_eq!(stats.probes, count as usize);
        assert_eq!(stats.relocated_probes, relocated);
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
