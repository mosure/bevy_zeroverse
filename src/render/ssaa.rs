//! GPU box resolve for capture-camera RGB and continuous annotation previews.
//! Numeric annotations retain the final pixel-center ray contract: neither
//! categorical IDs nor geometry/flow/visibility are averaged across surfaces.
use super::{
    sensor::{CameraSensorProfile, ResolveParams},
    RenderMode,
};
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::{
    collections::HashMap,
    num::NonZeroU64,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const SHADER: Handle<Shader> = uuid_handle!("e377b6c2-74c4-4306-a069-5c74d6f25aab");

#[path = "ssaa_flare.rs"]
mod flare;

const SENSOR_SHADER: Handle<Shader> = uuid_handle!("89a2d957-69a1-4828-b722-c1136262ccef");

pub use bevy_zeroverse_capture::sampling::CaptureSampling;

pub fn validate_factor(factor: u32) -> Result<(), String> {
    if factor <= 4 {
        Ok(())
    } else {
        Err("ssaa_factor must be 0 (auto), 1 (off), 2, 3 or 4".into())
    }
}

/// Resolve the default, scene-independent linear scale. Auto (0) selects 2× below the pixel
/// count of 1280×720, and 1× at or above it. Rotation/aspect do not change the
/// threshold. Explicit values override Auto; limits are checked before allocation.
pub fn effective_factor(output_size: UVec2, configured: u32) -> u32 {
    bevy_zeroverse_capture::sampling::base_factor(output_size.to_array(), configured)
}

/// Resolve the actual rig-wide factor, including projected window coverings
/// and device limits. The capture auditor replays this decision against the
/// sampling metadata read back from the camera targets.
pub fn capture_factor(
    size: UVec2,
    config: &crate::app::BevyZeroverseConfig,
    scene: Option<&crate::scene::procedural_indoor::layout::IndoorManifest>,
    device_limit: u32,
) -> u32 {
    let base = effective_factor(size, config.ssaa_factor);
    if config.ssaa_factor != 0
        || base != 2
        || config.scene_type != crate::scene::ZeroverseSceneType::ProceduralIndoor
        || (config.headless && !super::sensor::captures_rgb(config))
    {
        return base;
    }
    scene.map_or(base, |s| {
        crate::scene::procedural_indoor::architecture::facade::treatments::minification::rgb_factor(
            s,
            size,
            config.num_cameras,
            device_limit,
        )
    })
}

pub(crate) fn render_size(size: UVec2, factor: u32, limit: u32) -> Result<UVec2, String> {
    validate_factor(factor)?;
    let factor = effective_factor(size, factor);
    let result = size.x.checked_mul(factor).zip(size.y.checked_mul(factor));
    match result {
        Some((x, y)) if x > 0 && y > 0 && x <= limit && y <= limit => Ok(UVec2::new(x, y)),
        _ => Err(format!(
            "SSAA target {}×{} × {factor} exceeds the device texture limit {limit}",
            size.x, size.y
        )),
    }
}

#[derive(Component, Clone)]
pub struct SsaaCamera {
    pub source: Handle<Image>,
    pub output: Handle<Image>,
    pub output_size: UVec2,
    pub factor: u32,
    completed: Arc<AtomicU64>,
}

impl SsaaCamera {
    pub(crate) fn new(
        source: Handle<Image>,
        output: Handle<Image>,
        output_size: UVec2,
        factor: u32,
    ) -> Self {
        Self {
            source,
            output,
            output_size,
            factor,
            completed: Arc::default(),
        }
    }
    pub(crate) fn recycled(&self) -> Self {
        Self {
            completed: Arc::default(),
            ..self.clone()
        }
    }
    pub(crate) fn ready(&self, request: u64) -> bool {
        self.completed.load(Ordering::Acquire) == request
    }
}

/// Categorical previews also keep exact final-grid samples. Dataset annotation
/// passes keep their original final-size MRTs, including forward flow history.
fn filter_factor(mode: &RenderMode, capture: bool, factor: u32) -> u32 {
    if (capture && *mode != RenderMode::Color)
        || matches!(mode, RenderMode::Semantic | RenderMode::CoVisibility)
    {
        1
    } else {
        factor
    }
}

fn needs_subviewport(mode: &RenderMode, capture: bool, factor: u32, angular: bool) -> bool {
    // The angular logical entity renders the forward cube face. Its physical
    // face size can differ from both the final sensor size and the SSAA source.
    !angular && factor > 1 && filter_factor(mode, capture, factor) == 1
}

fn viewport(
    mode: Res<RenderMode>,
    mut cameras: Query<(
        &SsaaCamera,
        &mut Camera,
        Has<crate::io::image_copy::ImageCopier>,
        Has<crate::camera::lens::LensCamera>,
    )>,
) {
    for (ssaa, mut camera, capture, angular) in &mut cameras {
        if needs_subviewport(&mode, capture, ssaa.factor, angular) {
            if camera.viewport.as_ref().is_none_or(|v| {
                v.physical_size != ssaa.output_size || v.physical_position != UVec2::ZERO
            }) {
                camera.viewport = Some(bevy::camera::Viewport {
                    physical_size: ssaa.output_size,
                    ..default()
                });
            }
        } else if camera.viewport.is_some() {
            camera.viewport = None;
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub(crate) struct SsaaResolveLabel;

// Pure WGSL for both the native shader fast path and WebGPU. Share the exact
// box filter without requiring a shader preprocessor for either pipeline.
fn resolve_shader(source: &str, path: impl Into<String>) -> Shader {
    Shader::from_wgsl([include_str!("area_average.wgsl"), source].concat(), path)
}

pub struct SsaaPlugin;
impl Plugin for SsaaPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "ssaa.wgsl", resolve_shader);
        load_internal_asset!(app, SENSOR_SHADER, "sensor/optics.wgsl", resolve_shader);
        app.add_systems(
            PostUpdate,
            viewport.before(bevy::camera::CameraUpdateSystems),
        );
        let Some(render) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render
            .init_resource::<Jobs>()
            .init_resource::<Pipeline>()
            .init_resource::<Bindings>();
        render.add_systems(ExtractSchedule, extract);
        render.add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
        render.add_systems(
            RenderGraph,
            resolve
                .after(super::co_visibility::CoVisibilityLabel)
                .after(bevy::core_pipeline::schedule::camera_driver)
                .in_set(SsaaResolveLabel)
                .in_set(RenderGraphSystems::Render),
        );
    }
}

struct Job {
    camera: SsaaCamera,
    source: Handle<Image>,
    params: Uniform,
    active: bool,
    request: u64,
}
#[derive(Resource, Default)]
struct Jobs(Vec<Job>);

#[allow(clippy::type_complexity)]
fn extract(
    mut jobs: ResMut<Jobs>,
    mode: Extract<Res<RenderMode>>,
    config: Extract<Res<crate::app::BevyZeroverseConfig>>,
    display: Extract<Option<Res<crate::app::display::DisplayStatus>>>,
    playback: Extract<Res<crate::camera::Playback>>,
    sampler: Extract<Option<Res<crate::sample::SamplerState>>>,
    cameras: Extract<
        Query<(
            &SsaaCamera,
            &Camera,
            Option<&crate::io::image_copy::ImageCopier>,
            Option<&super::ground_truth::GroundTruthCamera>,
            Option<&crate::camera::lens::LensCamera>,
            Option<&crate::camera::ZeroverseCamera>,
            Option<&CameraSensorProfile>,
            Option<&crate::camera::CaptureCameraIndex>,
        )>,
    >,
) {
    jobs.0.clear();
    let preserve_hdr = config
        .tonemapping
        .preserves_hdr(crate::app::display::hdr_destination(
            &config,
            display.as_deref(),
        ));
    for (ssaa, camera, copier, gt, lens, logical, sensor, index) in &cameras {
        let source = if **mode == RenderMode::CoVisibility {
            gt.and_then(|g| g.co_visibility.as_ref())
                .and_then(|v| v.preview.clone())
                .unwrap_or_else(|| ssaa.source.clone())
        } else {
            ssaa.source.clone()
        };
        let factor = filter_factor(&mode, copier.is_some() || lens.is_some(), ssaa.factor);
        let normal = **mode == RenderMode::Normal && copier.is_none();
        let eligible = sampler
            .as_ref()
            .is_none_or(|s| s.rgb_artifact_eligible(index.map(|i| i.0)));
        let params =
            if let Some(profile) = sensor.filter(|_| **mode == RenderMode::Color && eligible) {
                let center = lens.map_or_else(
                    || Vec2::splat(0.5) + logical.map_or(Vec2::ZERO, |c| c.principal_point_offset),
                    |lens| {
                        let k = &lens.calibration;
                        Vec2::new(
                            k.k[0][2] / k.image_size[0] as f32,
                            k.k[1][2] / k.image_size[1] as f32,
                        )
                    },
                );
                let mut params = profile.uniform(factor, center.to_array(), playback.progress);
                params.header[2] = u32::from(preserve_hdr);
                Uniform::Sensor(params)
            } else {
                Uniform::Box([factor, u32::from(normal), 0, 0])
            };
        jobs.0.push(Job {
            camera: ssaa.clone(),
            source,
            params,
            active: crate::sample::camera_activity::render_active(
                &config,
                camera.is_active,
                copier,
            ),
            request: copier.map_or(0, |c| c.requested_id()),
        });
    }
}

#[derive(Resource, Default)]
struct Pipeline {
    flare: flare::Cache,
    clean: Option<(CachedRenderPipelineId, BindGroupLayout)>,
    sensor: Option<(CachedRenderPipelineId, BindGroupLayout)>,
}
#[derive(Clone, Copy, PartialEq)]
enum Uniform {
    Box([u32; 4]),
    Sensor(ResolveParams),
}
impl Uniform {
    // Both fragment pipelines use binding 0 for the same source texture and
    // binding 1 for uniforms. A clean frame reads only the first 16 bytes.
    // Identical layout descriptors let an eligibility switch reuse the group.
    const BUFFER_SIZE: u64 = std::mem::size_of::<ResolveParams>() as u64;
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Box(v) => bytemuck::bytes_of(v),
            Self::Sensor(v) => bytemuck::bytes_of(v),
        }
    }
}
impl Pipeline {
    fn slot(&mut self, params: &Uniform) -> &mut Option<(CachedRenderPipelineId, BindGroupLayout)> {
        match params {
            Uniform::Box(_) => &mut self.clean,
            Uniform::Sensor(_) => &mut self.sensor,
        }
    }
}
#[derive(Resource, Default)]
struct Bindings(HashMap<TextureId, Binding>);
struct Binding {
    flare_grid: Option<UVec2>,
    id: CachedRenderPipelineId,
    params: Uniform,
    source: TextureId,
    uniform: Buffer,
    group: BindGroup,
}

fn source_layout() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "ssaa_box",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer_sized(false, NonZeroU64::new(Uniform::BUFFER_SIZE)),
            ),
        ),
    )
}

fn queue_fragment(
    cache: &PipelineCache,
    name: &'static str,
    shader: Handle<Shader>,
    entry: &'static str,
    layout: Vec<BindGroupLayoutDescriptor>,
) -> CachedRenderPipelineId {
    cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some(name.into()),
        layout,
        vertex: VertexState {
            shader: SHADER,
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader,
            entry_point: Some(entry.into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba32Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    })
}

fn prepare(
    jobs: Res<Jobs>,
    mut pipeline: ResMut<Pipeline>,
    mut bindings: ResMut<Bindings>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    // No pipeline compilation, buffers or intermediate images while disabled.
    if jobs.0.is_empty() {
        bindings.0.clear();
        pipeline.flare.release_image();
        return;
    }
    let mut live = std::collections::HashSet::new();
    let mut max_grid = UVec2::ZERO;
    for job in &jobs.0 {
        let (Some(source), Some(output)) =
            (images.get(&job.source), images.get(&job.camera.output))
        else {
            continue;
        };
        let sensor = matches!(job.params, Uniform::Sensor(_));
        let (id, layout) = pipeline.slot(&job.params).get_or_insert_with(|| {
            let layout = source_layout();
            let (label, shader, entry) = if sensor {
                ("rgb_sensor_resolve", SENSOR_SHADER, "sensor_fragment")
            } else {
                ("ssaa_linear_box_resolve", SHADER, "fragment")
            };
            let id = queue_fragment(&cache, label, shader, entry, vec![layout.clone()]);
            (id, cache.get_bind_group_layout(&layout))
        });
        let grid = match &job.params {
            Uniform::Sensor(params) => {
                let size = UVec2::new(output.texture.width(), output.texture.height());
                let high = UVec2::new(source.texture.width(), source.texture.height());
                (size * params.header[0] == high)
                    .then(|| params.flare_grid(size))
                    .flatten()
            }
            Uniform::Box(_) => None,
        };
        if job.active {
            max_grid = max_grid.max(grid.unwrap_or_default());
        }
        let key = output.texture.id();
        live.insert(key);
        if bindings
            .0
            .get(&key)
            .is_none_or(|b| b.source != source.texture.id())
        {
            let uniform = device.create_buffer(&BufferDescriptor {
                label: Some("ssaa_box_params"),
                size: Uniform::BUFFER_SIZE,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(
                "ssaa_box",
                layout,
                &BindGroupEntries::sequential((&source.texture_view, uniform.as_entire_binding())),
            );
            queue.write_buffer(&uniform, 0, job.params.bytes());
            bindings.0.insert(
                key,
                Binding {
                    flare_grid: grid,
                    id: *id,
                    params: job.params,
                    source: source.texture.id(),
                    uniform,
                    group,
                },
            );
        } else if let Some(binding) = bindings.0.get_mut(&key).filter(|b| b.params != job.params) {
            queue.write_buffer(&binding.uniform, 0, job.params.bytes());
            binding.params = job.params;
            binding.id = *id;
            binding.flare_grid = grid;
        }
    }
    bindings.0.retain(|key, _| live.contains(key));
    pipeline.flare.prepare(max_grid, &device, &cache);
}

fn resolve(world: &World, mut context: RenderContext) {
    #[cfg(not(target_arch = "wasm32"))]
    if crate::sample::render_settling::defer_camera_render(world) {
        return;
    }
    let jobs = world.resource::<Jobs>();
    if jobs.0.is_empty() {
        return;
    }
    let cache = world.resource::<PipelineCache>();
    let images = world.resource::<RenderAssets<GpuImage>>();
    let bindings = world.resource::<Bindings>();
    let flare = &world.resource::<Pipeline>().flare;
    let diagnostics = context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let mut span = None;
    for job in &jobs.0 {
        if !job.active {
            continue;
        }
        let Some(image) = images.get(&job.camera.output) else {
            continue;
        };
        let Some(binding) = bindings.0.get(&image.texture.id()) else {
            continue;
        };
        let Some(pipeline) = cache.get_render_pipeline(binding.id) else {
            continue;
        };
        if span.is_none() {
            span = Some(diagnostics.time_span(context.command_encoder(), "ssaa_box_resolve"));
        }
        let cached = binding
            .flare_grid
            .and_then(|grid| flare.encode(context.command_encoder(), cache, &binding.group, grid));
        let (pipeline, highlights) = cached.map_or((pipeline, None), |(pipeline, group)| {
            (pipeline, Some(group))
        });
        draw(
            context.command_encoder(),
            "ssaa_box_resolve",
            &image.texture_view,
            pipeline,
            &binding.group,
            highlights,
            None,
        );
        // Copies run later in this same render graph/queue. No host fence.
        if let Some(frame) = world.get_resource::<crate::scene::capture_ready::render::Frame>() {
            frame.resolve(job.camera.output.id());
        }
        job.camera.completed.store(job.request, Ordering::Release);
    }
    if let Some(span) = span {
        span.end(context.command_encoder());
    }
}

fn draw(
    encoder: &mut CommandEncoder,
    label: &'static str,
    target: &TextureView,
    pipeline: &RenderPipeline,
    source: &BindGroup,
    highlights: Option<&BindGroup>,
    region: Option<UVec2>,
) {
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: StoreOp::Store,
            },
        })],
        ..default()
    });
    if let Some(size) = region {
        pass.set_scissor_rect(0, 0, size.x, size.y);
    }
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, source, &[]);
    if let Some(group) = highlights {
        pass.set_bind_group(1, group, &[]);
    }
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolve_shaders_are_portable_standalone_wgsl() {
        for entry in [
            include_str!("ssaa.wgsl"),
            include_str!("sensor/optics.wgsl"),
        ] {
            let source = [include_str!("area_average.wgsl"), entry].concat();
            let module = wgpu::naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }
    #[test]
    fn factors_and_texture_limits_fail_before_allocating() {
        assert_eq!(
            render_size(UVec2::new(257, 193), 2, 8192).unwrap(),
            UVec2::new(514, 386)
        );
        for (size, factor) in [
            (UVec2::ONE, 5),
            (UVec2::ZERO, 2),
            (UVec2::splat(4097), 2),
            (UVec2::splat(u32::MAX), 4),
        ] {
            assert!(render_size(size, factor, 8192).is_err());
        }
    }
    #[test]
    fn automatic_threshold_is_pixel_count_based_and_explicit_settings_win() {
        for size in [
            UVec2::splat(256),
            UVec2::splat(512),
            UVec2::new(1279, 720),
            UVec2::new(1600, 300),
        ] {
            assert_eq!(effective_factor(size, 0), 2);
            assert_eq!(render_size(size, 0, 8192).unwrap(), size * 2);
        }
        for size in [
            UVec2::new(1280, 720),
            UVec2::new(720, 1280),
            UVec2::splat(1024),
            UVec2::new(1920, 1080),
        ] {
            assert_eq!(effective_factor(size, 0), 1);
            assert_eq!(render_size(size, 0, 8192).unwrap(), size);
        }
        for factor in 1..=4 {
            assert_eq!(effective_factor(UVec2::splat(256), factor), factor);
            assert_eq!(effective_factor(UVec2::new(1920, 1080), factor), factor);
        }
        assert!(CaptureSampling {
            schema_version: 1,
            ssaa_factor: 0
        }
        .validate()
        .is_err());
    }
    #[test]
    fn forward_cube_faces_and_one_to_one_sensor_targets_use_their_native_viewport() {
        for factor in 1..=4 {
            for mode in [RenderMode::Color, RenderMode::Semantic, RenderMode::Depth] {
                assert!(!needs_subviewport(&mode, true, factor, true));
                assert!(!needs_subviewport(&mode, true, 1, false));
            }
        }
        assert!(needs_subviewport(&RenderMode::Semantic, true, 2, false));
    }
    #[test]
    fn labels_and_metric_capture_never_average_across_surfaces() {
        for mode in [
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
            RenderMode::OpticalFlow,
            RenderMode::MotionVectors,
            RenderMode::CoVisibility,
        ] {
            assert_eq!(filter_factor(&mode, true, 2), 1);
        }
        assert_eq!(filter_factor(&RenderMode::Color, true, 2), 2);
        assert_eq!(filter_factor(&RenderMode::Normal, false, 2), 2);
        assert_eq!(filter_factor(&RenderMode::Semantic, false, 2), 1);
        assert_eq!(filter_factor(&RenderMode::CoVisibility, false, 2), 1);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "ssaa_gpu_tests.rs"]
mod gpu_tests;
