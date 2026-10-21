//! GPU box resolve for capture-camera RGB and continuous annotation previews.
//! Numeric annotations retain the final pixel-center ray contract: neither
//! categorical IDs nor geometry/flow/visibility are averaged across surfaces.
use super::RenderMode;
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

/// Capture raster policy; width, height, K and flow units always describe the
/// final images. RGB uses a linear-light box filter after tone mapping. Metric
/// annotations and categorical masks retain exact final-grid pixel-center rays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSampling {
    pub schema_version: u32,
    pub ssaa_factor: u32,
}
impl CaptureSampling {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("unknown capture sampling schema".into());
        }
        if !(1..=4).contains(&self.ssaa_factor) {
            return Err(
                "capture sampling metadata must contain the resolved factor (1..=4)".into(),
            );
        }
        Ok(())
    }
}

pub fn validate_factor(factor: u32) -> Result<(), String> {
    if factor <= 4 {
        Ok(())
    } else {
        Err("ssaa_factor must be 0 (auto), 1 (off), 2, 3 or 4".into())
    }
}

/// Resolve the configured linear scale. Auto (0) selects 2× below the pixel
/// count of 1280×720, and 1× at or above it. Rotation/aspect do not change the
/// threshold. Explicit values override Auto; limits are checked before allocation.
pub fn effective_factor(output_size: UVec2, configured: u32) -> u32 {
    if configured == 0 {
        if u64::from(output_size.x) * u64::from(output_size.y) < 1280 * 720 {
            2
        } else {
            1
        }
    } else {
        configured
    }
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

fn viewport(
    mode: Res<RenderMode>,
    mut cameras: Query<(
        &SsaaCamera,
        &mut Camera,
        Has<crate::io::image_copy::ImageCopier>,
    )>,
) {
    for (ssaa, mut camera, capture) in &mut cameras {
        if filter_factor(&mode, capture, ssaa.factor) == 1 {
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

pub struct SsaaPlugin;
impl Plugin for SsaaPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER, "ssaa.wgsl", Shader::from_wgsl);
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
    factor: u32,
    normal: bool,
    active: bool,
    request: u64,
}
#[derive(Resource, Default)]
struct Jobs(Vec<Job>);

#[allow(clippy::type_complexity)]
fn extract(
    mut jobs: ResMut<Jobs>,
    mode: Extract<Res<RenderMode>>,
    cameras: Extract<
        Query<(
            &SsaaCamera,
            &Camera,
            Option<&crate::io::image_copy::ImageCopier>,
            Option<&super::ground_truth::GroundTruthCamera>,
            Has<crate::camera::lens::LensCamera>,
        )>,
    >,
) {
    jobs.0.clear();
    for (ssaa, camera, copier, gt, angular) in &cameras {
        let source = if **mode == RenderMode::CoVisibility {
            gt.and_then(|g| g.co_visibility.as_ref())
                .and_then(|v| v.preview.clone())
                .unwrap_or_else(|| ssaa.source.clone())
        } else {
            ssaa.source.clone()
        };
        jobs.0.push(Job {
            camera: ssaa.clone(),
            source,
            factor: filter_factor(&mode, copier.is_some() || angular, ssaa.factor),
            normal: **mode == RenderMode::Normal && copier.is_none(),
            active: camera.is_active,
            request: copier.map_or(0, |c| c.requested_id()),
        });
    }
}

#[derive(Resource, Default)]
struct Pipeline(Option<(CachedRenderPipelineId, BindGroupLayout)>);
#[derive(Resource, Default)]
struct Bindings(HashMap<TextureId, Binding>);
struct Binding {
    source: TextureId,
    uniform: Buffer,
    group: BindGroup,
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
        return;
    }
    let (_, layout) = pipeline.0.get_or_insert_with(|| {
        let layout = BindGroupLayoutDescriptor::new(
            "ssaa_box",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer_sized(false, NonZeroU64::new(16)),
                ),
            ),
        );
        let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ssaa_linear_box_resolve".into()),
            layout: vec![layout.clone()],
            vertex: VertexState {
                shader: SHADER,
                entry_point: Some("vertex".into()),
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
            ..default()
        });
        (id, cache.get_bind_group_layout(&layout))
    });
    let mut live = std::collections::HashSet::new();
    for job in &jobs.0 {
        let (Some(source), Some(output)) =
            (images.get(&job.source), images.get(&job.camera.output))
        else {
            continue;
        };
        let key = output.texture.id();
        live.insert(key);
        if bindings
            .0
            .get(&key)
            .is_none_or(|b| b.source != source.texture.id())
        {
            let uniform = device.create_buffer(&BufferDescriptor {
                label: Some("ssaa_box_params"),
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(
                "ssaa_box",
                layout,
                &BindGroupEntries::sequential((&source.texture_view, uniform.as_entire_binding())),
            );
            bindings.0.insert(
                key,
                Binding {
                    source: source.texture.id(),
                    uniform,
                    group,
                },
            );
        }
        let params = [job.factor, u32::from(job.normal), 0, 0];
        queue.write_buffer(&bindings.0[&key].uniform, 0, bytemuck::cast_slice(&params));
    }
    bindings.0.retain(|key, _| live.contains(key));
}

fn resolve(world: &World, mut context: RenderContext) {
    let jobs = world.resource::<Jobs>();
    if jobs.0.is_empty() {
        return;
    }
    let Some((id, _)) = world.resource::<Pipeline>().0.as_ref() else {
        return;
    };
    let Some(pipeline) = world.resource::<PipelineCache>().get_render_pipeline(*id) else {
        return;
    };
    let images = world.resource::<RenderAssets<GpuImage>>();
    let bindings = world.resource::<Bindings>();
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
        if span.is_none() {
            span = Some(diagnostics.time_span(context.command_encoder(), "ssaa_box_resolve"));
        }
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("ssaa_box_resolve"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &image.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: StoreOp::Store,
                        },
                    })],
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding.group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Copies run later in this same render graph/queue. No host fence.
        job.camera.completed.store(job.request, Ordering::Release);
    }
    if let Some(span) = span {
        span.end(context.command_encoder());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
