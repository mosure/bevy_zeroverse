use super::runtime::*;
use bevy::{
    asset::{load_internal_asset, uuid_handle},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};
use std::{collections::HashMap, num::NonZeroU64, sync::atomic::Ordering};
const SHADER: Handle<Shader> = uuid_handle!("c43e1c96-d9f3-408a-9477-c815481747f0");
#[derive(Debug, Clone, PartialEq, Eq, Hash, SystemSet)]
pub(crate) struct LensResolveLabel;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    lens: [[f32; 4]; 4],
    views: [[[f32; 4]; 4]; 6],
    regions: [[f32; 4]; 6],
    control: [f32; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    preview: [f32; 4],
    colors: [[f32; 4]; 128],
    turbo: [[f32; 4]; 6],
}
struct Job {
    lens: LensCamera,
    gt: Option<crate::render::ground_truth::GroundTruthCamera>,
    active: bool,
    request: u64,
    mode: u32,
    world: Mat4,
    bounds: [Vec3; 2],
    preview: [f32; 4],
}
#[derive(Resource, Default)]
struct Jobs(Vec<Job>);
#[derive(Resource, Default)]
struct State {
    pipeline: Option<(CachedRenderPipelineId, BindGroupLayout)>,
    bindings: HashMap<bevy::asset::AssetId<Image>, Binding>,
}
struct Binding {
    key: Vec<TextureId>,
    group: BindGroup,
    uniform: Buffer,
}
pub(super) fn install(app: &mut App) {
    load_internal_asset!(app, SHADER, "resolve.wgsl", |source: &str, path| {
        Shader::from_wgsl(super::shader(source, ""), path)
    });
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.init_resource::<State>().init_resource::<Jobs>();
    render.add_systems(ExtractSchedule, extract);
    render.add_systems(Render, prepare.in_set(RenderSystems::PrepareResources));
    render.add_systems(
        RenderGraph,
        resolve
            .after(bevy::core_pipeline::schedule::camera_driver)
            .after(crate::render::co_visibility::CoVisibilityLabel)
            .before(crate::render::ssaa::SsaaResolveLabel)
            .in_set(LensResolveLabel)
            .in_set(RenderGraphSystems::Render),
    );
}
#[allow(clippy::type_complexity)]
fn extract(
    mut jobs: ResMut<Jobs>,
    mode: Extract<Res<crate::render::RenderMode>>,
    config: Extract<Res<crate::app::BevyZeroverseConfig>>,
    bounds: Extract<Query<&crate::scene::SceneAabb>>,
    time: Extract<Res<Time>>,
    flow: Extract<Res<crate::render::optical_flow::FlowPreviewSettings>>,
    cameras: Extract<
        Query<(
            &LensCamera,
            &Camera,
            &GlobalTransform,
            Option<&crate::io::image_copy::ImageCopier>,
            Option<&crate::render::ground_truth::GroundTruthCamera>,
        )>,
    >,
) {
    use crate::render::RenderMode::*;
    jobs.0.clear();
    for (lens, camera, transform, copier, gt) in &cameras {
        jobs.0.push(Job {
            lens: lens.clone(),
            world: transform.to_matrix(),
            bounds: bounds
                .single()
                .map_or([Vec3::ZERO, Vec3::ONE], |b| [b.min, b.max]),
            preview: [
                config.depth_format as u32 as f32,
                flow.interval_seconds / time.delta_secs().max(0.000001),
                flow.full_scale_pixels,
                0.,
            ],
            gt: gt.cloned(),
            active: crate::sample::camera_activity::render_active(
                &config,
                camera.is_active,
                copier,
            ),
            request: copier.map_or(0, |c| c.requested_id()),
            mode: match **mode {
                Color => 0,
                Depth => 1,
                Normal => 2,
                Semantic => 3,
                Position => 4,
                CoVisibility => 5,
                OpticalFlow | MotionVectors => 6,
            },
        });
    }
}
fn prepare(
    jobs: Res<Jobs>,
    mut state: ResMut<State>,
    images: Res<RenderAssets<GpuImage>>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    if jobs.0.is_empty() {
        state.bindings.clear();
        return;
    }
    if state.pipeline.is_none() {
        let texture = || texture_2d(TextureSampleType::Float { filterable: false });
        let layout = BindGroupLayoutDescriptor::new(
            "angular_rgb",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer_sized(
                        false,
                        NonZeroU64::new(std::mem::size_of::<Uniform>() as u64),
                    ),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                    texture(),
                ),
            ),
        );
        let id = cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("fisheye_rgb_resolve".into()),
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
        state.pipeline = Some((id, cache.get_bind_group_layout(&layout)));
    }
    state
        .bindings
        .retain(|key, _| jobs.0.iter().any(|j| j.lens.output.id() == *key));
    for job in &jobs.0 {
        if !job.active {
            continue;
        }
        let Some(faces) = job
            .lens
            .faces
            .iter()
            .map(|h| images.get(h))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(output) = images.get(&job.lens.output) else {
            continue;
        };
        let a = job
            .gt
            .as_ref()
            .and_then(|g| images.get(&g.world_depth))
            .unwrap_or(faces[0]);
        let b = job
            .gt
            .as_ref()
            .and_then(|g| images.get(&g.normal_semantic))
            .unwrap_or(faces[0]);
        let flow = job
            .gt
            .as_ref()
            .and_then(|g| g.flow.as_ref())
            .and_then(|h| images.get(h))
            .unwrap_or(a);
        let key: Vec<_> = faces
            .iter()
            .map(|i| i.texture.id())
            .chain([
                a.texture.id(),
                b.texture.id(),
                flow.texture.id(),
                output.texture.id(),
            ])
            .collect();
        let mut calibration = job.lens.calibration.clone();
        if job.mode == 0 {
            let factor =
                output.texture_descriptor.size.width as f32 / calibration.image_size[0] as f32;
            calibration.k[0] = calibration.k[0].map(|v| v * factor);
            calibration.k[1] = calibration.k[1].map(|v| v * factor);
            calibration.image_size = [
                output.texture_descriptor.size.width,
                output.texture_descriptor.size.height,
            ];
        }
        let mut colors = [[0.; 4]; 128];
        for label in crate::render::semantic::SemanticLabel::ALL {
            let id = crate::render::ground_truth::semantic_id(label) as usize;
            if id < 128 {
                colors[id] = label.color().to_linear().to_f32_array();
            }
        }
        let uniform = Uniform {
            lens: calibration.shader_parameters(),
            views: orientations().map(|r| Mat4::from_quat(r.inverse()).to_cols_array_2d()),
            regions: job.lens.regions.map(|r| {
                [
                    r.min.x as f32,
                    r.min.y as f32,
                    r.max.x as f32,
                    r.max.y as f32,
                ]
            }),
            control: [
                job.lens.face_scale,
                job.mode as f32,
                job.lens.near,
                job.lens.far,
            ],
            bounds_min: job.bounds[0].extend(0.).to_array(),
            bounds_max: job.bounds[1].extend(0.).to_array(),
            preview: job.preview,
            turbo: bevy_zeroverse_capture::depth::TURBO_COEFFICIENTS
                .map(|c| [c[0], c[1], c[2], 0.]),
            colors,
        };
        if state
            .bindings
            .get(&job.lens.output.id())
            .is_none_or(|b| b.key != key)
        {
            let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("angular_rgb_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            let group = device.create_bind_group(
                "angular_rgb",
                &state.pipeline.as_ref().unwrap().1,
                &BindGroupEntries::sequential((
                    buffer.as_entire_binding(),
                    &faces[0].texture_view,
                    &faces[1].texture_view,
                    &faces[2].texture_view,
                    &faces[3].texture_view,
                    &faces[4].texture_view,
                    &faces[5].texture_view,
                    &a.texture_view,
                    &b.texture_view,
                    &flow.texture_view,
                )),
            );
            state.bindings.insert(
                job.lens.output.id(),
                Binding {
                    key,
                    uniform: buffer,
                    group,
                },
            );
        } else {
            queue.write_buffer(
                &state.bindings[&job.lens.output.id()].uniform,
                0,
                bytemuck::bytes_of(&uniform),
            );
        }
    }
}
fn resolve(
    world: &World,
    mut context: RenderContext,
    cameras: Query<(
        &bevy::render::camera::ExtractedCamera,
        &bevy::render::view::ExtractedView,
        &bevy::render::view::ViewTarget,
    )>,
) {
    let state = world.resource::<State>();
    let Some((id, _)) = &state.pipeline else {
        return;
    };
    let cache = world.resource::<PipelineCache>();
    if let CachedPipelineState::Err(error) = cache.get_render_pipeline_state(*id) {
        crate::io::channels::report_capture_failure(format!("fisheye resolve pipeline: {error}"));
    }
    let Some(pipeline) = cache.get_render_pipeline(*id) else {
        return;
    };
    let images = world.resource::<RenderAssets<GpuImage>>();
    for job in &world.resource::<Jobs>().0 {
        if !job.active {
            continue;
        }
        // Co-visibility has its own shared-palette preview. Numeric flow is
        // produced by its controlled-time history pass, never a screen warp.
        if job.mode == 5 {
            if job
                .gt
                .as_ref()
                .is_some_and(|g| g.rendered_frame() == Some(g.frame_id))
            {
                job.lens.completed.store(job.request, Ordering::Release);
            }
            continue;
        }
        // All source faces must belong to this exact logical pose and frame.
        // A newly extracted logical camera cannot release incomplete cube faces.
        if job.mode == 0 && job.lens.faces.iter().enumerate().any(|(i,h)| {
            if i>0 && h==&job.lens.faces[0] {return false;}
            let expected = job.world * Mat4::from_quat(orientations()[i]);
            !cameras.iter().any(|(c,v,_)| {
                matches!(&c.target,Some(bevy::camera::NormalizedRenderTarget::Image(t)) if t.handle==*h)
                && v.world_from_view.to_matrix().abs_diff_eq(expected,1e-5)
            })
        }) { continue; }
        if job.mode != 0
            && job
                .gt
                .as_ref()
                .is_none_or(|g| g.geometry_frame() != Some(g.frame_id))
        {
            continue;
        }
        let (Some(output), Some(binding)) = (
            images.get(&job.lens.output),
            state.bindings.get(&job.lens.output.id()),
        ) else {
            continue;
        };
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fisheye_rgb_resolve"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &output.texture_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: StoreOp::Store,
                        },
                    })],
                    ..default()
                });
            if job.mode != 0 {
                pass.set_viewport(
                    0.,
                    0.,
                    job.lens.calibration.image_size[0] as f32,
                    job.lens.calibration.image_size[1] as f32,
                    0.,
                    1.,
                );
            }
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding.group, &[]);
            pass.draw(0..3, 0..1);
        }
        job.lens.completed.store(job.request, Ordering::Release);
    }
}
