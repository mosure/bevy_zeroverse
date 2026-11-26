//! Observe the actual Core3d output pass and optical resolves. A queue completion
//! callback acknowledges the exact ticket after Bevy submits this frame.
use super::*;
use crate::{
    camera::lens::LensFace,
    io::image_copy::{readiness, CapturePipelineReadiness},
};
use bevy::{
    app::SubApp,
    camera::{CameraOutputMode, NormalizedRenderTarget},
    core_pipeline::{
        schedule::{camera_driver, Core3d},
        upscaling::{upscaling, ViewUpscalingPipeline},
    },
    render::{
        camera::ExtractedCamera,
        renderer::{CurrentView, RenderGraph, RenderGraphSystems, RenderQueue},
        sync_world::RenderEntity,
        view::ViewTarget,
        Extract, ExtractSchedule, Render, RenderSystems,
    },
};
use std::sync::atomic::AtomicBool;

struct Draw<K> {
    key: K,
    seen: AtomicBool,
}
impl<K: PartialEq> Draw<K> {
    fn new(key: K) -> Self {
        Self {
            key,
            seen: AtomicBool::new(false),
        }
    }
    fn mark(checks: &[Self], key: K) {
        if let Some(check) = checks.iter().find(|c| c.key == key) {
            check.seen.store(true, Ordering::Relaxed);
        }
    }
}
#[derive(Resource, Default)]
pub(crate) struct Frame {
    intent: Option<Arc<Intent>>,
    rasters: Vec<Draw<(Entity, AssetId<Image>)>>,
    angular: Vec<Draw<AssetId<Image>>>,
    resolves: Vec<Draw<AssetId<Image>>>,
    started: bool,
    encoded: bool,
}
impl Frame {
    fn all_drawn(&self) -> bool {
        !self.rasters.is_empty()
            && self.rasters.iter().all(|d| d.seen.load(Ordering::Relaxed))
            && self
                .angular
                .iter()
                .chain(&self.resolves)
                .all(|d| d.seen.load(Ordering::Relaxed))
    }
    pub(crate) fn lens(&self, image: AssetId<Image>) {
        if self.started {
            Draw::mark(&self.angular, image);
        }
    }
    pub(crate) fn resolve(&self, image: AssetId<Image>) {
        if self.started {
            Draw::mark(&self.resolves, image);
        }
    }
}
pub(super) fn install(render: &mut SubApp) {
    render
        .init_resource::<Frame>()
        .add_systems(ExtractSchedule, extract);
    render.add_systems(
        RenderGraph,
        (readiness::update, begin)
            .chain()
            .before(camera_driver)
            .in_set(RenderGraphSystems::Render)
            .run_if(pending),
    );
    render.add_systems(Core3d, drawn.after(upscaling).run_if(started));
    render.add_systems(
        RenderGraph,
        (readiness::update, finish)
            .chain()
            .after(crate::render::ssaa::SsaaResolveLabel)
            .after(camera_driver)
            .in_set(RenderGraphSystems::Render)
            .run_if(started),
    );
    render.add_systems(
        Render,
        submitted.in_set(RenderSystems::Cleanup).run_if(encoded),
    );
}
fn pending(frame: Res<Frame>) -> bool {
    frame.intent.is_some()
}
fn started(frame: Res<Frame>) -> bool {
    frame.started
}
fn encoded(frame: Res<Frame>) -> bool {
    frame.encoded
}

fn extract(
    mut frame: ResMut<Frame>,
    state: Extract<Res<SceneCaptureState>>,
    entities: Extract<Query<&RenderEntity>>,
    faces: Extract<Query<(&LensFace, &RenderTarget, &RenderEntity)>>,
) {
    frame.intent = None;
    frame.started = false;
    frame.encoded = false;
    frame.rasters.clear();
    frame.angular.clear();
    frame.resolves.clear();
    let Some(intent) = state
        .intent
        .as_ref()
        .filter(|i| i.completion.load(Ordering::Acquire) == WAITING)
    else {
        return;
    };
    for (v, path) in intent.receipt.views.iter().zip(&intent.paths) {
        let Ok(entity) = entities.get(v.camera) else {
            return;
        };
        frame.rasters.push(Draw::new((entity.id(), path.raster)));
        if let Some(ids) = &path.faces {
            for (i, &id) in ids.iter().enumerate().skip(1) {
                if ids[..i].contains(&id) {
                    continue;
                }
                let Some((_, _, entity)) = faces.iter().find(|(face, t, _)| {
                    face.owner == v.camera
                        && matches!(t,RenderTarget::Image(t) if t.handle.id()==id)
                }) else {
                    return;
                };
                frame.rasters.push(Draw::new((entity.id(), id)));
            }
        }
        if let Some(id) = path.lens {
            frame.angular.push(Draw::new(id));
        }
        if let Some(id) = path.resolve {
            frame.resolves.push(Draw::new(id));
        }
    }
    frame.intent = Some(intent.clone());
}
fn begin(mut frame: ResMut<Frame>, ready: Res<CapturePipelineReadiness>) {
    frame.started = frame
        .intent
        .as_ref()
        .is_some_and(|i| ready.ready_for_scene(i.receipt.root));
}
fn drawn(
    view: Res<CurrentView>,
    frame: Res<Frame>,
    cameras: Query<(&ExtractedCamera, &ViewTarget), With<ViewUpscalingPipeline>>,
) {
    let Ok((camera, target)) = cameras.get(view.0) else {
        return;
    };
    if !matches!(camera.output_mode, CameraOutputMode::Write { .. })
        || target.out_texture().is_none()
    {
        return;
    }
    if let Some(NormalizedRenderTarget::Image(t)) = &camera.target {
        Draw::mark(&frame.rasters, (view.0, t.handle.id()));
    }
}
fn finish(ready: Res<CapturePipelineReadiness>, mut frame: ResMut<Frame>) {
    frame.encoded = frame
        .intent
        .as_ref()
        .is_some_and(|i| ready.ready_for_scene(i.receipt.root))
        && frame.all_drawn();
}
fn submitted(frame: Res<Frame>, queue: Res<RenderQueue>) {
    let Some(intent) = &frame.intent else {
        return;
    };
    let done = intent.completion.clone();
    if done
        .compare_exchange(WAITING, SUBMITTED, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        queue.on_submitted_work_done(move || done.store(COMPLETE, Ordering::Release));
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn trace_test_frame(
    frame: Res<Frame>,
    gpu: Res<CapturePipelineReadiness>,
    mut previous: Local<String>,
) {
    let count = |draws: &[Draw<AssetId<Image>>]| {
        draws
            .iter()
            .filter(|d| d.seen.load(Ordering::Relaxed))
            .count()
    };
    let summary = format!(
        "ticket={:?} resident={} missing={} pipelines={} rasters={}/{} angular={}/{} resolve={}/{}",
        frame.intent.as_ref().map(|i| i.receipt.generation),
        gpu.ready(),
        gpu.missing_assets(),
        gpu.pipeline_count(),
        frame
            .rasters
            .iter()
            .filter(|d| d.seen.load(Ordering::Relaxed))
            .count(),
        frame.rasters.len(),
        count(&frame.angular),
        frame.angular.len(),
        count(&frame.resolves),
        frame.resolves.len()
    );
    if summary != *previous {
        println!("{summary}");
        *previous = summary;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_cold_draw_requires_every_raster_lens_and_final_resolve_in_one_frame() {
        let mut world = World::new();
        let mut images = Assets::<Image>::default();
        let targets: Vec<_> = (0..4).map(|_| images.add(Image::default()).id()).collect();
        let a = (world.spawn_empty().id(), targets[0]);
        let b = (world.spawn_empty().id(), targets[1]);
        let frame = || Frame {
            rasters: vec![Draw::new(a), Draw::new(b)],
            angular: vec![Draw::new(targets[2])],
            resolves: vec![Draw::new(targets[3])],
            ..default()
        };
        let mut first = frame();
        first.lens(targets[2]);
        first.resolve(targets[3]);
        first.started = true;
        Draw::mark(&first.rasters, a);
        Draw::mark(&first.rasters, (b.0, a.1));
        first.lens(targets[2]);
        assert!(!first.all_drawn());
        Draw::mark(&first.rasters, b);
        assert!(!first.all_drawn());
        first.resolve(targets[3]);
        assert!(first.all_drawn());
        assert!(!frame().all_drawn());
    }
}
