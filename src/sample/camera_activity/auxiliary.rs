//! Bevy retains point/spot shadow views independently of capture cameras.
//! Their Core3d schedules still dispatch preprocessing during preparation and
//! readback-only updates. Temporarily unhook those roots when there is no camera
//! to shade; restore them immediately after the driver, without touching their
//! textures, visibility, preparation, or the next capture's shadows.
use bevy::{
    core_pipeline::schedule::{camera_driver, RootNonCameraView},
    diagnostic::{Diagnostic, DiagnosticMeasurement, DiagnosticPath, DiagnosticsStore},
    pbr::LightEntity,
    prelude::*,
    render::{
        camera::{ExtractedCamera, SortedCameras},
        renderer::{RenderGraph, RenderGraphSystems},
        Extract, ExtractSchedule, RenderApp,
    },
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Resource, Default)]
struct Enabled(bool);

#[derive(Resource, Default)]
struct Suspended(Vec<(Entity, RootNonCameraView)>, Option<SortedCameras>);

#[derive(Resource, Clone, Default)]
struct Counts(Arc<Work>);

#[derive(Default)]
struct Work {
    camera_frames: AtomicU64,
    camera_views: AtomicU64,
    skipped: AtomicU64,
    unready: AtomicU64,
}

pub(super) fn install(app: &mut App) {
    let args = app.world().resource::<crate::app::BevyZeroverseConfig>();
    if !args.headless || args.editor || !args.image_copiers {
        return;
    }
    let counts = Counts::default();
    app.insert_resource(counts.clone());
    app.init_resource::<DiagnosticsStore>();
    app.add_systems(Last, publish);
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.insert_resource(counts);
        render.init_resource::<Enabled>();
        render.init_resource::<Suspended>();
        render.add_systems(ExtractSchedule, extract);
        render.add_systems(
            RenderGraph,
            suspend
                .after(super::super::render_settling::CaptureRenderReadiness)
                .before(camera_driver)
                .in_set(RenderGraphSystems::Render),
        );
        render.add_systems(
            RenderGraph,
            restore
                .after(camera_driver)
                .in_set(RenderGraphSystems::Render),
        );
    }
}

fn extract(args: Extract<Res<crate::app::BevyZeroverseConfig>>, mut enabled: ResMut<Enabled>) {
    enabled.0 = args.headless && !args.editor && args.image_copiers;
}

fn suspend(world: &mut World) {
    let unready = super::super::render_settling::defer_camera_render(world);
    suspend_views(world, unready);
}

fn suspend_views(world: &mut World, unready: bool) {
    if !world.resource::<Enabled>().0 {
        return;
    }
    let camera_count = world
        .query_filtered::<(), With<ExtractedCamera>>()
        .iter(world)
        .count();
    let defer = unready && camera_count != 0 && world.contains_resource::<SortedCameras>();
    if camera_count != 0 && !defer {
        let counts = &world.resource::<Counts>().0;
        counts.camera_frames.fetch_add(1, Ordering::Relaxed);
        counts
            .camera_views
            .fetch_add(camera_count as u64, Ordering::Relaxed);
        return;
    }
    world.resource_scope(|world, mut suspended: Mut<Suspended>| {
        debug_assert!(suspended.0.is_empty());
        debug_assert!(suspended.1.is_none());
        if defer {
            // Keep ExtractedView and the prepared attachments intact. Only the
            // camera driver's work list is empty, for this unaccepted frame.
            suspended.1 = Some(std::mem::take(&mut *world.resource_mut::<SortedCameras>()));
            world
                .resource::<Counts>()
                .0
                .unready
                .fetch_add(1, Ordering::Relaxed);
        }
        suspended.0.extend(
            world
                .query_filtered::<(Entity, &RootNonCameraView), With<LightEntity>>()
                .iter(world)
                .map(|(entity, root)| (entity, *root)),
        );
        for (entity, _) in &suspended.0 {
            world.entity_mut(*entity).remove::<RootNonCameraView>();
        }
        world
            .resource::<Counts>()
            .0
            .skipped
            .fetch_add(suspended.0.len() as u64, Ordering::Relaxed);
    });
}

fn restore(world: &mut World) {
    world.resource_scope(|world, mut suspended: Mut<Suspended>| {
        if let Some(cameras) = suspended.1.take() {
            world.insert_resource(cameras);
        }
        for (entity, root) in suspended.0.drain(..) {
            if let Ok(mut entity) = world.get_entity_mut(entity) {
                entity.insert(root);
            }
        }
    });
}

fn publish(counts: Res<Counts>, mut store: ResMut<DiagnosticsStore>) {
    for (name, value) in [
        ("capture/work/camera_driver_frames", &counts.0.camera_frames),
        (
            "capture/work/unready_camera_frames_skipped",
            &counts.0.unready,
        ),
        (
            "capture/work/camera_views_scheduled",
            &counts.0.camera_views,
        ),
        (
            "capture/work/idle_auxiliary_views_skipped",
            &counts.0.skipped,
        ),
    ] {
        let path = DiagnosticPath::const_new(name);
        if store.get(&path).is_none() {
            store.add(Diagnostic::new(path.clone()));
        }
        store
            .get_mut(&path)
            .unwrap()
            .add_measurement(DiagnosticMeasurement {
                time: bevy::platform::time::Instant::now(),
                value: value.load(Ordering::Relaxed) as f64,
            });
    }
}

#[cfg(test)]
mod tests;
