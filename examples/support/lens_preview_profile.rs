//! Optional CPU phase intervals. Overlapping schedules/sets are not additive.
use bevy::{
    light::SimulationLightSystems,
    prelude::*,
    render::{Render, RenderApp, RenderSystems},
};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

#[derive(Clone, Default)]
pub struct Profile {
    enabled: Arc<AtomicBool>,
    values: Arc<Mutex<BTreeMap<&'static str, Vec<f64>>>>,
}
impl Profile {
    pub fn record(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    pub fn add(&self, key: &'static str, ms: f64) {
        if self.enabled.load(Ordering::Relaxed) {
            self.values.lock().unwrap().entry(key).or_default().push(ms);
        }
    }
    pub fn take(&self) -> BTreeMap<&'static str, f64> {
        std::mem::take(&mut *self.values.lock().unwrap())
            .into_iter()
            .map(|(k, v)| (k, v.iter().sum::<f64>() / v.len() as f64))
            .collect()
    }
    fn timers(&self, key: &'static str) -> (impl FnMut(), impl FnMut()) {
        let start = Arc::new(Mutex::new(Instant::now()));
        let end = start.clone();
        let profile = self.clone();
        (
            move || *start.lock().unwrap() = Instant::now(),
            move || {
                profile.add(key, end.lock().unwrap().elapsed().as_secs_f64() * 1000.);
            },
        )
    }
}
pub fn install(app: &mut App) -> Profile {
    let profile = Profile::default();
    for (key, set) in [
        (
            "light_assignment",
            SimulationLightSystems::AssignLightsToClusters,
        ),
        (
            "light_cascades",
            SimulationLightSystems::UpdateDirectionalLightCascades,
        ),
        (
            "light_visibility",
            SimulationLightSystems::CheckLightVisibility,
        ),
    ] {
        let (start, end) = profile.timers(key);
        let start = start
            .before(set.clone())
            .after(bevy::camera::CameraUpdateSystems)
            .after(bevy::transform::TransformSystems::Propagate);
        if set == SimulationLightSystems::CheckLightVisibility {
            app.add_systems(
                PostUpdate,
                (
                    start.after(SimulationLightSystems::UpdateLightFrusta),
                    end.after(set),
                ),
            );
        } else if set == SimulationLightSystems::AssignLightsToClusters {
            app.add_systems(
                PostUpdate,
                (
                    start.after(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
                    end.after(set),
                ),
            );
        } else {
            app.add_systems(PostUpdate, (start, end.after(set)));
        }
    }
    let (start, end) = profile.timers("main_schedules");
    app.add_systems(First, start);
    app.add_systems(Last, end);
    let render = app.sub_app_mut(RenderApp);
    for (key, previous, set) in [
        (
            "render_prepare_assets",
            RenderSystems::ExtractCommands,
            RenderSystems::PrepareAssets,
        ),
        (
            "render_prepare_meshes",
            RenderSystems::PrepareAssets,
            RenderSystems::PrepareMeshes,
        ),
        (
            "render_create_views",
            RenderSystems::PrepareMeshes,
            RenderSystems::CreateViews,
        ),
        (
            "render_specialize",
            RenderSystems::CreateViews,
            RenderSystems::Specialize,
        ),
        (
            "render_prepare_views",
            RenderSystems::Specialize,
            RenderSystems::PrepareViews,
        ),
        (
            "render_queue",
            RenderSystems::PrepareViews,
            RenderSystems::Queue,
        ),
        (
            "render_prepare",
            RenderSystems::PhaseSort,
            RenderSystems::Prepare,
        ),
        (
            "render_prepare_resources",
            RenderSystems::PhaseSort,
            RenderSystems::PrepareResources,
        ),
        (
            "render_bind_groups",
            RenderSystems::PrepareResourcesFlush,
            RenderSystems::PrepareBindGroups,
        ),
        ("render_draw", RenderSystems::Prepare, RenderSystems::Render),
        (
            "render_cleanup",
            RenderSystems::Render,
            RenderSystems::Cleanup,
        ),
    ] {
        let (start, end) = profile.timers(key);
        render.add_systems(
            Render,
            (start.after(previous).before(set.clone()), end.after(set)),
        );
    }
    profile
}
