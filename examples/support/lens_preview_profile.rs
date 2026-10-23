//! Optional CPU phase intervals. Overlapping schedules/sets are not additive.
use bevy::{
    light::SimulationLightSystems,
    prelude::*,
    render::{Render, RenderApp, RenderSystems},
};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

/// A device poll only waits for work already submitted. With pipelined Bevy
/// rendering, App::update can return before the render thread submits this
/// frame. First fence render submission, then let the caller fence the device.
#[derive(Resource, Clone, Default)]
pub struct FrameFence {
    requested: Arc<AtomicU64>,
    submitted: Arc<AtomicU64>,
}
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct FrameSubmitted;
impl FrameFence {
    pub fn install(app: &mut App) {
        let fence = Self::default();
        app.insert_resource(fence.clone());
        app.sub_app_mut(RenderApp)
            .insert_resource(fence)
            .add_systems(
                Render,
                (|fence: Res<FrameFence>| {
                    fence
                        .submitted
                        .store(fence.requested.load(Ordering::Acquire), Ordering::Release);
                })
                .in_set(FrameSubmitted)
                .after(RenderSystems::Render)
                .after(RenderSystems::Cleanup)
                .after(RenderSystems::PostCleanup),
            );
    }

    pub fn update(app: &mut App) {
        let fence = app.world().resource::<Self>().clone();
        let request = fence.requested.fetch_add(1, Ordering::AcqRel) + 1;
        app.update();
        #[cfg(feature = "multi_threaded")]
        let executor = app
            .world()
            .get_resource::<bevy::ecs::schedule::MainThreadExecutor>()
            .cloned();
        #[cfg(feature = "multi_threaded")]
        let ticker = executor.as_ref().and_then(|e| e.0.ticker());
        let start = Instant::now();
        while fence.submitted.load(Ordering::Acquire) != request {
            assert!(
                start.elapsed().as_secs() < 60,
                "render submission timed out"
            );
            #[cfg(feature = "multi_threaded")]
            if ticker.as_ref().is_some_and(|t| t.try_tick()) {
                continue;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}

#[derive(Clone, Default)]
pub struct Profile {
    enabled: Arc<AtomicBool>,
    values: Arc<Mutex<BTreeMap<String, Vec<f64>>>>,
}
impl Profile {
    pub fn record(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    pub fn add(&self, key: &str, ms: f64) {
        if self.enabled.load(Ordering::Relaxed) {
            let mut values = self.values.lock().unwrap();
            if let Some(samples) = values.get_mut(key) {
                samples.push(ms);
            } else {
                values.insert(key.to_owned(), vec![ms]);
            }
        }
    }
    pub fn take(&self) -> BTreeMap<String, f64> {
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
        (
            "render_batch_phases",
            RenderSystems::PrepareResources,
            RenderSystems::PrepareResourcesBatchPhases,
        ),
        (
            "render_write_phase_buffers",
            RenderSystems::PrepareResourcesBatchPhases,
            RenderSystems::PrepareResourcesWritePhaseBuffers,
        ),
        (
            "render_collect_phase_buffers",
            RenderSystems::PrepareResourcesWritePhaseBuffers,
            RenderSystems::PrepareResourcesCollectPhaseBuffers,
        ),
        (
            "render_flush_resources",
            RenderSystems::PrepareResourcesCollectPhaseBuffers,
            RenderSystems::PrepareResourcesFlush,
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
            (
                start.after(previous).before(set.clone()),
                end.after(set).before(FrameSubmitted),
            ),
        );
    }
    profile
}

struct TimedSystem {
    key: String,
    profile: Profile,
}
impl<S: bevy::ecs::system::System> bevy::ecs::system::Adapt<S> for TimedSystem {
    type In = S::In;
    type Out = S::Out;
    fn adapt(
        &mut self,
        input: <Self::In as bevy::ecs::system::SystemInput>::Inner<'_>,
        run: impl FnOnce(
            bevy::ecs::system::SystemIn<'_, S>,
        ) -> Result<S::Out, bevy::ecs::system::RunSystemError>,
    ) -> Result<Self::Out, bevy::ecs::system::RunSystemError> {
        let start = Instant::now();
        let result = run(input);
        self.profile
            .add(&self.key, start.elapsed().as_secs_f64() * 1000.);
        result
    }
}

// Install after plugin initialization, before the first schedule run. Wrap in
// place: retain all ordering edges, conditions and system-local state. Bevy's
// adapter delegates access declarations and deferred commands to the original.
// Unlike phase markers, these intervals exclude other parallel systems.
pub fn instrument_resources(app: &mut App, profile: &Profile) {
    let world = app.sub_app_mut(RenderApp).world_mut();
    world.schedule_scope(Render, |_, schedule| instrument_schedule(schedule, profile));
}

fn instrument_schedule(schedule: &mut Schedule, profile: &Profile) {
    use bevy::ecs::{
        schedule::{graph::Direction, NodeId, SystemWithAccess},
        system::{AdapterSystem, System},
    };
    let mut memberships = std::collections::HashMap::new();
    let graph = schedule.graph();
    for (key, _, _) in graph.systems.iter() {
        let names: Vec<_> = graph
            .hierarchy()
            .graph()
            .neighbors_directed(NodeId::System(key), Direction::Incoming)
            .filter_map(|node| node.as_set())
            .filter_map(|key| graph.system_sets.get(key))
            .filter(|set| !set.is_anonymous() && set.system_type().is_none())
            .map(|set| format!("{set:?}"))
            .collect();
        memberships.insert(key, names.join("+"));
    }
    let systems = &mut schedule.graph_mut().systems;
    assert!(
        !systems.is_initialized(),
        "profile before the first schedule run"
    );
    let mut names = std::collections::HashMap::new();
    // Release builds may omit Bevy debug names. Identify public hot paths by
    // their system type, and keep unknown nodes distinct by their schedule key.
    macro_rules! name {
        ($system:path) => {
            names.insert(
                IntoSystem::into_system($system).system_type(),
                stringify!($system),
            );
        };
    }
    name!(bevy::pbr::prepare_clusters_for_cpu_clustering);
    name!(bevy::pbr::prepare_lights);
    name!(bevy::render::view::prepare_view_uniforms);
    name!(bevy::core_pipeline::core_3d::prepare_core_3d_depth_textures);
    name!(bevy::core_pipeline::core_3d::prepare_prepass_textures);
    let keys: Vec<_> = systems.iter().map(|(key, _, _)| key).collect();
    // is_initialized only says whether *all* nodes are initialized. Reject
    // partially initialized graphs too: replacing one of their cached access
    // records would otherwise allow conflicting systems to run concurrently.
    assert!(
        keys.iter()
            .all(|&key| systems.get(key).unwrap().access().is_none()),
        "profile before any system initialization"
    );
    for key in keys {
        let slot = systems.get_mut(key).unwrap();
        // Executors recognize this sentinel by type, rather than invoking it.
        if slot.system_type() == std::any::TypeId::of::<bevy::ecs::schedule::ApplyDeferred>() {
            continue;
        }
        let name = slot.name();
        let label = names.get(&slot.system_type()).map_or_else(
            || format!("{}/{key:?}", memberships[&key]),
            |name| (*name).to_owned(),
        );
        if name
            .to_string()
            .contains("lens_preview_profile::Profile::timers")
        {
            continue;
        }
        let original = std::mem::replace(
            slot,
            SystemWithAccess::new(Box::new(IntoSystem::into_system(|| {}))),
        );
        *slot = SystemWithAccess::new(Box::new(AdapterSystem::new(
            TimedSystem {
                key: format!("system/{label}"),
                profile: profile.clone(),
            },
            original,
            name,
        )));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Resource, Default)]
    struct Count(u32);
    fn producer(mut commands: Commands, mut count: Local<u32>) {
        *count += 1;
        commands.insert_resource(Count(*count));
    }
    fn consumer(count: Res<Count>, mut observed: ResMut<VecResource>) {
        observed.0.push(count.0);
    }
    #[derive(Resource, Default)]
    struct VecResource(Vec<u32>);
    #[test]
    fn frame_fence_waits_for_this_render_update() {
        use bevy::{app::SubApp, ecs::schedule::ScheduleLabel};
        let mut app = App::new();
        let mut render = SubApp::new();
        render.update_schedule = Some(Render.intern());
        render.init_schedule(Render).init_resource::<Count>();
        render.add_systems(
            Render,
            (|mut count: ResMut<Count>| count.0 += 1).in_set(RenderSystems::Render),
        );
        app.insert_sub_app(RenderApp, render);
        FrameFence::install(&mut app);
        for frame in 1..=3 {
            FrameFence::update(&mut app);
            assert_eq!(app.sub_app(RenderApp).world().resource::<Count>().0, frame);
            assert_eq!(
                app.world()
                    .resource::<FrameFence>()
                    .submitted
                    .load(Ordering::Acquire),
                u64::from(frame),
            );
        }
    }

    #[test]
    #[should_panic(expected = "before any system initialization")]
    fn profiler_rejects_partially_initialized_access_records() {
        let mut world = World::new();
        let mut schedule = Schedule::default();
        schedule.add_systems(producer);
        schedule.graph_mut().systems.initialize(&mut world);
        schedule.add_systems(|| {});
        instrument_schedule(&mut schedule, &Profile::default());
    }

    #[test]
    fn profiling_preserves_conditions_order_commands_and_local_state() {
        let mut world = World::new();
        world.init_resource::<VecResource>();
        let mut schedule = Schedule::default();
        schedule.add_systems((producer, consumer).chain());
        schedule.add_systems((|| panic!("disabled system ran")).run_if(|| false));
        let profile = Profile::default();
        profile.record(true);
        instrument_schedule(&mut schedule, &profile);
        schedule.run(&mut world);
        schedule.run(&mut world);
        assert_eq!(world.resource::<VecResource>().0, [1, 2]);
        let samples = profile.take();
        assert_eq!(samples.len(), 2);
        assert!(samples.values().all(|ms| ms.is_finite() && *ms >= 0.));
    }

    #[test]
    fn profiling_preserves_explicit_deferred_flushes() {
        use bevy::ecs::schedule::{ApplyDeferred, ScheduleBuildSettings};
        let mut world = World::new();
        world.init_resource::<VecResource>();
        let mut schedule = Schedule::default();
        schedule.set_build_settings(ScheduleBuildSettings {
            auto_insert_apply_deferred: false,
            ..default()
        });
        schedule.add_systems((producer, ApplyDeferred, consumer).chain());
        instrument_schedule(&mut schedule, &Profile::default());
        schedule.run(&mut world);
        assert_eq!(world.resource::<VecResource>().0, [1]);
    }
}
