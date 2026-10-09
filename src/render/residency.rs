//! Bound obsolete prepass keys and per-view GPU allocation bindings on Bevy 0.20.
//!
//! This does not clear compiled pipelines or material specialization caches.
//! Cleanup runs after rendering. Main view and light keys already have upstream
//! cleanup and are observed without repeating it.

use std::{
    collections::{BTreeMap, HashSet},
    hash::{BuildHasher, Hash},
    sync::{Arc, Mutex},
};

use bevy::{
    pbr::{
        BinUnpackingBindGroups, LightKeyCache, UniformAllocationBindGroups, ViewKeyCache,
        ViewKeyPrepassCache,
    },
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        view::ExtractedView,
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

/// Optional control for matched cache-residency experiments.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct RenderResidencyPolicy {
    pub prune: bool,
}

impl Default for RenderResidencyPolicy {
    fn default() -> Self {
        Self { prune: true }
    }
}

#[derive(Clone, Default, Debug, serde::Serialize)]
pub struct CacheResidency {
    pub before: usize,
    pub after: usize,
    pub capacity: usize,
    pub evicted_total: u64,
}

#[derive(Clone, Default, Debug, serde::Serialize)]
pub struct RenderResidencySnapshot {
    pub frame: u64,
    pub pruning_enabled: bool,
    pub main_world_entities: usize,
    pub render_world_entities: usize,
    pub live_meshes: usize,
    pub live_views: usize,
    pub caches: BTreeMap<String, CacheResidency>,
}

#[derive(Resource, Clone, Default)]
pub struct RenderResidencyDiagnostics(Arc<Mutex<RenderResidencySnapshot>>);

impl RenderResidencyDiagnostics {
    pub fn snapshot(&self) -> RenderResidencySnapshot {
        self.0
            .lock()
            .expect("render residency lock poisoned")
            .clone()
    }
}

#[derive(Resource, Default)]
struct LiveMainMeshes {
    live_meshes: usize,
    main_world_entities: usize,
}

pub struct RenderResidencyPlugin;

impl Plugin for RenderResidencyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderResidencyPolicy>();
        app.add_plugins(ExtractResourcePlugin::<RenderResidencyPolicy>::default());
        let diagnostics = RenderResidencyDiagnostics::default();
        app.insert_resource(diagnostics.clone());
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.insert_resource(diagnostics);
            render.init_resource::<LiveMainMeshes>();
            render.add_systems(ExtractSchedule, extract_live_meshes);
            render.add_systems(Render, prune_retired_keys.in_set(RenderSystems::Cleanup));
        }
    }
}

fn extract_live_meshes(
    meshes: Extract<Query<Entity, With<Mesh3d>>>,
    entities: Extract<Query<Entity, Without<bevy::ecs::resource::IsResource>>>,
    mut live: ResMut<LiveMainMeshes>,
) {
    live.live_meshes = meshes.iter().len();
    live.main_world_entities = entities.iter().len();
}

#[derive(bevy::ecs::system::SystemParam)]
struct ViewCaches<'w> {
    light_keys: Option<ResMut<'w, LightKeyCache>>,
    view_keys: Option<ResMut<'w, ViewKeyCache>>,
    prepass_keys: Option<ResMut<'w, ViewKeyPrepassCache>>,
    bin_unpacking: Option<ResMut<'w, BinUnpackingBindGroups>>,
    uniform_allocation: Option<ResMut<'w, UniformAllocationBindGroups>>,
}

fn retain_current<K: Eq + Hash, V, S: BuildHasher>(
    cache: &mut bevy::platform::collections::HashMap<K, V, S>,
    mut is_live: impl FnMut(&K) -> bool,
    prune: bool,
    stats: &mut CacheResidency,
) {
    stats.before = cache.len();
    if prune {
        cache.retain(|key, _| is_live(key));
    }
    stats.after = cache.len();
    stats.capacity = cache.capacity();
    stats.evicted_total += (stats.before - stats.after) as u64;
}

fn prune_retired_keys(
    caches: ViewCaches,
    live_meshes: Res<LiveMainMeshes>,
    views: Query<&ExtractedView>,
    render_entities: Query<Entity, Without<bevy::ecs::resource::IsResource>>,
    policy: Res<RenderResidencyPolicy>,
    diagnostics: Res<RenderResidencyDiagnostics>,
) {
    let live_views: HashSet<_> = views.iter().map(|view| view.retained_view_entity).collect();
    let mut stats = diagnostics
        .0
        .lock()
        .expect("render residency lock poisoned");
    stats.frame += 1;
    stats.pruning_enabled = policy.prune;
    stats.main_world_entities = live_meshes.main_world_entities;
    stats.render_world_entities = render_entities.iter().len();
    stats.live_meshes = live_meshes.live_meshes;
    stats.live_views = live_views.len();
    macro_rules! prune_view {
        ($cache:expr, $name:literal, $prune:expr) => {
            if let Some(mut cache) = $cache {
                retain_current(
                    &mut cache,
                    |key| live_views.contains(key),
                    $prune,
                    stats.caches.entry($name.into()).or_default(),
                );
            }
        };
    }
    // Bevy already retains live shadow keys in prepare_lights. Observe
    // that upstream cleanup without making its lifecycle depend on our policy.
    if let Some(mut cache) = caches.light_keys {
        retain_current(
            &mut cache,
            |key| live_views.contains(key),
            false,
            stats.caches.entry("light_keys".into()).or_default(),
        );
    }
    prune_view!(caches.view_keys, "view_keys", false);
    prune_view!(caches.prepass_keys, "prepass_keys", policy.prune);
    // Bevy prunes BinUnpackingBuffers, but its separate bind-group map
    // only inserts/replaces current (view, phase) keys. Retired cameras and
    // shadow views otherwise retain GPU bindings across every regeneration.
    // Cleanup follows rendering; active views retain all of their phases.
    if let Some(mut cache) = caches.bin_unpacking {
        retain_current(
            &mut cache,
            |key| live_views.contains(&key.view),
            policy.prune,
            stats.caches.entry("bin_unpacking".into()).or_default(),
        );
    }
    // Bevy 0.20 adds a second per-view allocation map. Like unpacking, it
    // replaces active keys but does not retire old camera/shadow keys itself.
    if let Some(mut cache) = caches.uniform_allocation {
        retain_current(
            &mut cache,
            |key| live_views.contains(&key.view),
            policy.prune,
            stats.caches.entry("uniform_allocation".into()).or_default(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::view::RetainedViewEntity;

    #[test]
    fn retained_view_cleanup_preserves_live_keys_and_tracks_eviction() {
        let mut world = World::new();
        let main = world.spawn_empty().id();
        let camera = world.spawn_empty().id();
        let old = RetainedViewEntity::new(main.into(), Some(camera.into()), 0);
        let live = RetainedViewEntity::new(main.into(), Some(camera.into()), 1);
        let mut cache = bevy::platform::collections::HashMap::<RetainedViewEntity, i32>::default();
        cache.insert(old, 11);
        cache.insert(live, 22);
        let mut stats = CacheResidency::default();
        retain_current(&mut cache, |key| *key == live, false, &mut stats);
        assert_eq!(cache.len(), 2);
        assert_eq!(stats.evicted_total, 0);
        retain_current(&mut cache, |key| *key == live, true, &mut stats);
        assert_eq!(cache.get(&live), Some(&22));
        assert!(!cache.contains_key(&old));
        assert_eq!((stats.before, stats.after, stats.evicted_total), (2, 1, 1));
        retain_current(&mut cache, |key| *key == live, true, &mut stats);
        assert_eq!(stats.evicted_total, 1);
    }

    #[test]
    fn unpacking_cleanup_keeps_each_phase_of_live_views() {
        use bevy::render::batching::gpu_preprocessing::SceneUnpackingBuffersKey;
        use std::any::TypeId;

        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let live = RetainedViewEntity::new(entity.into(), None, 0);
        let dead = RetainedViewEntity::new(entity.into(), None, 1);
        let mut cache =
            bevy::platform::collections::HashMap::<SceneUnpackingBuffersKey, i32>::default();
        for (phase, value) in [(TypeId::of::<u32>(), 11), (TypeId::of::<f32>(), 22)] {
            cache.insert(SceneUnpackingBuffersKey { phase, view: live }, value);
            cache.insert(SceneUnpackingBuffersKey { phase, view: dead }, value);
        }
        let mut stats = CacheResidency::default();
        retain_current(&mut cache, |key| key.view == live, true, &mut stats);
        assert_eq!((stats.before, stats.after, stats.evicted_total), (4, 2, 2));
        assert!(cache.keys().all(|key| key.view == live));
        assert_eq!(cache.values().sum::<i32>(), 33);
    }
}
