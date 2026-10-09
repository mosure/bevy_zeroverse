//! Capture waits for asset uploads and material preparation, not just shaders.
use bevy::{
    asset::{RenderAssetUsages, UntypedAssetId},
    ecs::change_detection::Tick,
    pbr::PreparedMaterial,
    prelude::*,
    render::{
        erased_render_asset::ErasedRenderAssets,
        material_bind_groups::MaterialBindGroupAllocators,
        mesh::RenderMesh,
        render_asset::RenderAssets,
        render_resource::{CachedPipelineState, PipelineCache},
        texture::GpuImage,
        Extract,
    },
    shader::ShaderCacheError,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

#[derive(Resource, Clone, Default)]
pub struct CapturePipelineReadiness {
    ready: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    pipeline_count: Arc<AtomicU64>,
    missing_assets: Arc<AtomicU64>,
    ready_scene: Arc<AtomicU64>,
}
impl CapturePipelineReadiness {
    pub fn pipeline_count(&self) -> u64 {
        self.pipeline_count.load(Ordering::Acquire)
    }
    pub fn missing_assets(&self) -> u64 {
        self.missing_assets.load(Ordering::Acquire)
    }
    pub fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
    /// An earlier room's prepared assets cannot release this room's capture.
    pub fn ready_for_scene(&self, scene: Entity) -> bool {
        self.ready() && self.ready_scene.load(Ordering::Acquire) == scene.to_bits()
    }
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().unwrap().clone()
    }
}

#[derive(Resource, Default)]
pub(crate) struct ExpectedAssets {
    key: Option<ExpectedAssetsKey>,
    scene: Option<Entity>,
    meshes: Vec<AssetId<Mesh>>,
    images: Vec<AssetId<Image>>,
    materials: Vec<UntypedAssetId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedAssetsKey {
    scene: Option<Entity>,
    meshes: Tick,
    images: Tick,
    materials: Tick,
    future: Option<Tick>,
}

pub(super) fn extract(
    mut expected: ResMut<ExpectedAssets>,
    meshes: Extract<Res<Assets<Mesh>>>,
    images: Extract<Res<Assets<Image>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    scene: Extract<Query<Entity, With<crate::scene::SceneAabbNode>>>,
    future: Extract<
        Option<Res<crate::scene::procedural_indoor::preparation::residency::FutureAssets>>,
    >,
) {
    // These are the resources' actual main-world change ticks, fetched through
    // Extract's main-world SystemState. Comparing snapshots does not depend on
    // the render schedule's tick or its previous run. Resource replacement and
    // asset mutations invalidate the lists, as does future-room promotion.
    let key = ExpectedAssetsKey {
        scene: scene.single().ok(),
        meshes: meshes.last_changed(),
        images: images.last_changed(),
        materials: materials.last_changed(),
        future: future.as_ref().map(|future| future.last_changed()),
    };
    if expected.key == Some(key) {
        return;
    }
    expected.gather(key.scene, &meshes, &images, &materials, future.as_deref());
    expected.key = Some(key);
}

#[cfg(test)]
#[path = "readiness_tests.rs"]
mod cache_tests;

impl ExpectedAssets {
    fn gather(
        &mut self,
        scene: Option<Entity>,
        meshes: &Assets<Mesh>,
        images: &Assets<Image>,
        materials: &Assets<StandardMaterial>,
        future: Option<&crate::scene::procedural_indoor::preparation::residency::FutureAssets>,
    ) {
        self.scene = scene;
        self.meshes.clear();
        self.images.clear();
        self.materials.clear();
        self.meshes.extend(meshes.iter().filter_map(|(id, asset)| {
            (asset.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)
                && future
                    .as_ref()
                    .is_none_or(|future| !future.meshes.contains(&id)))
            .then_some(id)
        }));
        self.images.extend(images.iter().filter_map(|(id, asset)| {
            (asset.asset_usage.contains(RenderAssetUsages::RENDER_WORLD)
                && future
                    .as_ref()
                    .is_none_or(|future| !future.images.contains(&id)))
            .then_some(id)
        }));
        self.materials
            .extend(materials.ids().map(AssetId::untyped).filter(|id| {
                future
                    .as_ref()
                    .is_none_or(|future| !future.materials.contains(id))
            }));
    }
}

pub(crate) fn update(
    cache: Res<PipelineCache>,
    expected: Res<ExpectedAssets>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    allocators: Res<MaterialBindGroupAllocators>,
    readiness: Res<CapturePipelineReadiness>,
) {
    let failure = cache
        .pipelines()
        .find_map(|pipeline| match &pipeline.state {
            CachedPipelineState::Err(
                ShaderCacheError::ShaderNotLoaded(_)
                | ShaderCacheError::ShaderImportNotYetAvailable,
            ) => None,
            CachedPipelineState::Err(error) => Some(error.to_string()),
            _ => None,
        });
    let missing = expected
        .meshes
        .iter()
        .filter(|id| meshes.get(**id).is_none())
        .count()
        + expected
            .images
            .iter()
            .filter(|id| images.get(**id).is_none())
            .count()
        + expected
            .materials
            .iter()
            .filter(|id| {
                materials.get(**id).is_none_or(|material| {
                    allocators
                        .get(&id.type_id())
                        .and_then(|allocator| allocator.get(material.binding.group))
                        .is_none_or(|slab| slab.bind_group().is_none())
                })
            })
            .count();
    let ready = failure.is_none() && missing == 0 && cache.waiting_pipelines().next().is_none();
    readiness.ready_scene.store(
        expected
            .scene
            .filter(|_| ready)
            .map_or(u64::MAX, Entity::to_bits),
        Ordering::Release,
    );
    readiness.ready.store(ready, Ordering::Release);
    *readiness.failure.lock().unwrap() = failure;
    readiness
        .pipeline_count
        .store(cache.pipelines().count() as u64, Ordering::Release);
    readiness
        .missing_assets
        .store(missing as u64, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_uploads_do_not_hide_current_assets() {
        use crate::scene::procedural_indoor::preparation::residency::FutureAssets;
        let mut meshes = Assets::<Mesh>::default();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let current_mesh = meshes.add(Cuboid::default());
        let next_mesh = meshes.add(Cuboid::default());
        let current_image = images.add(Image::default());
        let next_image = images.add(Image::default());
        let current_material = materials.add(StandardMaterial::default());
        let next_material = materials.add(StandardMaterial::default());
        let mut future = FutureAssets {
            meshes: [next_mesh.id()].into_iter().collect(),
            images: [next_image.id()].into_iter().collect(),
            materials: [next_material.id().untyped()].into_iter().collect(),
            ..default()
        };
        let mut expected = ExpectedAssets::default();
        let mut world = World::new();
        let current = world.spawn_empty().id();
        let next = world.spawn_empty().id();
        expected.gather(Some(current), &meshes, &images, &materials, Some(&future));
        assert_eq!(expected.scene, Some(current));
        assert_eq!(expected.meshes, [current_mesh.id()]);
        assert_eq!(expected.images, [current_image.id()]);
        assert_eq!(expected.materials, [current_material.id().untyped()]);

        // Promotion restores the requirement for every promoted asset.
        future.clear();
        expected.gather(Some(next), &meshes, &images, &materials, Some(&future));
        assert_eq!(expected.scene, Some(next));
        assert!(expected.meshes.contains(&next_mesh.id()));
        assert!(expected.images.contains(&next_image.id()));
        assert!(expected.materials.contains(&next_material.id().untyped()));
    }
    #[test]
    fn prepared_assets_from_a_previous_room_cannot_release_a_new_room() {
        let mut world = World::new();
        let previous = world.spawn_empty().id();
        let current = world.spawn_empty().id();
        let ready = CapturePipelineReadiness::default();
        ready.ready.store(true, Ordering::Release);
        ready
            .ready_scene
            .store(previous.to_bits(), Ordering::Release);
        assert!(ready.ready_for_scene(previous));
        assert!(!ready.ready_for_scene(current));
        ready
            .ready_scene
            .store(current.to_bits(), Ordering::Release);
        assert!(ready.ready_for_scene(current));
        ready.ready.store(false, Ordering::Release);
        assert!(!ready.ready_for_scene(current));
    }
}
