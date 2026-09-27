//! Capture waits for asset uploads and material preparation, not just shaders.
use bevy::{
    asset::{RenderAssetUsages, UntypedAssetId},
    pbr::PreparedMaterial,
    prelude::*,
    render::{
        erased_render_asset::ErasedRenderAssets,
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
    pub fn failure(&self) -> Option<String> {
        self.failure.lock().unwrap().clone()
    }
}

#[derive(Resource, Default)]
pub(super) struct ExpectedAssets {
    meshes: Vec<AssetId<Mesh>>,
    images: Vec<AssetId<Image>>,
    materials: Vec<UntypedAssetId>,
}

pub(super) fn extract(
    mut expected: ResMut<ExpectedAssets>,
    meshes: Extract<Res<Assets<Mesh>>>,
    images: Extract<Res<Assets<Image>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
) {
    expected.meshes.clear();
    expected.images.clear();
    expected.materials.clear();
    expected
        .meshes
        .extend(meshes.iter().filter_map(|(id, asset)| {
            asset
                .asset_usage
                .contains(RenderAssetUsages::RENDER_WORLD)
                .then_some(id)
        }));
    expected
        .images
        .extend(images.iter().filter_map(|(id, asset)| {
            asset
                .asset_usage
                .contains(RenderAssetUsages::RENDER_WORLD)
                .then_some(id)
        }));
    expected
        .materials
        .extend(materials.ids().map(AssetId::untyped));
}

pub(super) fn update(
    cache: Res<PipelineCache>,
    expected: Res<ExpectedAssets>,
    meshes: Res<RenderAssets<RenderMesh>>,
    images: Res<RenderAssets<GpuImage>>,
    materials: Res<ErasedRenderAssets<PreparedMaterial>>,
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
            .filter(|id| materials.get(**id).is_none())
            .count();
    readiness.ready.store(
        failure.is_none() && missing == 0 && cache.waiting_pipelines().next().is_none(),
        Ordering::Release,
    );
    *readiness.failure.lock().unwrap() = failure;
    readiness
        .pipeline_count
        .store(cache.pipelines().count() as u64, Ordering::Release);
    readiness
        .missing_assets
        .store(missing as u64, Ordering::Release);
}
