//! Once-per-recipe GPU baking into ordinary cached PBR images. Rendering needs no readback.
//! CPU-only preparation/export and unsupported programs retain the exact CPU path.
mod kernel;
mod recipe;
mod render;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
use super::{program::MaterialRecipe, MapType};
use bevy::prelude::*;
pub(crate) use render::MaterialBakePlugin;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Resource, Default)]
pub(crate) struct MaterialComputeAvailable(pub bool);

#[derive(Default)]
struct Status {
    encoded: AtomicBool,
    error: Mutex<Option<String>>,
}
#[derive(Clone)]
pub(crate) struct Request {
    pub images: [Handle<Image>; 3],
    recipe: recipe::Recipe,
    /// Exactly the CPU transport's existing point-sampled 32x32 albedo contract.
    albedo: Arc<Vec<[u8; 3]>>,
    status: Arc<Status>,
}
#[derive(Resource, Clone, Default)]
pub(crate) struct Requests(pub Vec<Request>);
impl Requests {
    pub(crate) fn ready(&self) -> bool {
        self.0
            .iter()
            .all(|r| r.status.encoded.load(Ordering::Acquire))
    }
    pub(crate) fn failure(&self) -> Option<String> {
        self.0
            .iter()
            .find_map(|r| r.status.error.lock().unwrap().clone())
    }
    pub(crate) fn albedo(&self, id: AssetId<Image>) -> Option<&[[u8; 3]]> {
        self.0
            .iter()
            .find(|r| r.images[0].id() == id)
            .map(|r| r.albedo.as_slice())
    }
}

pub(super) enum Maps {
    Cpu(Box<[Image; 3]>),
    Gpu(Box<Plan>),
}
pub(super) struct Plan {
    recipe: recipe::Recipe,
    albedo: Vec<[u8; 3]>,
}
impl Maps {
    pub(super) fn prepare(r: &MaterialRecipe, floor: u32, gpu: bool) -> Self {
        if gpu {
            if let Some(recipe) = recipe::Recipe::new(r, floor) {
                let n = r.map_size(floor);
                let mut albedo = Vec::with_capacity(1024);
                for y in 0..32 {
                    for x in 0..32 {
                        // Retain original texel-grid coordinates, truncation, and later sRGB decoding.
                        let t = r.texel(
                            (x * n / 32) as f32 / n as f32,
                            (y * n / 32) as f32 / n as f32,
                            floor,
                        );
                        albedo.push(t.color.map(|v| (v.clamp(0., 1.) * 255.) as u8));
                    }
                }
                return Self::Gpu(Box::new(Plan { recipe, albedo }));
            }
        }
        Self::Cpu(Box::new(r.maps(floor)))
    }
    pub(super) fn insert(
        self,
        images: &mut impl super::super::preparation::AssetStore<Image>,
        requests: &mut Requests,
    ) -> [Handle<Image>; 3] {
        match self {
            Self::Cpu(maps) => (*maps).map(|m| images.add(m)),
            Self::Gpu(plan) => {
                let handles = [MapType::Color, MapType::Normal, MapType::Data].map(|kind| {
                    let mut image = super::mip_chain_image(Vec::new(), plan.recipe.meta[1], kind);
                    image.data = None;
                    image.texture_descriptor.usage |=
                        bevy::render::render_resource::TextureUsages::COPY_SRC;
                    images.add(image)
                });
                requests.0.push(Request {
                    images: handles.clone(),
                    recipe: plan.recipe,
                    albedo: Arc::new(plan.albedo),
                    status: Arc::default(),
                });
                handles
            }
        }
    }
}

/// Actual per-room GPU coverage, independent of the requested configuration.
#[derive(Resource, Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MaterialBakeStatistics {
    pub gpu_materials: usize,
    pub synthesized_base_texels: u64,
}
impl Requests {
    pub(crate) fn statistics(&self) -> MaterialBakeStatistics {
        MaterialBakeStatistics {
            gpu_materials: self.0.len(),
            synthesized_base_texels: self
                .0
                .iter()
                .map(|r| u64::from(r.recipe.meta[1]).pow(2))
                .sum(),
        }
    }
}
