//! One future asset set can upload without entering the active room. Its IDs are
//! excluded from current-room readiness; promotion restores the full barrier.
//! This scheduling policy is internal and never changes a room's quality.
use super::{pipeline::PreparationJob, PreparedIndoor};
use bevy::{asset::UntypedAssetId, prelude::*};
use std::collections::HashSet;

// Bound speculation by bytes as well as room count. Larger rooms retain CPU
// lookahead and follow the ordinary upload path without reducing their detail.
const MAX_FUTURE_UPLOAD_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Resource, Default)]
pub(crate) struct FutureAssets {
    pub meshes: HashSet<AssetId<Mesh>>,
    pub images: HashSet<AssetId<Image>>,
    pub materials: HashSet<UntypedAssetId>,
}

impl FutureAssets {
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.images.clear();
        self.materials.clear();
    }

    /// Only the head of the bounded CPU queue is eligible. No future entities,
    /// cameras, probes, GI dispatches or motion jobs are installed here.
    pub fn stage(
        &mut self,
        job: &mut PreparationJob,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
        meshes: &mut Assets<Mesh>,
    ) -> bool {
        let Some(room) = job.prepared() else {
            return false;
        };
        if room.assets_staged() || room.upload_bytes() > MAX_FUTURE_UPLOAD_BYTES {
            return false;
        }
        self.meshes = room.meshes.entries.keys().copied().collect();
        self.images = room.images.entries.keys().copied().collect();
        self.materials = room
            .materials
            .entries
            .keys()
            .map(|id| id.untyped())
            .collect();
        room.images.upload(images);
        room.materials.upload(materials);
        room.meshes.upload(meshes);
        true
    }
}

impl PreparedIndoor {
    pub(crate) fn assets_staged(&self) -> bool {
        !self.meshes.resident.is_empty()
    }

    fn upload_bytes(&self) -> u64 {
        let images: u64 = self
            .images
            .entries
            .values()
            .map(|(_, image)| {
                let d = &image.texture_descriptor;
                let base = u64::from(d.size.width)
                    * u64::from(d.size.height)
                    * u64::from(d.size.depth_or_array_layers)
                    * u64::from(d.format.block_copy_size(None).unwrap_or(16));
                // Conservative bound for mip chains and images with no CPU data.
                let allocation = base * if d.mip_level_count > 1 { 2 } else { 1 };
                allocation.max(image.data.as_ref().map_or(0, |data| data.len() as u64))
            })
            .sum();
        let meshes: u64 = self
            .meshes
            .entries
            .values()
            .map(|(_, mesh)| {
                mesh.get_vertex_buffer_size() as u64
                    + mesh.indices().map_or(0, |indices| indices.len() as u64 * 4)
            })
            .sum();
        images + meshes
    }
}
