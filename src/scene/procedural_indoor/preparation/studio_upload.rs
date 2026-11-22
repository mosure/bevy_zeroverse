//! Spread render-world preparation over displayed frames. The prepared room owns
//! all strong handles until promotion or cancellation. No future entities render.
use super::*;
use bevy::platform::time::Instant;
use std::time::Duration;
pub(super) mod fence;
pub(in crate::scene::procedural_indoor) use fence::{install, UploadFence};

enum Asset {
    Image(AssetId<Image>),
    Mesh(AssetId<Mesh>),
    Material(AssetId<StandardMaterial>),
}
pub(in crate::scene::procedural_indoor) struct Uploads {
    pub finished: bool,
    pub last_batch: Option<Duration>,
    remaining: Vec<Asset>,
    recipes: usize,
    reflection_recipes: usize,
    slow_frames: u8,
}
impl Uploads {
    pub fn new(room: &PreparedIndoor) -> Self {
        // Publish a bake's own destinations together. An unrelated CPU texture
        // or large mesh must not serialize all uploads ahead of all GPU bakes.
        let bakes = &room.material_set.gpu_bakes;
        let mut seen = std::collections::HashSet::new();
        let destinations: Vec<_> = bakes
            .1
            .iter()
            .flat_map(|r| &r.images)
            .chain(bakes.0.iter().flat_map(|r| &r.images))
            .map(Handle::id)
            .filter(|id| room.images.entries.contains_key(id) && seen.insert(*id))
            .collect();
        let remaining = room
            .materials
            .entries
            .keys()
            .copied()
            .map(Asset::Material)
            .chain(room.meshes.entries.keys().copied().map(Asset::Mesh))
            .chain(
                room.images
                    .entries
                    .keys()
                    .filter(|id| !seen.contains(*id))
                    .copied()
                    .map(Asset::Image),
            )
            .chain(destinations.into_iter().rev().map(Asset::Image))
            .collect();
        Self {
            finished: false,
            last_batch: None,
            remaining,
            recipes: 0,
            reflection_recipes: 0,
            slow_frames: 0,
        }
    }
    pub fn remaining_assets(&self) -> usize {
        self.remaining.len()
    }
    pub fn stage(
        &mut self,
        room: &mut PreparedIndoor,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
        meshes: &mut Assets<Mesh>,
        bakes: &mut materials::gpu::Requests,
        frame: Duration,
    ) -> bool {
        // Give the displayed scene first use of the device. Catching up on uploads
        // after a slow frame compounds the stall, especially on integrated GPUs.
        self.slow_frames = if frame >= Duration::from_millis(80) {
            self.slow_frames.saturating_add(1)
        } else {
            0
        };
        // A software-rendered grid can take a second per displayed frame even
        // in a static room. Maintain a bounded upload rate per wall-clock second
        // there. A single hitch must not trigger a larger follow-up batch.
        let (limit, bytes_limit, slices) = if self.slow_frames >= 3 {
            // At one displayed frame per second the fast-frame byte limit
            // would take over a minute just to stage a 128 MiB furnished room.
            // Target ~10 MiB/s there, with 12 MiB and 2 ms batch budgets. A single asset can exceed either.
            let periods = (frame.as_millis() / 80).clamp(1, 16) as usize;
            (
                16 * periods,
                768 * 1024 * periods as u64,
                (2 * periods).min(8),
            )
        } else if frame >= Duration::from_millis(80) {
            (8, 256 * 1024, 1)
        } else if frame > Duration::from_millis(32) {
            (24, 1024 * 1024, 1)
        } else {
            (48, 2 * 1024 * 1024, 2)
        };
        let start = Instant::now();
        let mut bytes = 0;
        // The byte budget also limits render-side pressure from a cheap main-world insert.
        // Always admit one asset, including an individual mesh larger than the budget.
        for _ in 0..limit {
            let Some(asset) = self.remaining.pop() else {
                break;
            };
            match asset {
                Asset::Image(id) => {
                    bytes += publish(&mut room.images, images, id, residency::image_upload_bytes)
                }
                Asset::Mesh(id) => {
                    bytes += publish(&mut room.meshes, meshes, id, |mesh| {
                        mesh.get_vertex_buffer_size() as u64
                            + mesh
                                .get_index_buffer_bytes()
                                .map_or(0, |indices| indices.len() as u64)
                    })
                }
                Asset::Material(id) => {
                    publish(&mut room.materials, materials, id, |_| 0);
                }
            }
            if bytes >= bytes_limit
                || start.elapsed() >= Duration::from_micros(500 * slices.min(4) as u64)
            {
                break;
            }
        }
        let requested = &room.material_set.gpu_bakes;
        // Supply a small batch per rendered frame. Each recipe shares its original
        // completion flag with the room, so promotion cannot bake it twice.
        let staged = |handle: &Handle<Image>| !room.images.entries.contains_key(&handle.id());
        let end = self.recipes
            + requested.0[self.recipes..]
                .iter()
                .take(slices)
                .take_while(|r| r.images.iter().all(staged))
                .count();
        bakes.0.extend_from_slice(&requested.0[self.recipes..end]);
        self.recipes = end;
        let end = self.reflection_recipes
            + usize::from(
                requested
                    .1
                    .get(self.reflection_recipes)
                    .is_some_and(|r| r.images.iter().all(staged)),
            );
        bakes
            .1
            .extend_from_slice(&requested.1[self.reflection_recipes..end]);
        self.reflection_recipes = end;
        self.remaining.is_empty()
            && self.recipes == requested.0.len()
            && self.reflection_recipes == requested.1.len()
    }
}

fn publish<A: bevy::asset::Asset>(
    staged: &mut StagedAssets<A>,
    assets: &mut Assets<A>,
    id: AssetId<A>,
    size: impl FnOnce(&A) -> u64,
) -> u64 {
    let (handle, asset) = staged.entries.remove(&id).expect("owned studio upload");
    let bytes = size(&asset);
    assets
        .insert(id, asset)
        .expect("staged asset uses the live allocator");
    staged.resident.push(handle);
    bytes
}
