//! One future asset set and one further GI probe can upload without entering the
//! active room. Their IDs remain outside current-room readiness until promotion.
//! This scheduling policy is internal and never changes a room's quality.
use super::{
    pipeline::{PreparationJob, Request},
    PreparedIndoor,
};
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
    pub key: Option<Request>,
    #[cfg(not(target_arch = "wasm32"))]
    pub gi: Option<FutureGi>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) gi_only: Option<FutureGi>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) probe_only: Option<ProbeOwnership>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) full_upload_bytes: u64,
}

/// Dispatch permission can be canceled independently of the reserved image.
/// Keeping this ownership excludes a prepared probe until its exact CPU job
/// becomes current, even when the first future room is promoted meanwhile.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct ProbeOwnership {
    key: Request,
    image: AssetId<Image>,
    upload_bytes: u64,
}

/// An immutable request shares its probe and readiness with the CPU job. A
/// canceled or unrelated job can never promote this slot into the active room.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct FutureGi {
    pub key: Request,
    pub request: crate::scene::procedural_indoor::gi::gpu::GpuBakeRequest,
}

impl FutureAssets {
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.images.clear();
        self.materials.clear();
        self.key = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.cancel_gi();
            self.probe_only = None;
            self.full_upload_bytes = 0;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn cancel_gi(&mut self) {
        self.gi = None;
        self.gi_only = None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_requests(&self) -> impl Iterator<Item = &FutureGi> {
        let first = self.gi.as_ref().filter(|gi| {
            self.key.as_ref() == Some(&gi.key) && self.images.contains(&gi.request.image.id())
        });
        let second = self.gi_only.as_ref().filter(|gi| {
            first.is_some()
                && self
                    .key
                    .as_ref()
                    .is_some_and(|key| key.successor() == gi.key)
                && self.probe_only.as_ref().is_some_and(|probe| {
                    probe.key == gi.key && probe.image == gi.request.image.id()
                })
                && self.images.contains(&gi.request.image.id())
        });
        first.into_iter().chain(second)
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    fn gpu_request(&self) -> Option<&FutureGi> {
        self.gpu_requests().next()
    }

    /// The current room returns to the full readiness barrier. Only its exact
    /// consecutive successor may retain a GI-only image outside that barrier.
    pub fn promote(&mut self, current: &Request) {
        #[cfg(not(target_arch = "wasm32"))]
        let retained = self
            .probe_only
            .take()
            .filter(|probe| probe.key == current.successor());
        self.clear();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(probe) = retained {
            self.images.insert(probe.image);
            self.probe_only = Some(probe);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = current;
    }

    /// Only the head of the bounded CPU queue is eligible. No future entities,
    /// cameras or motion jobs enter the active room. Static GI has a separate
    /// render slot and may run only after the current capture copies submit.
    pub fn stage(
        &mut self,
        job: &mut PreparationJob,
        key: &Request,
        copies_submitted: bool,
        images: &mut Assets<Image>,
        materials: &mut Assets<StandardMaterial>,
        meshes: &mut Assets<Mesh>,
    ) -> bool {
        let Some(room) = job.prepared() else {
            return false;
        };
        let staged = room.assets_staged();
        if staged {
            if self.key.as_ref() != Some(key) {
                return false;
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            let retained = self
                .probe_only
                .as_ref()
                .filter(|probe| probe.key == key.successor())
                .cloned();
            #[cfg(not(target_arch = "wasm32"))]
            let extra_bytes = retained.as_ref().map_or(0, |probe| probe.upload_bytes);
            #[cfg(target_arch = "wasm32")]
            let extra_bytes = 0;
            let upload_bytes = room.upload_bytes();
            if !fits_budget(upload_bytes, extra_bytes) {
                return false;
            }
            self.clear();
            self.key = Some(key.clone());
            self.meshes = room.meshes.entries.keys().copied().collect();
            self.images = room.images.entries.keys().copied().collect();
            // A probe staged by the second slot is already resident. Moving
            // that image, rather than cloning it, prevents committing zeros
            // over its independently baked irradiance during promotion.
            self.images
                .extend(room.images.resident.iter().map(Handle::id));
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.full_upload_bytes = upload_bytes;
                if let Some(probe) = retained {
                    self.images.insert(probe.image);
                    self.probe_only = Some(probe);
                }
            }
            self.materials = room
                .materials
                .entries
                .keys()
                .map(|id| id.untyped())
                .collect();
            room.images.upload(images);
            room.materials.upload(materials);
            room.meshes.upload(meshes);
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.stage_gi(key, room.gpu_request.as_ref(), copies_submitted);
        #[cfg(target_arch = "wasm32")]
        let _ = copies_submitted;
        !staged
    }

    /// Poll only the second contiguous CPU-ready job. No wait, construction,
    /// material/mesh upload or trajectory work is introduced for this slot.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn stage_second(
        &mut self,
        job: &mut PreparationJob,
        key: &Request,
        copies_submitted: bool,
        images: &mut Assets<Image>,
    ) {
        if !copies_submitted
            || self.gpu_requests().next().is_none()
            || !self
                .key
                .as_ref()
                .is_some_and(|first| first.successor() == *key)
        {
            return;
        }
        let Some(room) = job.prepared() else {
            return;
        };
        let Some(request) = room.gpu_request.as_ref() else {
            return;
        };
        if !self.stage_probe(&mut room.images, key, request, images) {
            return;
        }
        self.gi_only = Some(FutureGi {
            key: key.clone(),
            request: request.clone(),
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stage_probe(
        &mut self,
        staged: &mut super::StagedAssets<Image>,
        key: &Request,
        request: &crate::scene::procedural_indoor::gi::gpu::GpuBakeRequest,
        images: &mut Assets<Image>,
    ) -> bool {
        if self.gpu_requests().next().is_none()
            || !self
                .key
                .as_ref()
                .is_some_and(|first| first.successor() == *key)
        {
            return false;
        }
        if let Some(probe) = &self.probe_only {
            return probe.key == *key
                && probe.image == request.image.id()
                && self.images.contains(&probe.image);
        }
        let Some((_, image)) = staged.entries.get(&request.image.id()) else {
            return false;
        };
        let upload_bytes = image_upload_bytes(image) + request.gpu_buffer_bytes();
        if !fits_budget(self.full_upload_bytes, upload_bytes) {
            return false;
        }
        let (handle, image) = staged.entries.remove(&request.image.id()).unwrap();
        images
            .insert(handle.id(), image)
            .expect("staged probe uses the live allocator");
        staged.resident.push(handle);
        self.images.insert(request.image.id());
        self.probe_only = Some(ProbeOwnership {
            key: key.clone(),
            image: request.image.id(),
            upload_bytes,
        });
        true
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stage_gi(
        &mut self,
        key: &Request,
        request: Option<&crate::scene::procedural_indoor::gi::gpu::GpuBakeRequest>,
        copies_submitted: bool,
    ) {
        if copies_submitted && self.gi.is_none() && self.key.as_ref() == Some(key) {
            self.gi = request
                .filter(|r| self.images.contains(&r.image.id()))
                .map(|request| FutureGi {
                    key: key.clone(),
                    request: request.clone(),
                });
        }
    }
}

fn fits_budget(full: u64, probe: u64) -> bool {
    full.checked_add(probe)
        .is_some_and(|bytes| bytes <= MAX_FUTURE_UPLOAD_BYTES)
}

fn image_upload_bytes(image: &Image) -> u64 {
    let d = &image.texture_descriptor;
    let base = u64::from(d.size.width)
        * u64::from(d.size.height)
        * u64::from(d.size.depth_or_array_layers)
        * u64::from(d.format.block_copy_size(None).unwrap_or(16));
    // Conservative bound for mip chains and images with no CPU data.
    let allocation = base * if d.mip_level_count > 1 { 2 } else { 1 };
    allocation.max(image.data.as_ref().map_or(0, |data| data.len() as u64))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::gi::gpu::test_request;

    #[test]
    fn future_gi_requires_submitted_copies_and_owned_exact_request() {
        let mut images = Assets::<Image>::default();
        let request = test_request(images.add(Image::default()));
        let key = Request::new(17, &default(), &default(), default());
        let mut future = FutureAssets {
            key: Some(key.clone()),
            ..default()
        };
        future.images.insert(request.image.id());
        future.stage_gi(&key, Some(&request), false);
        assert!(
            future.gpu_request().is_none(),
            "current copies must submit first"
        );
        future.stage_gi(&key.successor(), Some(&request), true);
        assert!(future.gpu_request().is_none(), "different CPU request");
        let unrelated = test_request(images.add(Image::default()));
        future.stage_gi(&key, Some(&unrelated), true);
        assert!(future.gpu_request().is_none(), "unowned probe image");
        future.stage_gi(&key, Some(&request), true);
        assert_eq!(
            future.gpu_request().unwrap().request.image.id(),
            request.image.id()
        );
        future.key = Some(key.successor());
        assert!(
            future.gpu_request().is_none(),
            "stale metadata cannot dispatch"
        );
    }

    #[test]
    fn canceling_speculation_preserves_the_promoted_probe_and_status() {
        let mut images = Assets::<Image>::default();
        let request = test_request(images.add(Image::default()));
        let key = Request::new(17, &default(), &default(), default());
        let mut future = FutureAssets {
            key: Some(key.clone()),
            ..default()
        };
        future.images.insert(request.image.id());
        future.stage_gi(&key, Some(&request), true);
        let promoted = future.gpu_request().unwrap().request.clone();
        // New requests cancel GI immediately, while staged asset exclusions
        // remain until the old CPU queue is consumed or discarded.
        future.cancel_gi();
        assert!(future.gpu_request().is_none());
        assert!(future.images.contains(&request.image.id()));
        assert_eq!(future.key.as_ref(), Some(&key));
        future.clear();
        assert!(future.gpu_request().is_none());
        assert!(
            future.images.is_empty() && future.meshes.is_empty() && future.materials.is_empty()
        );
        assert_eq!(promoted.image.id(), request.image.id());
        assert!(images.get(&promoted.image).is_some());
        assert_eq!(promoted.gpu_buffer_bytes(), request.gpu_buffer_bytes());
    }

    fn probe_assets(
        images: &Assets<Image>,
    ) -> (
        super::super::StagedAssets<Image>,
        crate::scene::procedural_indoor::gi::gpu::GpuBakeRequest,
    ) {
        use super::super::AssetStore;
        use bevy::{asset::RenderAssetUsages, render::render_resource::*};
        let mut staged = super::super::StagedAssets::new(images);
        let image = Image::new_fill(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            TextureDimension::D3,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::default(),
        );
        let request = test_request(staged.add(image));
        (staged, request)
    }

    fn first_slot(images: &mut Assets<Image>, key: &Request) -> FutureAssets {
        let request = test_request(images.add(Image::default()));
        let mut future = FutureAssets {
            key: Some(key.clone()),
            full_upload_bytes: 4096,
            ..default()
        };
        future.images.insert(request.image.id());
        future.stage_gi(key, Some(&request), true);
        future
    }

    #[test]
    fn two_future_gi_slots_are_exact_ordered_and_individually_owned() {
        let mut images = Assets::<Image>::default();
        let key = Request::new(u64::MAX, &default(), &default(), default());
        let mut future = first_slot(&mut images, &key);
        let (mut staged, request) = probe_assets(&images);
        assert!(!future.stage_probe(&mut staged, &key, &request, &mut images));
        assert!(!future.stage_probe(
            &mut staged,
            &key.successor().successor(),
            &request,
            &mut images
        ));
        assert_eq!(
            staged.entries.len(),
            1,
            "nonconsecutive probes must remain CPU-owned"
        );
        let second = key.successor();
        assert!(future.stage_probe(&mut staged, &second, &request, &mut images));
        future.gi_only = Some(FutureGi {
            key: second.clone(),
            request: request.clone(),
        });
        assert_eq!(
            future.gpu_requests().map(|gi| &gi.key).collect::<Vec<_>>(),
            [&key, &second]
        );
        assert!(staged.entries.is_empty());
        assert_eq!(staged.resident.len(), 1);
        assert!(future.images.contains(&request.image.id()));
        // Dispatch metadata cannot introduce a third slot or an unrelated seed.
        future.gi_only.as_mut().unwrap().key = second.successor();
        assert_eq!(future.gpu_requests().count(), 1);
        future.gi_only.as_mut().unwrap().key = second;
        future.images.remove(&request.image.id());
        assert_eq!(future.gpu_requests().count(), 1);
    }

    #[test]
    fn canceled_second_probe_survives_first_promotion_without_overwriting_baked_data() {
        let mut images = Assets::<Image>::default();
        let key = Request::new(17, &default(), &default(), default());
        let second = key.successor();
        let mut future = first_slot(&mut images, &key);
        let (mut staged, request) = probe_assets(&images);
        assert!(future.stage_probe(&mut staged, &second, &request, &mut images));
        future.gi_only = Some(FutureGi {
            key: second.clone(),
            request: request.clone(),
        });
        images
            .get_mut(&request.image)
            .unwrap()
            .data
            .as_mut()
            .unwrap()[0] = 37;
        future.cancel_gi();
        assert_eq!(future.gpu_requests().count(), 0);
        assert_eq!(future.images.len(), 2);
        future.promote(&key);
        assert_eq!(future.images, HashSet::from([request.image.id()]));
        assert!(future.key.is_none());
        assert_eq!(future.probe_only.as_ref().unwrap().key, second);
        // The eventual full-room commit drains only remaining entries. It must
        // never restore the zero probe created by CPU preparation.
        staged.commit(&mut images);
        assert_eq!(
            images.get(&request.image).unwrap().data.as_ref().unwrap()[0],
            37
        );
        future.promote(&second);
        assert!(future.images.is_empty());
        assert!(future.probe_only.is_none());
        assert_eq!(request.readiness.failure(), None);
    }

    #[test]
    fn future_probe_aggregate_budget_rejects_without_consuming_or_allocating_assets() {
        assert!(fits_budget(MAX_FUTURE_UPLOAD_BYTES, 0));
        assert!(fits_budget(MAX_FUTURE_UPLOAD_BYTES - 1, 1));
        assert!(!fits_budget(MAX_FUTURE_UPLOAD_BYTES, 1));
        assert!(!fits_budget(u64::MAX, 1));
        let mut images = Assets::<Image>::default();
        let key = Request::new(17, &default(), &default(), default());
        let mut future = first_slot(&mut images, &key);
        let (mut staged, request) = probe_assets(&images);
        let bytes =
            image_upload_bytes(&staged.entries[&request.image.id()].1) + request.gpu_buffer_bytes();
        future.full_upload_bytes = MAX_FUTURE_UPLOAD_BYTES - bytes + 1;
        assert!(!future.stage_probe(&mut staged, &key.successor(), &request, &mut images));
        assert!(staged.entries.contains_key(&request.image.id()));
        assert!(staged.resident.is_empty());
        assert!(images.get(&request.image).is_none());
        assert!(future.probe_only.is_none());
        future.full_upload_bytes -= 1;
        assert!(future.stage_probe(&mut staged, &key.successor(), &request, &mut images));
        assert_eq!(future.probe_only.as_ref().unwrap().upload_bytes, bytes);
        // Repolling uses the same resident image; it is not uploaded again.
        images
            .get_mut(&request.image)
            .unwrap()
            .data
            .as_mut()
            .unwrap()[0] = 19;
        assert!(future.stage_probe(&mut staged, &key.successor(), &request, &mut images));
        assert_eq!(staged.resident.len(), 1);
        assert_eq!(
            images.get(&request.image).unwrap().data.as_ref().unwrap()[0],
            19
        );
    }

    #[test]
    fn speculative_second_failure_is_delivered_only_when_requested() {
        let mut images = Assets::<Image>::default();
        let key = Request::new(17, &default(), &default(), default());
        let mut future = first_slot(&mut images, &key);
        let mut job = PreparationJob::completed(Err("second room rejected".into()));
        future.stage_second(&mut job, &key.successor(), false, &mut images);
        future.stage_second(&mut job, &key.successor(), true, &mut images);
        assert!(future.probe_only.is_none());
        assert!(future.gi_only.is_none());
        assert_eq!(
            job.take_ready().unwrap().err().as_deref(),
            Some("second room rejected")
        );
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
            .map(|(_, image)| image_upload_bytes(image))
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
        #[cfg(not(target_arch = "wasm32"))]
        let gi = self.gpu_request.as_ref().map_or(0, |r| {
            r.gpu_buffer_bytes()
                + if self.images.entries.contains_key(&r.image.id()) {
                    0
                } else {
                    r.probe_texture_bytes()
                }
        });
        #[cfg(target_arch = "wasm32")]
        let gi = 0;
        images + meshes + gi
    }
}
