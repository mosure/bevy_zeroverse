//! Storage packing matches Bevy 0.20's CPU cluster preparation. Public Bevy APIs
//! own the bindings; no dependency patch, private layout access or light culling
//! change is needed. Initial allocation still uses Bevy, but is amortized across
//! views/rooms. Subsequent uploads avoid libc ERMS writes to Vulkan BAR memory.
use super::*;
use bevy::render::render_resource::BindingResource;

const STORAGE: BufferBindingType = BufferBindingType::Storage { read_only: true };
const MAX_POOLED_BYTES: u64 = 32 * 1024 * 1024;
const MAX_POOLED_VIEWS: usize = 64;

#[derive(Component)]
pub(super) struct Managed;

#[derive(Resource, Default)]
pub(super) struct Pool {
    available: Vec<ViewClusterBindings>,
    bytes: u64,
}

impl Pool {
    fn take(&mut self, required: [u64; 2]) -> Option<ViewClusterBindings> {
        let index = self
            .available
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let size = sizes(b)?;
                fits(size, required).then_some((i, size[0] + size[1]))
            })
            .min_by_key(|(_, bytes)| *bytes)?
            .0;
        let binding = self.available.swap_remove(index);
        self.bytes -= sizes(&binding).unwrap().iter().sum::<u64>();
        Some(binding)
    }

    fn put(&mut self, binding: ViewClusterBindings) {
        let Some(size) = sizes(&binding) else { return };
        let bytes: u64 = size.iter().sum();
        if bytes > MAX_POOLED_BYTES {
            return;
        }
        // Recent, larger capacities are useful when the camera count or room
        // lights change. Bounds hold even after arbitrarily many regenerations.
        while self.available.len() >= MAX_POOLED_VIEWS || self.bytes + bytes > MAX_POOLED_BYTES {
            let old = self.available.remove(0);
            self.bytes -= sizes(&old).unwrap().iter().sum::<u64>();
        }
        self.bytes += bytes;
        self.available.push(binding);
    }
}

pub(super) fn recycle(
    mut world: bevy::ecs::world::DeferredWorld,
    context: bevy::ecs::lifecycle::HookContext,
) {
    if world.get::<Managed>(context.entity).is_none() {
        return;
    }
    if let Some(mut binding) = world.get_mut::<ViewClusterBindings>(context.entity) {
        let binding = std::mem::replace(&mut *binding, ViewClusterBindings::new(STORAGE));
        if let Some(mut pool) = world.get_resource_mut::<Pool>() {
            pool.put(binding);
        }
    }
}

fn sizes(bindings: &ViewClusterBindings) -> Option<[u64; 2]> {
    let BindingResource::Buffer(indices) = bindings.clusterable_object_index_lists_binding()?
    else {
        return None;
    };
    let BindingResource::Buffer(offsets) = bindings.offsets_and_counts_binding()? else {
        return None;
    };
    if !indices.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        || !offsets.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
    {
        return None;
    }
    Some([indices.size?.get(), offsets.size?.get()])
}

fn fits(capacity: [u64; 2], required: [u64; 2]) -> bool {
    capacity.into_iter().zip(required).all(|(a, b)| a >= b)
}

#[derive(Debug, PartialEq, Eq)]
struct Packed {
    indices: Vec<u32>,
    offsets: Vec<[u32; 8]>,
}

fn pack(
    input: &InputKey,
    objects: &GlobalClusterableObjectMeta,
    reflection: Option<&RenderViewLightProbes<EnvironmentMapLight>>,
    irradiance: Option<&RenderViewLightProbes<IrradianceVolume>>,
) -> Option<Packed> {
    let mut packed = Packed {
        indices: Vec::new(),
        offsets: Vec::new(),
    };
    let mut expected_indices = 0usize;
    for record in &input.0 {
        match *record {
            Record::Counts([point, spot, rect, reflection, irradiance, decals]) => {
                if packed.indices.len() != expected_indices {
                    return None;
                }
                expected_indices = [point, spot, rect, reflection, irradiance, decals]
                    .into_iter()
                    .try_fold(expected_indices, |n, count| n.checked_add(count as usize))?;
                packed.offsets.push([
                    u32::try_from(packed.indices.len()).ok()?,
                    point,
                    spot,
                    rect,
                    reflection,
                    irradiance,
                    decals,
                    0,
                ]);
            }
            Record::Object(main, render, [point, spot, rect, probe, volume, decal]) => {
                // Match the extractor's precedence and its independent probe
                // branches, including an entity with multiple classifications.
                if let Some(render) = render.filter(|_| point || spot || rect || decal) {
                    packed
                        .indices
                        .push(*objects.entity_to_index.get(&render)? as u32);
                }
                if probe {
                    packed.indices.push(
                        reflection
                            .and_then(|p| {
                                p.main_entity_to_render_light_probe_index
                                    .get(&MainEntity::from(main))
                            })
                            .copied()
                            .unwrap_or(u32::MAX),
                    );
                }
                if volume {
                    packed.indices.push(
                        irradiance
                            .and_then(|p| {
                                p.main_entity_to_render_light_probe_index
                                    .get(&MainEntity::from(main))
                            })
                            .copied()
                            .unwrap_or(u32::MAX),
                    );
                }
            }
        }
    }
    if packed.indices.len() != expected_indices {
        // Bevy may still be extracting an object whose render entity is not
        // ready. Preserve its preparation/diagnostics for inconsistent lists,
        // instead of letting padding become additional valid light indices.
        return None;
    }
    // Encase writes one zero element for an empty runtime-sized array.
    if packed.indices.is_empty() {
        packed.indices.push(0);
    }
    if packed.offsets.is_empty() {
        packed.offsets.push([0; 8]);
    }
    Some(packed)
}

fn write(binding: BindingResource<'_>, queue: &RenderQueue, bytes: &[u8]) {
    let BindingResource::Buffer(binding) = binding else {
        unreachable!()
    };
    let mut mapped = queue
        .write_buffer_with(binding.buffer, binding.offset, binding.size.unwrap())
        .expect("aligned cluster storage upload within the binding capacity");
    #[cfg(not(target_arch = "wasm32"))]
    crate::render::upload::copy_upload(mapped.slice(..bytes.len()), bytes);
    #[cfg(target_arch = "wasm32")]
    mapped.slice(..bytes.len()).copy_from_slice(bytes);
    // A pooled allocation can be larger than this view's lists. Fully clear
    // the unused range: stale indices must never become lights in another view.
    let (chunks, mut tail) = mapped.slice(bytes.len()..).into_chunks::<256>();
    for chunk in chunks {
        std::hint::black_box(chunk).write([0; 256]);
    }
    tail.fill(0);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn update(
    entity: Entity,
    input: &InputKey,
    objects: &GlobalClusterableObjectMeta,
    reflection: Option<&RenderViewLightProbes<EnvironmentMapLight>>,
    irradiance: Option<&RenderViewLightProbes<IrradianceVolume>>,
    existing: Option<&ViewClusterBindings>,
    device: &RenderDevice,
    queue: &RenderQueue,
    pool: &mut Pool,
    commands: &mut Commands,
) -> bool {
    let Some(packed) = pack(input, objects, reflection, irradiance) else {
        return false;
    };
    let indices = bytemuck::cast_slice(&packed.indices);
    let offsets = bytemuck::cast_slice(&packed.offsets);
    let required = [indices.len() as u64, offsets.len() as u64];
    let limit = device.limits().max_storage_buffer_binding_size;
    if required.iter().any(|&bytes| bytes > limit) {
        return false;
    }
    let mut allocated = None;
    let bindings =
        if let Some(existing) = existing.filter(|b| sizes(b).is_some_and(|s| fits(s, required))) {
            existing
        } else {
            allocated = Some(pool.take(required).unwrap_or_else(|| {
                let capacity = required.map(|bytes| bytes.next_power_of_two().min(limit));
                let mut binding = ViewClusterBindings::new(STORAGE);
                binding.reserve_indices(capacity[0] as usize / 4);
                binding.reserve_clusters(capacity[1] as usize / 32);
                binding.write_buffers(device, queue);
                binding
            }));
            allocated.as_ref().unwrap()
        };
    write(
        bindings.clusterable_object_index_lists_binding().unwrap(),
        queue,
        indices,
    );
    write(
        bindings.offsets_and_counts_binding().unwrap(),
        queue,
        offsets,
    );
    if let Some(allocated) = allocated {
        commands.entity(entity).insert((allocated, Managed));
    }
    true
}

#[cfg(test)]
mod tests;
