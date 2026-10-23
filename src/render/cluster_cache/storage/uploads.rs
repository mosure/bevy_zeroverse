//! Pack changed view buffers into one upload. Queue writes allocate staging
//! memory, so issuing two writes per source view is expensive on discrete GPUs.
//! The retained transfer buffer is bounded and never changes light membership,
//! order, buffer capacity, or zero padding.
use super::*;
use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages};

const MAX_RETAINED_BYTES: usize = 32 * 1024 * 1024;

struct Copy {
    buffer: wgpu::Buffer,
    offset: u64,
    start: usize,
    len: usize,
}

#[derive(Resource, Default)]
pub(in crate::render::cluster_cache) struct Uploads {
    data: Vec<u8>,
    copies: Vec<Copy>,
    staging: Option<Buffer>,
}

fn append_padded(data: &mut Vec<u8>, bytes: &[u8], capacity: usize) -> usize {
    assert!(bytes.len() <= capacity && capacity.is_multiple_of(4));
    let start = data.len();
    data.resize(start + capacity, 0);
    data[start..start + bytes.len()].copy_from_slice(bytes);
    start
}

impl Uploads {
    pub(super) fn push(&mut self, binding: BindingResource<'_>, bytes: &[u8]) {
        let BindingResource::Buffer(binding) = binding else {
            unreachable!()
        };
        let len = binding.size.unwrap().get() as usize;
        let start = append_padded(&mut self.data, bytes, len);
        self.copies.push(Copy {
            buffer: binding.buffer.clone(),
            offset: binding.offset,
            start,
            len,
        });
    }

    pub(super) fn submit(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.copies.is_empty() {
            return;
        }
        let limit = (device.limits().max_buffer_size as usize).min(MAX_RETAINED_BYTES);
        if self.copies.len() <= 2 || self.data.len() > limit {
            // Small updates don't need another GPU copy. Unusually large view
            // sets retain the original per-buffer path without growing a cache.
            for copy in &self.copies {
                write(
                    BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &copy.buffer,
                        offset: copy.offset,
                        size: std::num::NonZeroU64::new(copy.len as u64),
                    }),
                    queue,
                    &self.data[copy.start..copy.start + copy.len],
                );
            }
        } else {
            if self
                .staging
                .as_ref()
                .is_none_or(|b| b.size() < self.data.len() as u64)
            {
                self.staging = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("cluster_upload_batch"),
                    size: self.data.len().next_power_of_two().min(limit) as u64,
                    usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            let staging = self.staging.as_ref().unwrap();
            {
                let mut mapped = queue
                    .write_buffer_with(
                        staging,
                        0,
                        std::num::NonZeroU64::new(self.data.len() as u64).unwrap(),
                    )
                    .expect("aligned, bounded cluster upload");
                #[cfg(not(target_arch = "wasm32"))]
                crate::render::upload::copy_upload(mapped.slice(..), &self.data);
                #[cfg(target_arch = "wasm32")]
                mapped.copy_from_slice(&self.data);
            }
            let mut encoder = device.create_command_encoder(&default());
            for copy in &self.copies {
                encoder.copy_buffer_to_buffer(
                    staging,
                    copy.start as u64,
                    &copy.buffer,
                    copy.offset,
                    copy.len as u64,
                );
            }
            // Queue order keeps this upload ahead of all views and shadows. No
            // device wait, readback, mapping callback, or cross-frame aliasing.
            queue.submit([encoder.finish()]);
        }
        self.copies.clear();
        self.data.clear();
        if self.data.capacity() > MAX_RETAINED_BYTES {
            self.data = Vec::new();
        }
        if self.copies.capacity() > 4096 {
            self.copies = Vec::new();
        }
    }
}

pub(in crate::render::cluster_cache) fn flush(
    mut uploads: ResMut<Uploads>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    uploads.submit(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concatenated_views_clear_capacity_padding_after_shrink_and_reuse() {
        let mut data = vec![91; 8192];
        data.clear();
        assert_eq!(append_padded(&mut data, &[1, 2, 3, 4], 16), 0);
        assert_eq!(append_padded(&mut data, &[5; 12], 12), 16);
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
        assert_eq!(&data[4..16], &[0; 12]);
        assert_eq!(&data[16..], &[5; 12]);
        data.clear();
        append_padded(&mut data, &[9; 4], 8);
        assert_eq!(data, [9, 9, 9, 9, 0, 0, 0, 0]);
    }
}
