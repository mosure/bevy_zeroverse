//! Queue-ordered capture into consumer-owned GPU storage, without host mapping.
use super::*;

/// One contiguous allocation for a view's RGBA32Float attachments. The lease
/// must own the allocator reservation, not merely a clone of its wgpu buffer.
#[derive(Clone)]
pub struct GpuCaptureTarget {
    pub token: u64,
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub byte_len: u64,
    pub lease: Arc<dyn Send + Sync>,
}

#[derive(Clone)]
pub struct GpuCapturedImages {
    pub request_id: u64,
    pub target: GpuCaptureTarget,
}

impl ImageCopier {
    /// (attachments, rows, padded pixels per row, RGBA channels). Padding is
    /// storage only and must be sliced away before model consumption.
    pub fn gpu_shape(&self) -> [usize; 4] {
        [self.sources.len(), self.rows, self.padded_row_bytes / 16, 4]
    }
    pub fn gpu_target_armed(&self) -> bool {
        self.state.gpu_target.lock().unwrap().is_some()
    }
    pub fn bind_gpu_target(&self, target: GpuCaptureTarget) -> Result<(), String> {
        if self.state.busy.load(Ordering::Acquire) {
            return Err("cannot replace an in-flight GPU capture target".into());
        }
        if self.format != TextureFormat::Rgba32Float
            || !self.row_bytes.is_multiple_of(16)
            || !target.offset.is_multiple_of(16)
            || target.byte_len < self.staging_bytes() as u64
            || target
                .offset
                .checked_add(self.staging_bytes() as u64)
                .is_none_or(|end| end > target.buffer.size())
            || !target.buffer.usage().contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(
                "GPU target must be aligned COPY_DST RGBA32Float storage of sufficient size".into(),
            );
        }
        *self.state.gpu_target.lock().unwrap() = Some(target);
        Ok(())
    }
    pub fn take_gpu(&self, id: u64) -> Option<GpuCapturedImages> {
        let mut packet = self.state.gpu_completed.lock().unwrap();
        if !packet.as_ref().is_some_and(|p| p.request_id == id) {
            return None;
        }
        let packet = packet.take();
        self.state.gpu_target.lock().unwrap().take();
        self.state.busy.store(false, Ordering::Release);
        packet
    }
}
