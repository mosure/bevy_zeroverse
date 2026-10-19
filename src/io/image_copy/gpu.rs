//! Queue-ordered capture into consumer-owned GPU storage, without host mapping.
use super::*;

/// One contiguous allocation for a view's RGBA32Float attachments. The lease
/// must own the allocator reservation, not merely a clone of its wgpu buffer.
#[derive(Clone)]
pub struct GpuCaptureTarget {
    pub token: u64,
    /// Empty means all sources; otherwise copy only these unique source indices.
    pub source_indices: Vec<usize>,
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
        let selected = if target.source_indices.is_empty() {
            self.sources.len()
        } else {
            target.source_indices.len()
        };
        let mut indices = target.source_indices.clone();
        indices.sort_unstable();
        if indices.iter().any(|i| *i >= self.sources.len())
            || indices.windows(2).any(|w| w[0] == w[1])
        {
            return Err("GPU copy source indices must be unique and in range".into());
        }
        let required = selected
            .checked_mul(self.padded_row_bytes)
            .and_then(|n| n.checked_mul(self.rows))
            .ok_or("GPU capture size overflow")? as u64;
        if self.format != TextureFormat::Rgba32Float
            || !self.row_bytes.is_multiple_of(16)
            || !target.offset.is_multiple_of(16)
            || target.byte_len < required
            || target
                .offset
                .checked_add(required)
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
    /// Retire speculative bindings that never received a capture request.
    /// Submitted copies retain their own queue-completion leases.
    pub fn unbind_idle_gpu_target(&self) {
        if !self.state.busy.load(Ordering::Acquire) {
            self.state.gpu_target.lock().unwrap().take();
        }
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
