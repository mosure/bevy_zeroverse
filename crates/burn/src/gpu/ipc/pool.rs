//! One registered arena per supervised lane, reused only after completed-copy ACK.
use super::*;
use crate::gpu::cuda::{memory::ExportBuffer, semaphore::VulkanSemaphore, vulkan::Exporter};
use std::collections::HashMap;

pub(in crate::gpu) struct ExportPool {
    pub exporter: Exporter,
    pub arena: Arc<ExportBuffer>,
    pub signal: VulkanSemaphore,
    pending: HashMap<u64, u64>,
    next: u64,
    released: bool,
    budget: u64,
}
impl ExportPool {
    pub fn new(setup: &WgpuSetup, spec: &ResolvedCaptureSpec, budget: u64) -> Result<Self> {
        let exporter = Exporter::new(setup)?;
        let bytes = spec
            .room_bytes
            .checked_mul(2)
            .context("capture arena overflow")?;
        ensure!(
            bytes <= budget && bytes <= setup.device.limits().max_buffer_size,
            "capture arena exceeds budget/device limit"
        );
        let arena = Arc::new(ExportBuffer::new(&exporter, bytes)?);
        ensure!(
            arena.raw.allocation_size <= budget,
            "dedicated allocation exceeds capture budget"
        );
        let signal = VulkanSemaphore::new(&exporter)?;
        Ok(Self {
            exporter,
            arena,
            signal,
            pending: HashMap::new(),
            next: 0,
            released: false,
            budget,
        })
    }
    pub fn allocate(&mut self, shape: [usize; 4], token: u64) -> Result<GpuCaptureTarget> {
        ensure!(!self.released, "external capture arena still leased");
        let bytes = shape.into_iter().try_fold(4_u64, |n, v| {
            n.checked_mul(v as u64).context("capture size overflow")
        })?;
        let offset = self.next;
        self.next = offset
            .checked_add(bytes)
            .context("capture offset overflow")?;
        ensure!(
            self.next <= self.arena.bytes && self.next <= self.budget,
            "capture exhausted bounded arena"
        );
        ensure!(
            self.pending.insert(token, offset).is_none(),
            "duplicate target token"
        );
        Ok(GpuCaptureTarget {
            token,
            source_indices: Vec::new(),
            buffer: self.arena.buffer.clone(),
            offset,
            byte_len: bytes,
            lease: self.arena.clone(),
        })
    }
    pub fn retire_rejection(&mut self) -> Result<()> {
        ensure!(
            !self.released,
            "cannot recover an externally leased capture"
        );
        self.next = 0;
        self.pending.clear();
        Ok(())
    }
    pub fn complete(&mut self, timeout: Duration) -> Result<()> {
        ensure!(!self.released, "duplicate external release");
        self.exporter
            .ownership(std::slice::from_ref(&self.arena), true)?;
        self.released = true;
        let submission = self.signal.signal(&self.exporter.setup.queue)?;
        // The receiver never waits for an unsignaled producer after a crash.
        self.exporter.setup.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(timeout),
        })?;
        Ok(())
    }
    pub fn retire(&mut self) -> Result<()> {
        ensure!(self.released, "external arena was not released");
        self.exporter
            .ownership(std::slice::from_ref(&self.arena), false)?;
        self.released = false;
        self.next = 0;
        self.pending.clear();
        Ok(())
    }
}
impl Drop for ExportPool {
    fn drop(&mut self) {
        // Own only this producer queue. Imported FDs retain the allocation if the
        // process exits; no trainer context is reset or synchronized here.
        let _ = self
            .exporter
            .setup
            .device
            .poll(wgpu::PollType::wait_indefinitely());
    }
}
