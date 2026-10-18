//! CPU wall intervals at the GPU transport boundary. Observing these counters
//! adds no GPU fence, readback or timestamp query. They are not GPU execution
//! durations and do not include the consumer's backward/optimizer work.

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuCaptureTimings {
    pub seed: u64,
    pub views: usize,
    pub updates: u64,
    pub waiting_updates: u64,
    pub sequence_reset_seconds: f64,
    pub target_binding_seconds: f64,
    pub app_update_seconds: f64,
    pub interop_finish_seconds: f64,
    pub tensor_pack_seconds: f64,
    /// Entire producer call, including resets and metadata. WGPU returns after
    /// submission; CUDA also waits on its existing allocation/transfer fences.
    pub producer_seconds: f64,
    /// Bytes copied from render textures, including row padding and all planes.
    pub captured_storage_bytes: u64,
    /// Annotation vertex/index uploads during this capture (excludes flow).
    pub geometry_uploaded_bytes: u64,
    pub geometry_uploads: u64,
    pub deformation_uploads: u64,
    pub flow_uploaded_bytes: u64,
    pub flow_pair_reuses: u64,
    pub cuda: Option<CudaTransferTimings>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CudaTransferTimings {
    pub tensor_allocation_seconds: f64,
    pub allocation_fence_seconds: f64,
    pub release_and_enqueue_seconds: f64,
    pub transfer_fence_seconds: f64,
    pub reacquire_and_retire_seconds: f64,
}
