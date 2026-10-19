//! CPU wall intervals at the GPU transport boundary. Observing these counters
//! adds no GPU fence, readback or timestamp query. They are not GPU execution
//! durations and do not include the consumer's backward/optimizer work.

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
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
    /// Producer-side in-process CUDA bridge. IPC receivers report their work
    /// separately in `cuda_import`, without changing producer wall times.
    pub cuda: Option<CudaTransferTimings>,
    /// Receiver-side CUDA IPC intervals. Absent in producer/older wire receipts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_import: Option<CudaImportTimings>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CudaTransferTimings {
    pub tensor_allocation_seconds: f64,
    pub allocation_fence_seconds: f64,
    pub release_and_enqueue_seconds: f64,
    pub transfer_fence_seconds: f64,
    pub reacquire_and_retire_seconds: f64,
}

/// Host wall intervals for one room imported by the receiver. These observe
/// existing waits; they add no GPU synchronization, readback or timestamp work.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct CudaImportTimings {
    pub tensor_allocation_seconds: f64,
    /// Existing Burn allocation/dependency fence before external CUDA writes.
    pub allocation_fence_seconds: f64,
    /// Host enqueue of the external semaphore wait and device copies.
    pub copy_enqueue_seconds: f64,
    /// Existing transfer-stream completion wait, including polling delay.
    pub transfer_fence_seconds: f64,
    /// Host color/layout packing and metadata work, including kernel submission
    /// or lazy compilation. Does not fence or measure GPU kernel execution.
    pub tensor_pack_seconds: f64,
    /// Metadata validation through packed sample return; excludes socket waits
    /// and release acknowledgement, and is separate from `producer_seconds`.
    pub import_seconds: f64,
    pub copied_bytes: u64,
    pub copied_views: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_timings_preserve_old_producer_receipts_and_separate_clocks() {
        let producer = GpuCaptureTimings {
            producer_seconds: 2.0,
            cuda: Some(CudaTransferTimings {
                transfer_fence_seconds: 0.25,
                ..Default::default()
            }),
            ..Default::default()
        };
        let older_wire = serde_json::to_value(&producer).unwrap();
        assert!(older_wire.get("cuda_import").is_none());
        let mut received: GpuCaptureTimings = serde_json::from_value(older_wire.clone()).unwrap();
        assert!(received.cuda_import.is_none());
        received.cuda_import = Some(CudaImportTimings {
            allocation_fence_seconds: 0.01,
            transfer_fence_seconds: 0.02,
            import_seconds: 0.05,
            copied_bytes: 1024,
            copied_views: 2,
            ..Default::default()
        });
        let wire = serde_json::to_value(&received).unwrap();
        let decoded: GpuCaptureTimings = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(decoded.producer_seconds, 2.0);
        assert_eq!(decoded.cuda.unwrap().transfer_fence_seconds, 0.25);
        let import = decoded.cuda_import.unwrap();
        assert_eq!(import.import_seconds, 0.05);
        assert_eq!(import.copied_bytes, 1024);
        assert_eq!(import.copied_views, 2);
        for (key, value) in older_wire.as_object().unwrap() {
            assert_eq!(wire.get(key), Some(value));
        }
    }
}
