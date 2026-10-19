//! CUDA-owned ingress: one arena import, one destination lease and transfer fence
//! per physical room. No cuIpcGetMemHandle on allocator-owned/interoperable memory.
use super::*;
use crate::gpu::cuda::{memory::Mapping, semaphore::CudaSemaphore};
use burn::cubecl::cuda::CudaRuntime;
use cudarc::driver::{CudaContext, CudaStream, sys};
use std::ffi::c_void;
type CudaServer = <CudaRuntime as cubecl_runtime::runtime::Runtime>::Server;

pub(super) struct Importer {
    device: Device,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    mapping: Mapping,
    ready: CudaSemaphore,
    allocation: wire::Allocation,
    // Keep duplicate OS references through all copies, including producer loss.
    _fd_leases: Vec<OwnedFd>,
    reservations: Vec<Box<dyn Any + Send>>,
}
impl Importer {
    pub fn new(
        device: Device,
        uuid: [u8; 16],
        allocation: wire::Allocation,
        mut fds: Vec<OwnedFd>,
    ) -> Result<Self> {
        ensure!(fds.len() == 2, "expected memory and semaphore descriptors");
        let probe = primitive(Tensor::<1>::empty([1], &device).reshape([1, 1, 1, 1]))?;
        ensure!(
            !probe.client.is_capturing(),
            "attach CUDA ingress outside graph capture"
        );
        let cubecl::Device::Cuda(cuda) = &probe.device else {
            anyhow::bail!("IPC receiver requires an existing CUDA device");
        };
        let context = CudaContext::new(cuda.index)?;
        ensure!(
            context.uuid()?.bytes.map(|v| v as u8) == uuid,
            "CUDA/Vulkan UUID mismatch; cross-device IPC is unsupported"
        );
        let resource = probe
            .client
            .get_resource::<CudaServer>(probe.handle.clone())?;
        Self::check_context(&context, resource.resource().ptr)?;
        let fd_leases = fds
            .iter()
            .map(OwnedFd::try_clone)
            .collect::<std::io::Result<Vec<_>>>()?;
        let semaphore = fds.pop().unwrap();
        let memory = fds.pop().unwrap();
        let mapping = Mapping::import(
            context.clone(),
            memory,
            allocation.allocation_bytes,
            allocation.bytes,
        )?;
        let ready = CudaSemaphore::import(context.clone(), semaphore)?;
        let stream = context.new_stream()?;
        Ok(Self {
            device,
            context,
            stream,
            mapping,
            ready,
            allocation,
            _fd_leases: fd_leases,
            reservations: Vec::new(),
        })
    }
    fn check_context(context: &CudaContext, ptr: u64) -> Result<()> {
        context.bind_to_thread()?;
        let mut actual = std::ptr::null_mut();
        unsafe {
            sys::cuPointerGetAttribute(
                (&mut actual as *mut sys::CUcontext).cast::<c_void>(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                ptr,
            )
            .result()?;
        }
        ensure!(
            actual == context.cu_ctx(),
            "CUDA destination context mismatch"
        );
        Ok(())
    }
    pub fn copy(
        &mut self,
        spec: &ResolvedCaptureSpec,
        metadata: bevy_zeroverse::sample::Sample,
        ranges: &[wire::ViewRange],
        timeout: Duration,
    ) -> Result<(GpuSample, CudaImportTimings)> {
        let import_started = Instant::now();
        let mut timings = CudaImportTimings::default();
        wire::validate_views(
            ranges,
            spec.scene().num_cameras,
            spec.scene().playback_steps as usize,
            spec.view_bytes,
            self.allocation.bytes,
        )
        .map_err(anyhow::Error::msg)?;
        ensure!(
            metadata.views.len() == spec.views
                && metadata.view_dim as usize == spec.scene().num_cameras,
            "CUDA metadata view identity mismatch"
        );
        for view in &metadata.views {
            let calibration = view
                .calibration
                .as_ref()
                .context("missing camera calibration")?;
            calibration.validate().map_err(anyhow::Error::msg)?;
            ensure!(
                calibration.image_size == [spec.scene().width as u32, spec.scene().height as u32],
                "calibration/image extent mismatch"
            );
            ensure!(
                [
                    &view.color,
                    &view.depth,
                    &view.normal,
                    &view.position,
                    &view.semantic,
                    &view.optical_flow,
                    &view.motion_vectors,
                    &view.co_visibility
                ]
                .iter()
                .all(|v| v.is_empty()),
                "IPC metadata must not contain host images"
            );
        }
        let shape = [
            spec.views * spec.plan.copied.len(),
            spec.scene().height as usize,
            spec.row_pixels,
            4,
        ];
        let allocation_started = Instant::now();
        let storage =
            Tensor::<1>::empty([spec.room_bytes as usize / 4], &self.device).reshape(shape);
        let destination = primitive(storage.clone())?;
        ensure!(
            !destination.client.is_capturing(),
            "IPC ingress cannot execute inside a Burn CUDA graph capture"
        );
        let resource = destination
            .client
            .get_resource::<CudaServer>(destination.handle.clone())?;
        let ptr = resource.resource().ptr;
        ensure!(
            resource.resource().size >= spec.room_bytes,
            "CUDA destination undersized"
        );
        Self::check_context(&self.context, ptr)?;
        self.reservations.push(Box::new(resource));
        timings.tensor_allocation_seconds = allocation_started.elapsed().as_secs_f64();
        let allocation_fence_started = Instant::now();
        bevy::tasks::block_on(destination.client.sync_buffers([&destination.handle]))?;
        timings.allocation_fence_seconds = allocation_fence_started.elapsed().as_secs_f64();
        let enqueue_started = Instant::now();
        // Ready is accepted only after the exporter completed this signal.
        self.ready.wait(&self.stream)?;
        for (index, range) in ranges.iter().enumerate() {
            unsafe {
                sys::cuMemcpyDtoDAsync_v2(
                    ptr + index as u64 * spec.view_bytes,
                    self.mapping.ptr + range.offset,
                    range.bytes as usize,
                    self.stream.cu_stream(),
                )
                .result()?;
            }
        }
        timings.copy_enqueue_seconds = enqueue_started.elapsed().as_secs_f64();
        let start = Instant::now();
        loop {
            let result = unsafe { sys::cuStreamQuery(self.stream.cu_stream()) };
            if result == sys::CUresult::CUDA_SUCCESS {
                break;
            }
            ensure!(
                result == sys::CUresult::CUDA_ERROR_NOT_READY,
                "CUDA transfer failed: {result:?}"
            );
            ensure!(
                start.elapsed() < timeout,
                "CUDA transfer timeout; lane quarantined until reads retire"
            );
            thread::sleep(Duration::from_millis(1));
        }
        timings.transfer_fence_seconds = start.elapsed().as_secs_f64();
        self.reservations.clear();
        let pack_started = Instant::now();
        let planes = spec.plan.copied.len();
        let height = spec.scene().height as usize;
        let color_batch = spec
            .plan
            .index(Plane::Color)
            .map(|plane| {
                color::convert_room(
                    storage.clone(),
                    plane,
                    spec.scene().width as usize,
                    planes,
                    spec.color_output(),
                )
            })
            .transpose()?;
        let tensors = (0..spec.views)
            .map(|i| {
                storage.clone().slice([
                    i * planes..(i + 1) * planes,
                    0..height,
                    0..spec.row_pixels,
                    0..4,
                ])
            })
            .collect();
        let mut sample = pack_sample(spec, &self.device, metadata, tensors, color_batch)?;
        if let Some(contract) = sample
            .metadata
            .indoor_render_metadata
            .as_mut()
            .and_then(|v| v.get_mut("gpu_tensor_contract"))
            .and_then(|v| v.as_object_mut())
        {
            contract.insert("transport".into(),serde_json::json!({"backend":"vulkan_cuda_ipc","host_image_bytes":0,"registered_arenas":1,"registered_bytes":self.allocation.allocation_bytes,"base_gpu_copies_per_attachment":2,"owned_burn_outputs":true,"shared_physical_device_verified_by_uuid":true,"producer_completion_fence":true}));
        }
        timings.tensor_pack_seconds = pack_started.elapsed().as_secs_f64();
        timings.import_seconds = import_started.elapsed().as_secs_f64();
        timings.copied_bytes = spec.room_bytes;
        timings.copied_views = ranges.len();
        Ok((sample, timings))
    }
}
impl Drop for Importer {
    fn drop(&mut self) {
        // Runs on the ingress thread. Never free a mapping/managed reservation
        // while a timed-out copy might still read it. Caller shutdown is bounded.
        let _ = self.stream.synchronize();
    }
}
