//! Vulkan producer -> CUDA transfer-stream binary semaphore. Ownership and the
//! signal-before-wait requirement are explicit; each signal has exactly one wait.
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

pub(super) struct Semaphore {
    _device: wgpu::Device,
    device: ash::Device,
    raw: vk::Semaphore,
    context: Arc<CudaContext>,
    cuda: sys::CUexternalSemaphore,
}
impl Semaphore {
    pub fn new(
        setup: &WgpuSetup,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        context: Arc<CudaContext>,
    ) -> Result<Self> {
        let handle_type = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
        let mut properties = vk::ExternalSemaphoreProperties::default();
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(handle_type);
        // SAFETY: Live physical device and valid output pointer.
        unsafe {
            instance.get_physical_device_external_semaphore_properties(
                physical,
                &info,
                &mut properties,
            )
        };
        ensure!(
            properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
            "Vulkan cannot export an opaque FD semaphore"
        );
        let mut export = vk::ExportSemaphoreCreateInfo::default().handle_types(handle_type);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        let raw = unsafe { device.create_semaphore(&info, None)? };
        let mut semaphore = Self {
            _device: setup.device.clone(),
            device: device.clone(),
            raw,
            context,
            cuda: std::ptr::null_mut(),
        };
        let exporter = khr::external_semaphore_fd::Device::new(instance, device);
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(raw)
            .handle_type(handle_type);
        let fd = unsafe { OwnedFd::from_raw_fd(exporter.get_semaphore_fd(&info)?) };
        let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
                fd: fd.as_raw_fd(),
            },
            flags: 0,
            reserved: [0; 16],
        };
        semaphore.context.bind_to_thread()?;
        // SAFETY: The FD names this live exported binary semaphore. Success
        // transfers ownership to CUDA; failure closes it via OwnedFd.
        unsafe { sys::cuImportExternalSemaphore(&mut semaphore.cuda, &desc).result()? };
        let _ = fd.into_raw_fd();
        Ok(semaphore)
    }
    pub fn signal(&self, queue: &wgpu::Queue) -> Result<()> {
        // Remove an unconsumed signal even if queue submission unwinds. A stale
        // raw semaphore must never remain staged after its owner is destroyed.
        struct PendingSignal<'a> {
            queue: &'a wgpu::Queue,
            semaphore: vk::Semaphore,
        }
        impl Drop for PendingSignal<'_> {
            fn drop(&mut self) {
                // SAFETY: The semaphore owner outlives this guard. Removal only
                // affects signals not yet taken by an actual queue submission.
                if let Some(queue) = unsafe { self.queue.as_hal::<wgpu::hal::api::Vulkan>() } {
                    queue.remove_signal_semaphore(self.semaphore);
                }
            }
        }
        // SAFETY: Semaphore and queue share a device. Bridge ensures its prior
        // wait completed before this binary semaphore is signaled again.
        unsafe {
            queue
                .as_hal::<wgpu::hal::api::Vulkan>()
                .context("Vulkan queue unavailable")?
                .add_signal_semaphore(self.raw, None)
        };
        let _pending = PendingSignal {
            queue,
            semaphore: self.raw,
        };
        queue.submit([]);
        Ok(())
    }
    pub fn wait(&self, stream: &CudaStream) -> Result<()> {
        let params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = unsafe { std::mem::zeroed() };
        // SAFETY: Signal was enqueued first; semaphore, context and stream are
        // retained until the CUDA wait/copies complete.
        unsafe {
            sys::cuWaitExternalSemaphoresAsync(&self.cuda, &params, 1, stream.cu_stream())
                .result()?
        };
        Ok(())
    }
}
impl Drop for Semaphore {
    fn drop(&mut self) {
        // Bridge drains its own transfer stream and Vulkan submissions first.
        if self.context.bind_to_thread().is_ok() {
            unsafe {
                if !self.cuda.is_null() {
                    let _ = sys::cuDestroyExternalSemaphore(self.cuda).result();
                }
                self.device.destroy_semaphore(self.raw, None);
            }
        }
    }
}
