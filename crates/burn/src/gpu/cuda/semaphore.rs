//! Exported binary signal, consumed once after producer completion is observed.
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

pub(in crate::gpu) struct VulkanSemaphore {
    _device: wgpu::Device,
    device: ash::Device,
    instance: ash::Instance,
    raw: vk::Semaphore,
}
impl VulkanSemaphore {
    pub fn new(exporter: &vulkan::Exporter) -> Result<Self> {
        let handle_type = vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
        let mut properties = vk::ExternalSemaphoreProperties::default();
        let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(handle_type);
        unsafe {
            exporter
                .instance
                .get_physical_device_external_semaphore_properties(
                    exporter.physical,
                    &info,
                    &mut properties,
                );
        }
        ensure!(
            properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
            "Vulkan cannot export opaque FD semaphores"
        );
        let mut external = vk::ExportSemaphoreCreateInfo::default().handle_types(handle_type);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut external);
        let raw = unsafe { exporter.raw.create_semaphore(&info, None)? };
        Ok(Self {
            _device: exporter.setup.device.clone(),
            device: exporter.raw.clone(),
            instance: exporter.instance.clone(),
            raw,
        })
    }
    pub fn export_fd(&self) -> Result<OwnedFd> {
        let exporter = khr::external_semaphore_fd::Device::new(&self.instance, &self.device);
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.raw)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        Ok(unsafe { OwnedFd::from_raw_fd(exporter.get_semaphore_fd(&info)?) })
    }
    pub fn signal(&self, queue: &wgpu::Queue) -> Result<wgpu::SubmissionIndex> {
        struct Pending<'a>(&'a wgpu::Queue, vk::Semaphore);
        impl Drop for Pending<'_> {
            fn drop(&mut self) {
                if let Some(queue) = unsafe { self.0.as_hal::<wgpu::hal::api::Vulkan>() } {
                    queue.remove_signal_semaphore(self.1);
                }
            }
        }
        unsafe {
            queue
                .as_hal::<wgpu::hal::api::Vulkan>()
                .context("Vulkan queue unavailable")?
                .add_signal_semaphore(self.raw, None);
        }
        let _pending = Pending(queue, self.raw);
        Ok(queue.submit([]))
    }
}
impl Drop for VulkanSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.raw, None);
        }
    }
}

pub(in crate::gpu) struct CudaSemaphore {
    context: Arc<CudaContext>,
    cuda: sys::CUexternalSemaphore,
}
unsafe impl Send for CudaSemaphore {}
impl CudaSemaphore {
    pub fn import(context: Arc<CudaContext>, fd: OwnedFd) -> Result<Self> {
        context.bind_to_thread()?;
        let descriptor = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
            type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
                fd: fd.as_raw_fd(),
            },
            flags: 0,
            reserved: [0; 16],
        };
        let mut cuda = std::ptr::null_mut();
        unsafe {
            sys::cuImportExternalSemaphore(&mut cuda, &descriptor).result()?;
        }
        let _ = fd.into_raw_fd();
        Ok(Self { context, cuda })
    }
    pub fn wait(&self, stream: &CudaStream) -> Result<()> {
        self.context.bind_to_thread()?;
        let params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = unsafe { std::mem::zeroed() };
        unsafe {
            sys::cuWaitExternalSemaphoresAsync(&self.cuda, &params, 1, stream.cu_stream())
                .result()?;
        }
        Ok(())
    }
}
impl Drop for CudaSemaphore {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            unsafe {
                let _ = sys::cuDestroyExternalSemaphore(self.cuda).result();
            }
        }
    }
}
pub(in crate::gpu) struct Semaphore {
    cuda: CudaSemaphore,
    vulkan: VulkanSemaphore,
}
impl Semaphore {
    pub fn new(exporter: &vulkan::Exporter, context: Arc<CudaContext>) -> Result<Self> {
        let vulkan = VulkanSemaphore::new(exporter)?;
        let cuda = CudaSemaphore::import(context, vulkan.export_fd()?)?;
        Ok(Self { cuda, vulkan })
    }
    pub fn signal(&self, queue: &wgpu::Queue) -> Result<()> {
        self.vulkan.signal(queue).map(|_| ())
    }
    pub fn wait(&self, stream: &CudaStream) -> Result<()> {
        self.cuda.wait(stream)
    }
}
