//! Vulkan-only export: no CUDA or Burn allocation is initialized by the producer.
use super::*;
pub(in crate::gpu) struct Exporter {
    pub setup: WgpuSetup,
    pub raw: ash::Device,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub family: u32,
    pub uuid: [u8; 16],
}
impl Exporter {
    pub fn new(setup: &WgpuSetup) -> Result<Self> {
        ensure!(
            setup.backend == wgpu::Backend::Vulkan,
            "CUDA interop requires a Vulkan renderer; no CPU fallback is performed"
        );
        ensure!(
            setup.adapter.get_info().vendor == 0x10de,
            "CUDA interop requires an NVIDIA Vulkan adapter"
        );
        // SAFETY: We only inspect the live device; cloned handles are kept alive
        // by setup for the lifetime of every imported object.
        let hal = unsafe { setup.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .context("Vulkan HAL device is unavailable")?;
        for extension in [
            khr::external_memory_fd::NAME,
            khr::external_semaphore_fd::NAME,
        ] {
            ensure!(
                hal.enabled_device_extensions().contains(&extension),
                "CUDA interop needs enabled {extension:?}"
            );
        }
        let raw = hal.raw_device().clone();
        let instance = hal.shared_instance().raw_instance().clone();
        let physical = hal.raw_physical_device();
        let family = hal.queue_family_index();
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: Physical device belongs to this live instance; pNext points to id.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        Ok(Self {
            setup: setup.clone(),
            raw,
            instance,
            physical,
            family,
            uuid: id.device_uuid,
        })
    }
    pub fn ownership(&self, buffers: &[Arc<memory::ExportBuffer>], release: bool) -> Result<()> {
        let mut encoder =
            self.setup
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("zeroverse CUDA ownership"),
                });
        let barriers: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::BufferMemoryBarrier::default()
                    .buffer(buffer.raw.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(if release {
                        self.family
                    } else {
                        vk::QUEUE_FAMILY_EXTERNAL
                    })
                    .dst_queue_family_index(if release {
                        vk::QUEUE_FAMILY_EXTERNAL
                    } else {
                        self.family
                    })
                    .src_access_mask(if release {
                        vk::AccessFlags::TRANSFER_WRITE
                    } else {
                        vk::AccessFlags::empty()
                    })
                    .dst_access_mask(if release {
                        vk::AccessFlags::empty()
                    } else {
                        vk::AccessFlags::TRANSFER_WRITE
                    })
            })
            .collect();
        // SAFETY: The encoder is live, buffers are exclusive allocations on its
        // device, and CUDA only reads between the release and acquire. WGPU's
        // tracked usage remains COPY_DST on both sides of the external interval.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                let hal = hal.context("Vulkan encoder unavailable")?;
                self.raw.cmd_pipeline_barrier(
                    hal.raw_handle(),
                    if release {
                        vk::PipelineStageFlags::TRANSFER
                    } else {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    },
                    if release {
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE
                    } else {
                        vk::PipelineStageFlags::TRANSFER
                    },
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
                Ok::<_, anyhow::Error>(())
            })?;
        }
        self.setup.queue.submit([encoder.finish()]);
        let leases = buffers.to_vec();
        self.setup
            .queue
            .on_submitted_work_done(move || drop(leases));
        Ok(())
    }
}
