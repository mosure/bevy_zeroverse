//! Dedicated exportable Vulkan allocations and CUDA mappings. No allocator
//! pointers are guessed, and no ordinary wgpu/CUDA allocation is reinterpreted.
use super::*;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

pub(super) struct RawBuffer {
    // The owning wgpu device must outlive these raw Vulkan handles.
    _device: wgpu::Device,
    device: ash::Device,
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    allocation_size: u64,
}
impl Drop for RawBuffer {
    fn drop(&mut self) {
        // SAFETY: Our last lease drops only after both APIs finish their work.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}

pub(super) struct Mapping {
    context: Arc<CudaContext>,
    external: sys::CUexternalMemory,
    pub ptr: u64,
}
// SAFETY: The mapping is immutable, its context is retained and rebound on drop,
// and Bridge fences all GPU reads before releasing its ownership lease.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
impl Drop for Mapping {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_ok() {
            // SAFETY: This mapping is uniquely owned; no stream still uses it.
            unsafe {
                if self.ptr != 0 {
                    let _ = sys::cuMemFree_v2(self.ptr).result();
                }
                let _ = sys::cuDestroyExternalMemory(self.external).result();
            }
        }
    }
}

pub(super) struct SharedBuffer {
    // Drop the CUDA mapping before the Vulkan owners; WGPU also retains raw
    // through its deferred HAL destructor while a render submission uses it.
    pub mapping: Mapping,
    pub buffer: wgpu::Buffer,
    pub raw: Arc<RawBuffer>,
    pub bytes: u64,
}
impl SharedBuffer {
    pub fn new(bridge: &Bridge, bytes: u64) -> Result<Self> {
        let handle_type = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let mut properties = vk::ExternalBufferProperties::default();
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(handle_type);
        // SAFETY: Live instance/physical device; all output pointers are valid.
        unsafe {
            bridge
                .instance
                .get_physical_device_external_buffer_properties(
                    bridge.physical,
                    &info,
                    &mut properties,
                )
        };
        ensure!(
            properties
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
            "Vulkan cannot export capture buffers as opaque FDs"
        );
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: Extension support was checked during device initialization.
        let buffer = unsafe { bridge.raw.create_buffer(&info, None)? };
        let mut raw = RawBuffer {
            _device: bridge.setup.device.clone(),
            device: bridge.raw.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            allocation_size: 0,
        };
        // SAFETY: Fresh buffer on the same live device.
        let requirements = unsafe { bridge.raw.get_buffer_memory_requirements(buffer) };
        let properties = unsafe {
            bridge
                .instance
                .get_physical_device_memory_properties(bridge.physical)
        };
        let memory_type = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .context("no device-local exportable Vulkan memory type")?;
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        // SAFETY: Requirements and dedicated/export structures describe buffer.
        raw.memory = unsafe { bridge.raw.allocate_memory(&allocate, None)? };
        raw.allocation_size = requirements.size;
        unsafe { bridge.raw.bind_buffer_memory(buffer, raw.memory, 0)? };
        let raw = Arc::new(raw);

        // create_buffer_from_hal requires initialized bytes. Zero on the GPU
        // once per pooled allocation, before importing into wgpu. This cold-path
        // fence never reads data and is absent when the allocation is reused.
        let mut encoder =
            bridge
                .setup
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("initialize CUDA shared storage"),
                });
        // SAFETY: This fresh buffer has no other users; the submission lease
        // below pins it even if polling fails. We leave it in COPY_DST usage.
        unsafe {
            encoder.as_hal_mut::<wgpu::hal::api::Vulkan, _, _>(|hal| {
                let hal = hal.context("Vulkan encoder unavailable")?;
                bridge
                    .raw
                    .cmd_fill_buffer(hal.raw_handle(), buffer, 0, bytes, 0);
                Ok::<_, anyhow::Error>(())
            })?;
        }
        let submission = bridge.setup.queue.submit([encoder.finish()]);
        let lease = raw.clone();
        bridge
            .setup
            .queue
            .on_submitted_work_done(move || drop(lease));
        bridge.setup.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(120)),
        })?;
        let exporter = khr::external_memory_fd::Device::new(&bridge.instance, &bridge.raw);
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(raw.memory)
            .handle_type(handle_type);
        // SAFETY: A successful export returns a new owned fd. CUDA takes fd
        // ownership only if import succeeds; failure closes it via OwnedFd.
        let fd = unsafe { OwnedFd::from_raw_fd(exporter.get_memory_fd(&fd_info)?) };
        bridge.context.bind_to_thread()?;
        let descriptor = sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
            type_: sys::CUexternalMemoryHandleType::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            handle: sys::CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 { fd: fd.as_raw_fd() },
            size: raw.allocation_size,
            flags: sys::CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut external = std::ptr::null_mut();
        unsafe { sys::cuImportExternalMemory(&mut external, &descriptor).result()? };
        let _ = fd.into_raw_fd();
        let mut mapping = Mapping {
            context: bridge.context.clone(),
            external,
            ptr: 0,
        };
        let descriptor = sys::CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
            offset: 0,
            size: bytes,
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: Offset and range match the bound Vulkan buffer and allocation.
        unsafe {
            sys::cuExternalMemoryGetMappedBuffer(&mut mapping.ptr, external, &descriptor)
                .result()?
        };
        let lease = raw.clone();
        // SAFETY: Same logical device, initialized, correctly sized COPY_DST
        // storage. HAL never destroys the buffer itself; the final lease does.
        let hal = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                buffer,
                Box::new(move || drop(lease)),
            )
        };
        let buffer = unsafe {
            bridge
                .setup
                .device
                .create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                    hal,
                    &wgpu::BufferDescriptor {
                        label: Some("zeroverse CUDA shared capture"),
                        size: bytes,
                        usage: wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
        };
        Ok(Self {
            mapping,
            buffer,
            raw,
            bytes,
        })
    }
}
