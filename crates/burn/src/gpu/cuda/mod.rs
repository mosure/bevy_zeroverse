//! Linux Vulkan/CUDA interoperability with published Bevy, wgpu and CubeCL APIs.
//! External memory is staging on the device, not CPU staging. CubeCL cannot yet
//! own foreign allocations or accept their completion events, so one device copy
//! and one producer-side stream fence establish ordinary Burn tensor ownership.
use super::*;
use ash::{khr, vk};
use burn::cubecl::{cuda::CudaRuntime, wgpu::wgpu};
use cudarc::driver::{CudaContext, CudaStream, sys};
use std::ffi::c_void;

mod memory;
mod semaphore;
use memory::SharedBuffer;
use semaphore::Semaphore;
type CudaServer = <CudaRuntime as cubecl_runtime::runtime::Runtime>::Server;

/// Add only supported extensions, and only for an explicitly requested CUDA
/// renderer. Normal WGPU renderers never install these callbacks.
pub(super) fn settings() -> bevy_render_cuda::renderer::raw_vulkan_init::RawVulkanInitSettings {
    let mut settings =
        bevy_render_cuda::renderer::raw_vulkan_init::RawVulkanInitSettings::default();
    // SAFETY: The callback only adds advertised extensions. Bridge::new checks
    // that both were enabled and fails closed if the adapter is unsuitable.
    unsafe {
        settings.add_create_device_callback(|args, adapter, _| {
            for extension in [
                khr::external_memory_fd::NAME,
                khr::external_semaphore_fd::NAME,
            ] {
                if adapter
                    .physical_device_capabilities()
                    .supports_extension(extension)
                    && !args.extensions.contains(&extension)
                {
                    args.extensions.push(extension);
                }
            }
        });
    }
    settings
}

pub(super) struct Bridge {
    device: Device,
    setup: WgpuSetup,
    raw: ash::Device,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    ready: Semaphore,
    pending: HashMap<u64, (Arc<SharedBuffer>, [usize; 4])>,
    available: Vec<Arc<SharedBuffer>>,
    // Retain allocator reservations even on a partially submitted/error batch.
    cuda_leases: Vec<Box<dyn Any + Send>>,
    poisoned: bool,
    pub(super) last_timings: CudaTransferTimings,
}

impl Bridge {
    pub(super) fn new(setup: &WgpuSetup) -> Result<Self> {
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
        let mut context = None;
        for index in 0..CudaContext::device_count()? as usize {
            let candidate = CudaContext::new(index)?;
            if candidate.uuid()?.bytes.map(|v| v as u8) == id.device_uuid {
                context = Some(candidate);
                break;
            }
        }
        let context = context
            .context("no CUDA device matches the Vulkan UUID; refusing cross-device transfer")?;
        let device = burn_tensor_cuda::Device::cuda(context.ordinal());
        let stream = context.new_stream()?;
        let ready = Semaphore::new(setup, &instance, &raw, physical, context.clone())?;
        Ok(Self {
            device,
            setup: setup.clone(),
            raw,
            instance,
            physical,
            family,
            context,
            stream,
            ready,
            pending: HashMap::new(),
            available: Vec::new(),
            cuda_leases: Vec::new(),
            poisoned: false,
            last_timings: CudaTransferTimings::default(),
        })
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    pub(super) fn begin_capture(&self) -> Result<()> {
        ensure!(
            !self.poisoned && self.pending.is_empty(),
            "previous CUDA capture was incomplete; recreate the renderer before retrying"
        );
        Ok(())
    }

    pub(super) fn allocate(&mut self, shape: [usize; 4], token: u64) -> Result<GpuCaptureTarget> {
        ensure!(
            !self.poisoned,
            "CUDA capture failed; recreate the renderer before retrying"
        );
        let bytes = checked_bytes(shape)?;
        ensure!(
            bytes <= self.setup.device.limits().max_buffer_size,
            "CUDA capture allocation exceeds the wgpu buffer limit"
        );
        ensure!(
            usize::try_from(bytes).is_ok(),
            "CUDA capture size exceeds the host address type"
        );
        let buffer = match self
            .available
            .iter()
            // Idle targets may still be held by a retiring camera/render-world
            // packet. Reuse only after all of those leases have been released.
            .position(|buffer| buffer.bytes == bytes && Arc::strong_count(buffer) == 1)
        {
            Some(index) => self.available.swap_remove(index),
            None => Arc::new(SharedBuffer::new(self, bytes)?),
        };
        ensure!(
            !self.pending.contains_key(&token),
            "duplicate CUDA capture token"
        );
        self.pending.insert(token, (buffer.clone(), shape));
        Ok(GpuCaptureTarget {
            token,
            buffer: buffer.buffer.clone(),
            offset: 0,
            byte_len: bytes,
            lease: buffer,
        })
    }

    pub(super) fn finish(
        &mut self,
        tokens: impl Iterator<Item = u64>,
    ) -> Result<HashMap<u64, Tensor<4>>> {
        ensure!(
            !self.poisoned,
            "CUDA capture failed; recreate the renderer before retrying"
        );
        // Any failure after queue submission must prevent buffer/semaphore reuse.
        self.poisoned = true;
        self.last_timings = CudaTransferTimings::default();
        let allocation_started = Instant::now();
        let tokens: Vec<_> = tokens.collect();
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.pending.len(),
            "incomplete CUDA capture batch"
        );
        let mut tensors = HashMap::new();
        let mut destinations = Vec::new();
        let mut buffers = Vec::new();
        let mut producer = None;
        let mut handles = Vec::new();
        for &token in &tokens {
            let (source, shape) = self
                .pending
                .get(&token)
                .context("unknown CUDA capture token")?;
            ensure!(
                !tensors.contains_key(&token),
                "duplicate CUDA completed token"
            );
            // A flat allocation avoids CUDA's optional pitched tensor layouts;
            // render row padding already defines our physical storage layout.
            let tensor =
                Tensor::<1>::empty([source.bytes as usize / 4], &self.device).reshape(*shape);
            let primitive = primitive(tensor.clone())?;
            ensure!(
                !primitive.client.is_capturing(),
                "CUDA interop cannot run inside a Burn graph capture"
            );
            let resource = primitive
                .client
                .get_resource::<CudaServer>(primitive.handle.clone())?;
            producer.get_or_insert_with(|| primitive.client.clone());
            handles.push(primitive.handle.clone());
            let view = resource.resource();
            ensure!(
                view.size >= source.bytes,
                "CUDA tensor allocation is undersized"
            );
            self.context.bind_to_thread()?;
            let mut context = std::ptr::null_mut();
            // SAFETY: The managed resource pins this live CUDA pointer. Verify
            // its context, not merely its ordinal, before an external write.
            unsafe {
                sys::cuPointerGetAttribute(
                    (&mut context as *mut sys::CUcontext).cast::<c_void>(),
                    sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                    view.ptr,
                )
                .result()?;
            }
            ensure!(
                context == self.context.cu_ctx(),
                "Burn CUDA allocation uses a different context"
            );
            destinations.push(view.ptr);
            self.cuda_leases.push(Box::new(resource));
            buffers.push(source.clone());
            tensors.insert(token, tensor);
        }
        // Empty allocations may be asynchronous or recycled. Fence their
        // producer stream once for the whole batch, not once per camera.
        self.last_timings.tensor_allocation_seconds = allocation_started.elapsed().as_secs_f64();
        let fence_started = Instant::now();
        bevy::tasks::block_on(producer.unwrap().sync_buffers(&handles))?;
        self.last_timings.allocation_fence_seconds = fence_started.elapsed().as_secs_f64();
        let enqueue_started = Instant::now();
        self.ownership(&buffers, true)?;
        // Attach the signal only AFTER the release submission. Another Bevy
        // submission may consume it, but can never signal before that release.
        self.ready.signal(&self.setup.queue)?;
        self.context.bind_to_thread()?;
        self.ready.wait(&self.stream)?;
        for (source, destination) in buffers.iter().zip(&destinations) {
            // SAFETY: Both pointers are on the verified primary context, leases
            // remain held, and the external semaphore orders Vulkan writes.
            unsafe {
                sys::cuMemcpyDtoDAsync_v2(
                    *destination,
                    source.mapping.ptr,
                    source.bytes as usize,
                    self.stream.cu_stream(),
                )
                .result()?;
            }
        }
        // CubeCL exposes neither a foreign allocation owner nor an external
        // readiness event. Fence this transfer stream before publishing tensors;
        // never synchronize the training context/device or read pixels on CPU.
        self.last_timings.release_and_enqueue_seconds = enqueue_started.elapsed().as_secs_f64();
        let transfer_started = Instant::now();
        self.stream.synchronize()?;
        self.last_timings.transfer_fence_seconds = transfer_started.elapsed().as_secs_f64();
        let retire_started = Instant::now();
        self.cuda_leases.clear();
        self.ownership(&buffers, false)?;
        // Arming occurs before app.update so the sampler can request a copy in
        // that update. Warmup/retiring views can therefore have unused targets.
        // Keep those on the Vulkan side and never expose them as captured data.
        let capacity = self.pending.len().saturating_mul(2);
        self.available
            .extend(self.pending.drain().map(|(_, (buffer, _))| buffer));
        self.available.truncate(capacity);
        self.setup.device.poll(wgpu::PollType::Poll)?;
        self.poisoned = false;
        self.last_timings.reacquire_and_retire_seconds = retire_started.elapsed().as_secs_f64();
        Ok(tensors)
    }

    fn ownership(&self, buffers: &[Arc<SharedBuffer>], release: bool) -> Result<()> {
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

impl Drop for Bridge {
    fn drop(&mut self) {
        // Also covers errors after enqueue. Keep imports and allocator leases
        // alive until this stream stops touching them. No image data is mapped.
        let _ = self.stream.synchronize();
        let _ = self.setup.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

fn checked_bytes(shape: [usize; 4]) -> Result<u64> {
    ensure!(
        shape.iter().all(|n| *n > 0) && shape[3] == 4,
        "CUDA capture requires nonempty RGBA planes"
    );
    shape.into_iter().try_fold(4_u64, |size, n| {
        size.checked_mul(n as u64)
            .context("CUDA capture size overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_sizes_are_checked() {
        assert_eq!(checked_bytes([3, 512, 512, 4]).unwrap(), 3 * 512 * 512 * 16);
        assert!(checked_bytes([0, 512, 512, 4]).is_err());
        assert!(checked_bytes([3, 512, 512, 3]).is_err());
        assert!(checked_bytes([usize::MAX, 512, 512, 4]).is_err());
    }
}
