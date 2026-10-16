//! Native GPU-resident JIT training. WGPU shares Bevy's exact device/queue and
//! copies textures into Burn storage without host image transfer or completion
//! waits. Opt-in Linux Vulkan/CUDA interop uses an additional device copy and
//! producer fences. Use the renderer's `device()` for model and optimizer too.
use anyhow::{Context as ContextExt, Result, ensure};
use bevy::{
    prelude::*,
    render::{
        RenderApp,
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
    },
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    io::image_copy::{GpuCaptureTarget, ImageCopier},
    sample::{CaptureFailure, SamplerState, gpu::GpuSampleSink},
    scene::{RegenerateSceneEvent, ZeroverseSceneType, procedural_indoor},
};
use burn::{
    backend::wgpu::{AutoCompiler, Wgpu},
    tensor::{Device, Tensor, wgpu::WgpuSetup},
};
use burn_cubecl::tensor::CubeTensor;
use std::{
    any::Any,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(all(feature = "cuda_interop", target_os = "linux"))]
mod cuda;

#[derive(Clone, Copy)]
enum Transport {
    Wgpu,
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    Cuda,
}

/// Raw planes preserve full precision and the raster oracle contract. All
/// tensors have [height, width, 4] layout, without row padding. Normals are
/// encoded in [0,1]; semantic IDs are integral floats in normal_semantic alpha.
/// Optical flow is forward from this source view; terminal sources are invalid.
pub struct GpuView {
    pub color: Tensor<3>,
    pub world_depth: Tensor<3>,
    pub normal_semantic: Tensor<3>,
    pub optical_flow: Option<Tensor<3>>,
    pub co_visibility: Option<Tensor<3>>,
}
/// Small CPU metadata plus GPU images, ordered timestep-major like Sample::views.
/// CPU metadata's image byte vectors are intentionally empty; this type cannot
/// accidentally be passed to the CPU archive/Dataset API as a captured sample.
pub struct GpuSample {
    pub metadata: bevy_zeroverse::sample::Sample,
    pub views: Vec<GpuView>,
}
impl GpuView {
    /// RGBA depth/normal/position views can be derived on the training device.
    pub fn depth(&self) -> Tensor<3> {
        self.world_depth.clone().slice([
            0..self.world_depth.dims()[0],
            0..self.world_depth.dims()[1],
            3..4,
        ])
    }
    pub fn semantic_ids(&self) -> Tensor<3> {
        self.normal_semantic.clone().slice([
            0..self.normal_semantic.dims()[0],
            0..self.normal_semantic.dims()[1],
            3..4,
        ])
    }
}

/// Persistent, full-quality indoor renderer. Scene lookahead and tensor
/// allocation are automatic; consumers retain samples for as long as needed.
struct Renderer {
    app: App,
    device: Device,
    config: BevyZeroverseConfig,
    queued_seed: Option<u64>,
    next_token: u64,
    tensors: HashMap<u64, Tensor<4>>,
    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    cuda: Option<cuda::Bridge>,
}
mod worker;
pub use worker::{GpuDataset, GpuLiveDataset};

impl Renderer {
    fn with_transport(mut config: BevyZeroverseConfig, transport: Transport) -> Result<Self> {
        ensure!(
            config.width.is_finite()
                && config.height.is_finite()
                && config.width >= 1.
                && config.height >= 1.
                && config.num_cameras > 0
                && config.playback_steps > 0,
            "GPU capture needs finite positive image dimensions, cameras and timesteps"
        );
        ensure!(
            config.scene_type == ZeroverseSceneType::ProceduralIndoor,
            "GPU JIT capture currently requires procedural_indoor"
        );
        ensure!(
            config.ovoxel_mode == bevy_zeroverse::app::OvoxelMode::Disabled,
            "GPU JIT capture does not export O-voxels; use the CPU archive API"
        );
        config.indoor_seed.get_or_insert_with(rand::random);
        config.headless = true;
        config.image_copiers = true;
        config.editor = false;
        config.keybinds = false;
        config.press_esc_close = false;
        // The training transport always exposes shared geometric raster outputs,
        // independently of which derived maps a model uses. Never reduce quality.
        for mode in [
            bevy_zeroverse::render::RenderMode::Color,
            bevy_zeroverse::render::RenderMode::Depth,
            bevy_zeroverse::render::RenderMode::Normal,
        ] {
            if !config.render_modes.contains(&mode) {
                config.render_modes.push(mode);
            }
        }
        config.validate_ovoxel().map_err(anyhow::Error::msg)?;
        bevy_zeroverse::headless::setup_globals(std::env::var("BEVY_ASSET_ROOT").ok());
        let initial_app = match transport {
            Transport::Wgpu => App::new(),
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            Transport::Cuda => {
                let mut app = App::new();
                app.insert_resource(cuda::settings());
                app
            }
        };
        let mut app =
            bevy_zeroverse::headless::create_app(Some(initial_app), Some(config.clone()), false);
        app.finish();
        let world = app.sub_app(RenderApp).world();
        let adapter = world.resource::<RenderAdapter>();
        let setup = WgpuSetup {
            instance: (**world.resource::<RenderInstance>()).clone(),
            adapter: (**adapter).clone(),
            device: world.resource::<RenderDevice>().wgpu_device().clone(),
            queue: (**world.resource::<RenderQueue>()).clone(),
            backend: adapter.get_info().backend,
        };
        // Use Bevy's normal executor: uploads, light preparation and independent
        // main-world systems must retain the same parallelism as CPU capture.
        // Cleanup transfers RenderApp to the pipelined rendering thread.
        // Clone its instance/device/queue first; handles refer to the same GPU.
        app.cleanup();
        #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
        let cuda = match transport {
            Transport::Cuda => Some(cuda::Bridge::new(&setup)?),
            Transport::Wgpu => None,
        };
        let device = match transport {
            Transport::Wgpu => Device::wgpu_options().setup(setup).init()?,
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            Transport::Cuda => cuda.as_ref().unwrap().device().clone(),
        };
        app.insert_resource(GpuSampleSink::default());
        app.world_mut()
            .resource_mut::<procedural_indoor::preparation::IndoorPrefetch>()
            .depth = 3;
        Ok(Self {
            app,
            device,
            queued_seed: None,
            config,
            next_token: 0,
            tensors: HashMap::new(),
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            cuda,
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[cfg(test)]
    fn sample_seed(&mut self, seed: u64) -> Result<GpuSample> {
        self.capture(seed, true)
    }
    fn capture(&mut self, seed: u64, reset_sequence: bool) -> Result<GpuSample> {
        #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
        if let Some(bridge) = &self.cuda {
            bridge.begin_capture()?;
        }
        self.tensors.clear();
        *self.app.world_mut().resource_mut::<GpuSampleSink>() = GpuSampleSink::default();
        self.app.world_mut().resource_mut::<CaptureFailure>().0 = None;
        if reset_sequence {
            procedural_indoor::reset_indoor_sequence(self.app.world_mut(), seed);
            self.app.world_mut().write_message(RegenerateSceneEvent);
            self.app.update();
        }
        let mut sampler = SamplerState::from_config(&self.config);
        // Preserve the continuous sequence so complete future-room uploads and
        // lighting preparation can overlap capture, just like the CPU transport.
        sampler.regenerate_scene = true;
        self.app.insert_resource(sampler);
        let start = Instant::now();
        let waiting_before = self
            .app
            .world()
            .resource::<bevy_zeroverse::sample::CaptureProgress>()
            .waiting_updates;
        let mut updates = 0_u64;
        let mut target_binding_seconds = 0.0_f64;
        let mut app_update_seconds = 0.0_f64;
        let expected_views = self
            .config
            .num_cameras
            .checked_mul(self.config.playback_steps as usize)
            .context("GPU capture view count overflow")?;
        while self.app.world().resource::<SamplerState>().enabled {
            ensure!(
                start.elapsed() < Duration::from_secs(120),
                "GPU capture timed out for seed {seed}"
            );
            let binding_started = Instant::now();
            // The sampler has a final metadata update after taking the last
            // planes. Do not allocate another set of targets for that update.
            let copiers: Vec<_> =
                if self.app.world().resource::<GpuSampleSink>().images.len() >= expected_views {
                    Vec::new()
                } else {
                    self.app
                        .world_mut()
                        .query::<&ImageCopier>()
                        .iter(self.app.world())
                        .cloned()
                        .collect()
                };
            let mut copiers = copiers
                .into_iter()
                .filter(|copier| !copier.gpu_target_armed())
                .peekable();
            while let Some(copier) = copiers.next() {
                let shape = copier.gpu_shape();
                self.next_token = self
                    .next_token
                    .checked_add(1)
                    .context("GPU capture token overflow")?;
                #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
                if let Some(bridge) = &mut self.cuda {
                    let target = bridge.allocate(shape, self.next_token)?;
                    copier.bind_gpu_target(target).map_err(anyhow::Error::msg)?;
                    continue;
                }
                let tensor = Tensor::<4>::empty(shape, &self.device);
                // All allocations are resolved on this producer stream. Flush
                // it once before Bevy submits copies, rather than once per view.
                let target = target(tensor.clone(), self.next_token, copiers.peek().is_none())?;
                copier.bind_gpu_target(target).map_err(anyhow::Error::msg)?;
                self.tensors.insert(self.next_token, tensor);
            }
            target_binding_seconds += binding_started.elapsed().as_secs_f64();
            let update_started = Instant::now();
            self.app.update();
            app_update_seconds += update_started.elapsed().as_secs_f64();
            updates += 1;
            if let Some(exit) = self.app.should_exit() {
                anyhow::bail!("GPU renderer exited: {exit:?}");
            }
            if let Some(error) = &self.app.world().resource::<CaptureFailure>().0 {
                anyhow::bail!(error.clone());
            }
        }
        bevy::log::debug!(target: "bevy_zeroverse_burn::gpu", seed, updates, target_binding_seconds, app_update_seconds, waiting_updates = self.app.world().resource::<bevy_zeroverse::sample::CaptureProgress>().waiting_updates - waiting_before, seconds = start.elapsed().as_secs_f64(), "GPU capture submitted");
        let mut sink = std::mem::take(&mut *self.app.world_mut().resource_mut::<GpuSampleSink>());
        #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
        if let Some(bridge) = &mut self.cuda {
            self.tensors = bridge.finish(sink.images.iter().map(|(_, p)| p.target.token))?;
        }
        let mut metadata = sink
            .metadata
            .take()
            .context("GPU capture finished without metadata")?;
        if let Some(object) = metadata
            .indoor_render_metadata
            .as_mut()
            .and_then(|m| m.as_object_mut())
        {
            let transport = serde_json::json!({
                "backend": "wgpu", "host_image_bytes": 0,
                "base_gpu_copies_per_attachment": 1,
                "shared_device_and_queue": true,
                "producer_completion_fence": false,
                "allocated_host_readback_bytes": self.app.world_mut().query::<&ImageCopier>().iter(self.app.world()).map(ImageCopier::allocated_readback_bytes).sum::<u64>(),
                "padding_removal": "GPU only, if required by image width"
            });
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            let transport = if self.cuda.is_some() {
                serde_json::json!({
                    "backend": "vulkan_cuda", "host_image_bytes": 0,
                    "base_gpu_copies_per_attachment": 2,
                    "shared_physical_device_verified_by_uuid": true,
                    "shared_external_device_memory": true,
                    "producer_completion_fence": true,
                    "padding_removal": "GPU only, if required by image width"
                })
            } else {
                transport
            };
            object.insert("gpu_tensor_contract".into(), serde_json::json!({
                "version": 1, "layout": "HWC", "dtype": "float32",
                "color": "tonemapped linear RGBA", "world_depth": "world XYZ and metric camera Z depth",
                "normal_semantic": "view normal encoded in [0,1] and integer semantic class ID",
                "optical_flow": "forward pixel dx/dy, source validity, target visibility",
                "co_visibility": "camera membership bits, peer count, source validity, zero",
                "transport": transport
            }));
        }
        sink.images.sort_by_key(|(index, _)| *index);
        ensure!(
            sink.images.len() == metadata.views.len(),
            "GPU views and metadata do not align"
        );
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let flow = self
            .config
            .render_modes
            .iter()
            .any(bevy_zeroverse::render::RenderMode::is_flow);
        let cov = self
            .config
            .render_modes
            .contains(&bevy_zeroverse::render::RenderMode::CoVisibility);
        let mut views = Vec::with_capacity(sink.images.len());
        for (expected, (index, packet)) in sink.images.into_iter().enumerate() {
            ensure!(index == expected, "GPU view ordering mismatch");
            let tensor = self
                .tensors
                .remove(&packet.target.token)
                .context("unknown GPU allocation token")?;
            let plane = |i| {
                tensor
                    .clone()
                    .slice([i..i + 1, 0..height, 0..width, 0..4])
                    .reshape([height, width, 4])
            };
            views.push(GpuView {
                color: plane(0),
                world_depth: plane(1),
                normal_semantic: plane(2),
                optical_flow: flow.then(|| plane(3)),
                co_visibility: cov.then(|| plane(3 + usize::from(flow))),
            });
        }
        if flow {
            let cameras = metadata.view_dim as usize;
            // The raster writes source N-1 when target N is rendered. Shift by
            // one timestep exactly like CPU export; terminal sources are invalid.
            for source in 0..views.len() {
                views[source].optical_flow = if source + cameras < views.len() {
                    views[source + cameras].optical_flow.take()
                } else {
                    Some(Tensor::zeros([height, width, 4], &self.device))
                };
            }
        }
        self.tensors.clear();
        self.queued_seed = seed.checked_add(1);
        Ok(GpuSample { metadata, views })
    }
}

fn target(tensor: Tensor<4>, token: u64, flush: bool) -> Result<GpuCaptureTarget> {
    let bytes = tensor.dims().iter().product::<usize>() as u64 * 4;
    let primitive = primitive(tensor)?;
    let resource = primitive
        .client
        .get_resource::<burn::cubecl::wgpu::WgpuServer<AutoCompiler>>(primitive.handle.clone())?;
    if flush {
        primitive.client.flush()?;
    }
    let view = resource.resource();
    ensure!(view.size >= bytes, "GPU tensor allocation is undersized");
    Ok(GpuCaptureTarget {
        token,
        buffer: view.buffer.clone(),
        offset: view.offset,
        byte_len: bytes,
        lease: Arc::new(resource),
    })
}

fn primitive(tensor: Tensor<4>) -> Result<CubeTensor> {
    let primitive = tensor
        .try_into_primitive::<Wgpu>()
        // Burn 0.22 uses the same Cube backend for WGPU and CUDA tensors.
        .map_err(|e| anyhow::anyhow!("GPU capture requires a Cube tensor: {e:?}"))?;
    let primitive: Box<dyn Any> = Box::new(primitive);
    let primitive = match primitive.downcast::<CubeTensor>() {
        Ok(tensor) => *tensor,
        Err(primitive) => {
            #[cfg(feature = "gpu_tensor_fusion")]
            {
                let tensor = primitive
                    .downcast::<burn_fusion::FusionTensor<burn_cubecl::fusion::FusionCubeRuntime>>()
                    .map_err(|_| anyhow::anyhow!("unknown Cube GPU primitive"))?;
                tensor
                    .client
                    .clone()
                    .resolve_tensor_float::<burn::backend::wgpu::CubeBackend>(*tensor)
            }
            #[cfg(not(feature = "gpu_tensor_fusion"))]
            {
                drop(primitive);
                anyhow::bail!(
                    "Burn fusion is enabled: enable bevy_zeroverse_burn/gpu_tensor_fusion too"
                );
            }
        }
    };
    ensure!(
        primitive.dtype == burn::tensor::DType::F32
            && primitive.is_contiguous()
            && !primitive.meta.is_tiled(),
        "GPU capture requires plain contiguous float32 tensor allocation"
    );
    Ok(primitive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::{
        headless::update_capture,
        io::channels,
        render::{RenderMode, depth::DepthFormat, glass::AnnotationGlass},
    };

    pub(super) fn setup_test_assets() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        bevy_zeroverse::headless::setup_globals(Some(root.display().to_string()));
    }

    fn cpu_capture(dataset: &mut Renderer, seed: u64) -> bevy_zeroverse::sample::Sample {
        dataset.app.world_mut().remove_resource::<GpuSampleSink>();
        procedural_indoor::reset_indoor_sequence(dataset.app.world_mut(), seed);
        dataset.app.world_mut().write_message(RegenerateSceneEvent);
        dataset.app.update();
        let mut sampler = SamplerState::from_config(&dataset.config);
        sampler.regenerate_scene = false;
        dataset.app.insert_resource(sampler);
        let start = Instant::now();
        while dataset.app.world().resource::<SamplerState>().enabled {
            assert!(start.elapsed() < Duration::from_secs(120));
            update_capture(&mut dataset.app);
            assert!(dataset.app.world().resource::<CaptureFailure>().0.is_none());
        }
        let sample = channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        dataset.app.insert_resource(GpuSampleSink::default());
        sample
    }
    pub(super) fn values(tensor: Tensor<3>) -> Vec<f32> {
        tensor.into_data().try_to_vec().unwrap()
    }
    pub(super) fn equal_plane(actual: Vec<f32>, expected: Vec<f32>, description: &str) {
        assert_eq!(actual.len(), expected.len());
        let different = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert_eq!(
            different, 0,
            "{description}: {different} changed values, max error {max}"
        );
    }
    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_ne_bytes(*v))
            .collect()
    }

    #[test]
    #[ignore = "native GPU: same-scene CPU/GPU parity, row padding, glass, retained allocations and replay"]
    fn gpu_transport_matches_cpu_capture_and_retains_old_samples() -> Result<()> {
        transport_parity(
            Transport::Wgpu,
            bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Mixed,
            1,
        )
    }

    #[test]
    #[ignore = "native GPU: new tiered program, poses, float32 annotations and retained tensors"]
    fn public_program_gpu_transport_matches_cpu_capture() -> Result<()> {
        transport_parity(
            Transport::Wgpu,
            bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::LectureHall,
            1,
        )
    }

    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    #[test]
    #[ignore = "native NVIDIA GPU: Vulkan/CUDA exact annotation parity, padding and retained allocations"]
    fn cuda_transport_matches_cpu_capture_and_retains_old_samples() -> Result<()> {
        transport_parity(
            Transport::Cuda,
            bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Mixed,
            1,
        )
    }

    #[test]
    #[ignore = "native GPU: supersampled RGB and final-grid annotations share the WGPU device"]
    fn ssaa_wgpu_transport_parity() -> Result<()> {
        transport_parity(
            Transport::Wgpu,
            bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Mixed,
            2,
        )
    }

    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    #[test]
    #[ignore = "native NVIDIA GPU: supersampled Vulkan/CUDA transport and retained tensors"]
    fn ssaa_cuda_transport_parity() -> Result<()> {
        transport_parity(
            Transport::Cuda,
            bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout::Mixed,
            2,
        )
    }

    fn transport_parity(
        transport: Transport,
        layout: bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout,
        ssaa_factor: u32,
    ) -> Result<()> {
        setup_test_assets();
        channels::init_channels();
        let mut dataset = Renderer::with_transport(
            BevyZeroverseConfig {
                scene_type: ZeroverseSceneType::ProceduralIndoor,
                indoor_layout: layout,
                ssaa_factor,
                indoor_seed: Some(6),
                num_cameras: 3,
                width: 321.,
                height: 239.,
                playback_steps: 2,
                indoor_human_density: 0.25,
                indoor_camera: Some(r#"{"principal_point_jitter":[0.12,0.12],"multiview":{"graph":"connected","target_variation":0.7}}"#.into()),
                depth_format: DepthFormat::Linear,
                z_depth: true,
                render_modes: vec![
                    RenderMode::Color,
                    RenderMode::Depth,
                    RenderMode::Normal,
                    RenderMode::Position,
                    RenderMode::Semantic,
                    RenderMode::OpticalFlow,
                    RenderMode::CoVisibility,
                ],
                ..default()
            },
            transport,
        )?;
        let mut retained = None;
        for policy in [AnnotationGlass::Surface, AnnotationGlass::Through] {
            let _ = channels::take_capture_telemetry();
            dataset.config.annotation_glass = policy;
            dataset
                .app
                .world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .annotation_glass = policy;
            let gpu = dataset.sample_seed(6)?;
            assert!(
                channels::take_capture_telemetry().is_none(),
                "tensor renderers must not publish channel-consumer telemetry"
            );
            if retained.is_none() {
                assert!(
                    dataset
                        .app
                        .world_mut()
                        .query::<&ImageCopier>()
                        .iter(dataset.app.world())
                        .all(|copier| copier.allocated_readback_bytes() == 0),
                    "GPU capture allocated unused CPU readback storage"
                );
            }
            let cpu = cpu_capture(&mut dataset, 6);
            let telemetry = channels::take_capture_telemetry().expect("CPU capture diagnostics");
            assert_eq!(telemetry.seed, Some(6));
            assert!(telemetry.preparation.is_some());
            assert!(
                dataset
                    .app
                    .world_mut()
                    .query::<&ImageCopier>()
                    .iter(dataset.app.world())
                    .all(|copier| copier.allocated_readback_bytes() > 0),
                "CPU fallback did not allocate its readback storage"
            );
            assert_eq!(gpu.metadata.aabb, cpu.aabb);
            assert_eq!(gpu.metadata.sampling, cpu.sampling);
            assert_eq!(
                cpu.sampling.map_or(1, |policy| policy.ssaa_factor),
                ssaa_factor
            );
            assert!(
                !cpu.human_poses.is_empty(),
                "pose parity requires a populated scene"
            );
            assert_eq!(gpu.metadata.human_instance_ids, cpu.human_instance_ids);
            assert_eq!(gpu.metadata.human_bone_names, cpu.human_bone_names);
            assert_eq!(gpu.metadata.human_bone_parents, cpu.human_bone_parents);
            assert_eq!(gpu.metadata.human_pose_steps, cpu.human_pose_steps);
            assert_eq!(gpu.metadata.human_poses, cpu.human_poses);
            assert_eq!(gpu.metadata.annotation_glass, policy);
            assert_eq!(gpu.metadata.views.len(), cpu.views.len());
            for (index, (actual, expected)) in gpu.views.iter().zip(&cpu.views).enumerate() {
                assert_eq!(
                    gpu.metadata
                        .human_poses_for_view(index)
                        .map_err(anyhow::Error::msg)?,
                    cpu.human_poses_for_view(index)
                        .map_err(anyhow::Error::msg)?
                );
                assert_eq!(gpu.metadata.views[index].calibration, expected.calibration);
                let calibration = expected
                    .calibration
                    .as_ref()
                    .expect("full camera calibration");
                assert_ne!(
                    calibration.k[0][2],
                    321. * 0.5,
                    "shifted projection was not exercised"
                );
                assert_eq!(actual.color.dims(), [239, 321, 4]);
                assert_eq!(
                    gpu.metadata.views[index].world_from_view,
                    expected.world_from_view
                );
                equal_plane(
                    values(actual.color.clone()),
                    floats(&expected.color),
                    &format!("RGB {index} {policy:?}"),
                );
                equal_plane(
                    values(actual.co_visibility.clone().unwrap()),
                    floats(&expected.co_visibility),
                    "co-visibility",
                );
                equal_plane(
                    values(actual.optical_flow.clone().unwrap()),
                    floats(&expected.optical_flow),
                    "forward optical flow",
                );
                let wd = values(actual.world_depth.clone());
                let ns = values(actual.normal_semantic.clone());
                let depths = floats(&expected.depth);
                let normals = floats(&expected.normal);
                let positions = floats(&expected.position);
                let semantic = floats(&expected.semantic);
                for pixel in 0..321 * 239 {
                    let k = pixel * 4;
                    let hit = wd[k + 3] > 0.;
                    assert_eq!(wd[k + 3], depths[k]);
                    assert_eq!(&ns[k..k + 3], &normals[k..k + 3]);
                    if hit {
                        for channel in 0..3 {
                            let min = cpu.aabb[0][channel];
                            let range = (cpu.aabb[1][channel] - min).max(1e-5);
                            assert_eq!((wd[k + channel] - min) / range, positions[k + channel]);
                        }
                        let id = ns[k + 3] as u32;
                        let color = bevy_zeroverse::render::ground_truth::semantic_label(id)
                            .unwrap()
                            .color()
                            .to_linear()
                            .to_f32_array();
                        assert_eq!(&color[..3], &semantic[k..k + 3]);
                    }
                }
            }
            retained = Some(gpu);
        }
        let retained = retained.unwrap();
        let original = values(retained.views[0].color.clone());
        assert!(
            retained
                .metadata
                .views
                .iter()
                .all(|v| v.color.is_empty() && v.depth.is_empty() && v.normal.is_empty())
        );
        equal_plane(
            values(retained.views[0].color.clone() * 0.5 + 0.25),
            original.iter().map(|v| v * 0.5 + 0.25).collect(),
            "consumer arithmetic on training device",
        );
        for seed in 7..11 {
            let _ = dataset.sample_seed(seed)?;
        }
        assert_eq!(
            original,
            values(retained.views[0].color.clone()),
            "new rooms reused a retained tensor allocation"
        );
        #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
        if let Some(bridge) = &mut dataset.cuda {
            assert!(bridge.finish(std::iter::empty()).is_err());
            assert!(
                bridge.begin_capture().is_err(),
                "failed batches must fail closed"
            );
        }
        drop(dataset);
        equal_plane(
            values(retained.views[0].color.clone()),
            original,
            "tensor survives renderer destruction",
        );
        Ok(())
    }

    #[test]
    #[ignore = "native GPU: aligned 512x512 planes share the capture allocation without GPU slicing copies"]
    fn gpu_aligned_planes_share_capture_allocation() -> Result<()> {
        let device = Device::wgpu(Default::default());
        let tensor = Tensor::<4>::empty([5, 512, 512, 4], &device);
        let allocation = target(tensor.clone(), 0, true)?;
        for plane in 0..5 {
            let view = tensor
                .clone()
                .slice([plane..plane + 1, 0..512, 0..512, 0..4]);
            let view = target(view, 0, true)?;
            assert_eq!(view.buffer, allocation.buffer);
            assert_eq!(
                view.offset,
                allocation.offset + plane as u64 * 512 * 512 * 16
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "native GPU: bounded worker sequencing, indexed replay and tensor lifetime"]
    fn gpu_worker_preserves_stream_seeds_across_indexed_capture() -> Result<()> {
        worker_lifetime(Transport::Wgpu)
    }

    #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
    #[test]
    #[ignore = "native NVIDIA GPU: CUDA worker sequencing, consumer thread and shutdown lifetime"]
    fn cuda_worker_preserves_stream_seeds_across_indexed_capture() -> Result<()> {
        worker_lifetime(Transport::Cuda)
    }

    fn worker_lifetime(transport: Transport) -> Result<()> {
        setup_test_assets();
        let config = BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            num_cameras: 3,
            width: 128.,
            height: 128.,
            playback_steps: 1,
            indoor_human_density: 0.25,
            ..default()
        };
        let mut dataset = match transport {
            Transport::Wgpu => GpuLiveDataset::new(config)?,
            #[cfg(all(feature = "cuda_interop", target_os = "linux"))]
            Transport::Cuda => GpuLiveDataset::new_cuda(config)?,
        };
        let retained = dataset.next_sample()?;
        assert_eq!(retained.metadata.indoor.as_ref().unwrap().seed, 6);
        let before = values(retained.views[0].color.clone());
        let sample = dataset.next_sample()?;
        assert_eq!(sample.metadata.indoor.as_ref().unwrap().seed, 7);
        let indexed = dataset.sample_seed(19)?;
        assert_eq!(indexed.metadata.indoor.as_ref().unwrap().seed, 19);
        for seed in 8..11 {
            let sample = dataset.next_sample()?;
            assert_eq!(sample.metadata.indoor.as_ref().unwrap().seed, seed);
            assert!(
                values(sample.views[0].color.clone())
                    .iter()
                    .any(|v| *v > 0.1)
            );
        }
        equal_plane(
            values(retained.views[0].color.clone()),
            before,
            "retained worker tensor",
        );
        let dataset = dataset.into_dataset(2);
        use burn::data::dataset::Dataset;
        assert_eq!(dataset.len(), 2);
        let sample = dataset.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        assert_eq!(sample.metadata.indoor.as_ref().unwrap().seed, 11);
        let sample = dataset.get(0).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        assert_eq!(sample.metadata.indoor.as_ref().unwrap().seed, 12);
        drop(dataset);
        assert!(
            values(retained.views[0].color.clone())
                .iter()
                .any(|v| *v > 0.1)
        );
        Ok(())
    }
}

#[cfg(test)]
mod material_qualification;
