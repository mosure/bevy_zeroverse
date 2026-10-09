//! Same-device JIT training: render once into Burn-owned storage. No image
//! readback, host decoding, upload, or GPU completion wait occurs in capture.
//! Use this renderer's `device()` for the model and optimizer too.
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
}
mod worker;
pub use worker::{GpuDataset, GpuLiveDataset};

impl Renderer {
    pub fn new(mut config: BevyZeroverseConfig) -> Result<Self> {
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
        let mut app = bevy_zeroverse::headless::create_app(None, Some(config.clone()), false);
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
        // The small native render schedule spends more time waking workers
        // than preparing draws. Construction still uses the bounded CPU pools;
        // render graph execution and GPU work are unchanged.
        app.sub_app_mut(RenderApp)
            .get_schedule_mut(bevy::render::Render)
            .context("missing render schedule")?
            .set_executor(bevy::ecs::schedule::SingleThreadedExecutor::new());
        // The capture worker has no simulation/input workload. Keep its small
        // ECS schedules local; geometry, maps and transform propagation retain
        // their explicit parallel work, without waking the schedule pool for
        // each readiness update.
        let schedules = app
            .world()
            .resource::<bevy::app::MainScheduleOrder>()
            .labels
            .clone();
        for label in schedules {
            if let Some(schedule) = app.get_schedule_mut(label) {
                schedule.set_executor(bevy::ecs::schedule::SingleThreadedExecutor::new());
            }
        }
        // Cleanup transfers RenderApp to the pipelined rendering thread.
        // Clone its instance/device/queue first; handles refer to the same GPU.
        app.cleanup();
        let device = Device::wgpu_options().setup(setup).init()?;
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
        while self.app.world().resource::<SamplerState>().enabled {
            ensure!(
                start.elapsed() < Duration::from_secs(120),
                "GPU capture timed out for seed {seed}"
            );
            let copiers: Vec<_> = self
                .app
                .world_mut()
                .query::<&ImageCopier>()
                .iter(self.app.world())
                .cloned()
                .collect();
            for copier in copiers {
                if copier.gpu_target_armed() {
                    continue;
                }
                let shape = copier.gpu_shape();
                let tensor = Tensor::<4>::empty(shape, &self.device);
                self.next_token = self
                    .next_token
                    .checked_add(1)
                    .context("GPU capture token overflow")?;
                let target = target(tensor.clone(), self.next_token)?;
                copier.bind_gpu_target(target).map_err(anyhow::Error::msg)?;
                self.tensors.insert(self.next_token, tensor);
            }
            self.app.update();
            updates += 1;
            if let Some(exit) = self.app.should_exit() {
                anyhow::bail!("GPU renderer exited: {exit:?}");
            }
            if let Some(error) = &self.app.world().resource::<CaptureFailure>().0 {
                anyhow::bail!(error.clone());
            }
        }
        bevy::log::debug!(target: "bevy_zeroverse_burn::gpu", seed, updates, waiting_updates = self.app.world().resource::<bevy_zeroverse::sample::CaptureProgress>().waiting_updates - waiting_before, seconds = start.elapsed().as_secs_f64(), "GPU capture submitted");
        let mut sink = std::mem::take(&mut *self.app.world_mut().resource_mut::<GpuSampleSink>());
        let mut metadata = sink
            .metadata
            .take()
            .context("GPU capture finished without metadata")?;
        if let Some(object) = metadata
            .indoor_render_metadata
            .as_mut()
            .and_then(|m| m.as_object_mut())
        {
            object.insert("gpu_tensor_contract".into(), serde_json::json!({
                "version": 1, "layout": "HWC", "dtype": "float32",
                "color": "tonemapped linear RGBA", "world_depth": "world XYZ and metric camera Z depth",
                "normal_semantic": "view normal encoded in [0,1] and integer semantic class ID",
                "optical_flow": "forward pixel dx/dy, source validity, target visibility",
                "co_visibility": "camera membership bits, peer count, source validity, zero"
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

fn target(tensor: Tensor<4>, token: u64) -> Result<GpuCaptureTarget> {
    let bytes = tensor.dims().iter().product::<usize>() as u64 * 4;
    let primitive = tensor
        .try_into_primitive::<Wgpu>()
        .map_err(|e| anyhow::anyhow!("GPU capture requires a WGPU tensor: {e:?}"))?;
    let primitive: Box<dyn Any> = Box::new(primitive);
    let primitive = match primitive.downcast::<CubeTensor>() {
        Ok(tensor) => *tensor,
        Err(primitive) => {
            #[cfg(feature = "gpu_tensor_fusion")]
            {
                let tensor = primitive
                    .downcast::<burn_fusion::FusionTensor<burn_cubecl::fusion::FusionCubeRuntime>>()
                    .map_err(|_| anyhow::anyhow!("unknown WGPU primitive"))?;
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
    let resource = primitive
        .client
        .get_resource::<burn::cubecl::wgpu::WgpuServer<AutoCompiler>>(primitive.handle.clone())?;
    primitive.client.flush()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::{
        headless::update_capture,
        io::channels,
        render::{RenderMode, depth::DepthFormat, glass::AnnotationGlass},
    };

    fn setup_test_assets() {
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
    fn values(tensor: Tensor<3>) -> Vec<f32> {
        tensor.into_data().try_to_vec().unwrap()
    }
    fn equal_plane(actual: Vec<f32>, expected: Vec<f32>, description: &str) {
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
        setup_test_assets();
        channels::init_channels();
        let mut dataset = Renderer::new(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            num_cameras: 3,
            width: 321.,
            height: 239.,
            playback_steps: 2,
            indoor_human_density: 0.25,
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
        })?;
        let mut retained = None;
        for policy in [AnnotationGlass::Surface, AnnotationGlass::Through] {
            dataset.config.annotation_glass = policy;
            dataset
                .app
                .world_mut()
                .resource_mut::<BevyZeroverseConfig>()
                .annotation_glass = policy;
            let cpu = cpu_capture(&mut dataset, 6);
            let gpu = dataset.sample_seed(6)?;
            assert_eq!(gpu.metadata.aabb, cpu.aabb);
            assert_eq!(gpu.metadata.annotation_glass, policy);
            assert_eq!(gpu.metadata.views.len(), cpu.views.len());
            for (index, (actual, expected)) in gpu.views.iter().zip(&cpu.views).enumerate() {
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
        for seed in 7..11 {
            let _ = dataset.sample_seed(seed)?;
        }
        assert_eq!(
            original,
            values(retained.views[0].color.clone()),
            "new rooms reused a retained tensor allocation"
        );
        Ok(())
    }

    #[test]
    #[ignore = "native GPU: bounded worker sequencing, indexed replay and tensor lifetime"]
    fn gpu_worker_preserves_stream_seeds_across_indexed_capture() -> Result<()> {
        setup_test_assets();
        let mut dataset = GpuLiveDataset::new(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            num_cameras: 3,
            width: 128.,
            height: 128.,
            playback_steps: 1,
            indoor_human_density: 0.25,
            ..default()
        })?;
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
