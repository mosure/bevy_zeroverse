//! Resolved scene/output policy, independent of transport and queue scheduling.
use super::*;
use bevy_zeroverse::render::{RenderMode, depth::DepthFormat};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use bevy_zeroverse_capture::gpu::{ColorOutput, Plane};

/// Raw planes needed internally and copied to the consumer are distinct.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentPlan {
    pub rendered: Vec<Plane>,
    pub copied: Vec<Plane>,
}
impl AttachmentPlan {
    fn resolve(modes: &[RenderMode]) -> Self {
        let geometry = modes.iter().any(|mode| *mode != RenderMode::Color);
        let flow = modes.iter().any(RenderMode::is_flow);
        let cov = modes.contains(&RenderMode::CoVisibility);
        let mut rendered = vec![Plane::Color];
        if geometry {
            rendered.extend([Plane::WorldDepth, Plane::NormalSemantic]);
        }
        if flow {
            rendered.push(Plane::OpticalFlow);
        }
        if cov {
            rendered.push(Plane::CoVisibility);
        }
        let copied = rendered
            .iter()
            .copied()
            .filter(|plane| match plane {
                Plane::Color => modes.contains(&RenderMode::Color),
                Plane::WorldDepth => {
                    modes.contains(&RenderMode::Depth) || modes.contains(&RenderMode::Position)
                }
                Plane::NormalSemantic => {
                    modes.contains(&RenderMode::Normal) || modes.contains(&RenderMode::Semantic)
                }
                Plane::OpticalFlow => flow,
                Plane::CoVisibility => cov,
            })
            .collect();
        Self { rendered, copied }
    }
    pub fn source_indices(&self) -> Vec<usize> {
        self.copied
            .iter()
            .map(|p| self.rendered.iter().position(|r| r == p).unwrap())
            .collect()
    }
    pub fn index(&self, plane: Plane) -> Option<usize> {
        self.copied.iter().position(|p| *p == plane)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaptureSpec {
    pub scene: BevyZeroverseConfig,
    /// Explicit output transfer function/channel layout. Rendering is always linear.
    pub color_output: ColorOutput,
}
impl CaptureSpec {
    pub fn from_scene(scene: BevyZeroverseConfig) -> Self {
        Self {
            scene,
            color_output: ColorOutput::TonemappedLinearRgbaF32,
        }
    }
    pub fn color_output(mut self, output: ColorOutput) -> Self {
        self.color_output = output;
        self
    }
    pub fn from_gen(config: &crate::generator::GenConfig) -> Result<Self> {
        crate::generator::validate_gen_config(config)?;
        ensure!(
            !config.export_schematic && !config.export_ovoxel,
            "GPU capture does not export archives, schematics or O-voxels"
        );
        ensure!(
            config.rgb_sensor.is_none(),
            "GPU sensor transforms are not supported; they must not be silently omitted"
        );
        Ok(Self::from_scene(config.scene_config()).color_output(ColorOutput::SrgbRgbF32))
    }
    pub fn resolve(mut self) -> Result<ResolvedCaptureSpec> {
        let config = &mut self.scene;
        ensure!(
            config.scene_type == ZeroverseSceneType::ProceduralIndoor,
            "GPU capture requires procedural_indoor"
        );
        ensure!(
            config.ovoxel_mode == bevy_zeroverse::app::OvoxelMode::Disabled,
            "GPU capture does not export O-voxels"
        );
        ensure!(
            [config.width, config.height]
                .iter()
                .all(|v| v.is_finite() && *v >= 1. && v.fract() == 0. && *v <= 16384.),
            "capture dimensions must be integral in 1..=16384"
        );
        ensure!(
            config.num_cameras > 0 && config.playback_steps > 0,
            "capture requires cameras and timesteps"
        );
        ensure!(config.ssaa_factor <= 4, "SSAA factor must be 0..=4");
        ensure!(
            config.playback_step.is_finite()
                && config.playback_step >= 0.
                && config.playback_step * config.playback_steps.saturating_sub(1) as f32 <= 1.,
            "capture trajectory progress must remain in [0,1]"
        );
        ensure!(
            (0.0..=1.0).contains(&config.indoor_density)
                && (0.0..=1.0).contains(&config.indoor_human_density),
            "indoor densities must be finite in [0,1]"
        );
        ensure!(
            config.human_motion.is_none() || cfg!(feature = "human_motion"),
            "human motion requires the human_motion Cargo feature"
        );
        // Reuse field validators (camera, motion, appearance, GI), including when
        // Rust callers assembled the config directly rather than deserializing.
        *config = serde_json::from_slice(&serde_json::to_vec(&config)?)?;
        if config.render_modes.is_empty() {
            config.render_modes.push(config.render_mode.clone());
        }
        config.render_modes.sort_by_key(|m| m.clone() as u8);
        config.render_modes.dedup();
        if config.render_modes.contains(&RenderMode::Depth) {
            ensure!(
                config.depth_format == DepthFormat::Linear && config.z_depth,
                "GPU depth is metric camera-Z; request depth_format=Linear and z_depth=true"
            );
        }
        ensure!(
            !config.render_modes.contains(&RenderMode::CoVisibility) || config.num_cameras <= 16,
            "co-visibility supports at most 16 cameras"
        );
        config.headless = true;
        config.image_copiers = true;
        config.editor = false;
        config.keybinds = false;
        config.press_esc_close = false;
        config.render_mode = RenderMode::Color; // Native MRT provides requested geometry.
        config.regenerate_ms = 0;
        config.validate_ovoxel().map_err(anyhow::Error::msg)?;
        let plan = AttachmentPlan::resolve(&config.render_modes);
        let views = config
            .num_cameras
            .checked_mul(config.playback_steps as usize)
            .context("view count overflow")?;
        ensure!(views <= 4096, "capture exceeds 4096 views per room");
        let row_pixels = (config.width as usize).div_ceil(16) * 16;
        let view_bytes = (plan.copied.len() as u64)
            .checked_mul(config.height as u64)
            .and_then(|n| n.checked_mul(row_pixels as u64 * 16))
            .context("capture byte count overflow")?;
        let room_bytes = view_bytes
            .checked_mul(views as u64)
            .context("capture byte count overflow")?;
        // Seed belongs to a request, not the immutable session policy.
        let mut identity = self.clone();
        identity.scene.indoor_seed = None;
        let fingerprint = bevy_zeroverse_capture::provenance::sha(&serde_json::to_vec(&identity)?);
        Ok(ResolvedCaptureSpec {
            spec: self,
            plan,
            views,
            #[cfg(all(feature = "cuda_ipc", target_os = "linux"))]
            row_pixels,
            #[cfg(all(feature = "cuda_ipc", target_os = "linux"))]
            view_bytes,
            room_bytes,
            fingerprint,
        })
    }
}

/// Construct only through resolve(), so budgets and copied-plane offsets agree.
#[derive(Clone, Debug)]
pub struct ResolvedCaptureSpec {
    pub(crate) spec: CaptureSpec,
    pub(crate) plan: AttachmentPlan,
    pub(crate) views: usize,
    #[cfg(all(feature = "cuda_ipc", target_os = "linux"))]
    pub(crate) row_pixels: usize,
    #[cfg(all(feature = "cuda_ipc", target_os = "linux"))]
    pub(crate) view_bytes: u64,
    pub(crate) room_bytes: u64,
    pub(crate) fingerprint: String,
}
impl ResolvedCaptureSpec {
    pub fn scene(&self) -> &BevyZeroverseConfig {
        &self.spec.scene
    }
    pub fn attachments(&self) -> &AttachmentPlan {
        &self.plan
    }
    pub fn color_output(&self) -> ColorOutput {
        self.spec.color_output
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn captured_room_bytes(&self) -> u64 {
        self.room_bytes
    }
    pub fn view_count(&self) -> usize {
        self.views
    }
}

/// Bounds transport work; never changes scene quality, shadows or annotations.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureRuntimeOptions {
    pub asset_root: Option<PathBuf>,
    pub initialization_timeout: Duration,
    pub capture_timeout: Duration,
    /// Global outstanding/unreceived room bound for an in-process session,
    /// not a per-camera queue. CUDA IPC instead admits at most one room per
    /// connected lane and sizes its completion queue from the lane count.
    pub queue_capacity: usize,
    pub preparation_depth: usize,
    /// Raw capture storage budget per renderer, excluding model and owned outputs.
    pub max_capture_bytes: u64,
    /// Aggregate registered arenas across a CUDA receiver's supervised lanes.
    /// Delivered tensor inventory remains owned/budgeted by the training caller.
    pub max_imported_bytes: u64,
    pub preparation_threads: Option<usize>,
    /// Optional pin for supervised workers built separately from the consumer.
    pub expected_generator_source: Option<String>,
}
impl Default for CaptureRuntimeOptions {
    fn default() -> Self {
        Self {
            asset_root: None,
            initialization_timeout: Duration::from_secs(120),
            capture_timeout: Duration::from_secs(120),
            queue_capacity: 2,
            preparation_depth: 3,
            max_capture_bytes: 512 * 1024 * 1024,
            max_imported_bytes: 1024 * 1024 * 1024,
            preparation_threads: None,
            expected_generator_source: None,
        }
    }
}
impl CaptureRuntimeOptions {
    pub fn from_gen(config: &crate::generator::GenConfig) -> Self {
        Self {
            asset_root: config.asset_root.clone(),
            capture_timeout: Duration::from_secs(config.timeout_secs),
            preparation_depth: if config.indoor_prefetch {
                config.indoor_prefetch_depth
            } else {
                0
            },
            ..Self::default()
        }
    }
    pub(crate) fn validate(&self, spec: &ResolvedCaptureSpec) -> Result<()> {
        ensure!(
            (1..=16).contains(&self.queue_capacity),
            "queue capacity must be 1..=16"
        );
        ensure!(
            self.preparation_depth <= 4,
            "preparation depth must be 0..=4"
        );
        ensure!(
            !self.initialization_timeout.is_zero() && !self.capture_timeout.is_zero(),
            "deadlines must be positive"
        );
        ensure!(
            self.preparation_threads
                .is_none_or(|n| (1..=64).contains(&n)),
            "preparation threads must be 1..=64"
        );
        // Include warmup/retiring camera targets and the next set of bindings.
        ensure!(
            spec.room_bytes
                .checked_mul(2)
                .is_some_and(|n| n <= self.max_capture_bytes),
            "capture policy exceeds raw storage budget; raise the explicit budget or reduce requested outputs"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> BevyZeroverseConfig {
        BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            width: 256.,
            height: 256.,
            num_cameras: 8,
            playback_steps: 5,
            ..Default::default()
        }
    }
    #[test]
    fn rgb_only_has_no_annotation_dependency_and_production_sizes_are_checked() {
        let spec = CaptureSpec::from_scene(scene()).resolve().unwrap();
        assert_eq!(spec.plan.rendered, [Plane::Color]);
        assert_eq!(spec.room_bytes, 40 * 1024 * 1024);
        assert_eq!(spec.views, 40);
        CaptureRuntimeOptions::default().validate(&spec).unwrap();
        let mut config = scene();
        config.width = 512.;
        config.height = 512.;
        config.playback_steps = 3;
        assert_eq!(
            CaptureSpec::from_scene(config)
                .resolve()
                .unwrap()
                .room_bytes,
            96 * 1024 * 1024
        );
    }
    #[test]
    fn requested_modes_override_stale_viewer_mode_and_dependencies_are_not_copied() {
        let mut config = scene();
        config.render_modes = vec![RenderMode::OpticalFlow];
        config.render_mode = RenderMode::Semantic;
        let spec = CaptureSpec::from_scene(config).resolve().unwrap();
        assert_eq!(
            spec.plan.rendered,
            [
                Plane::Color,
                Plane::WorldDepth,
                Plane::NormalSemantic,
                Plane::OpticalFlow
            ]
        );
        assert_eq!(spec.plan.copied, [Plane::OpticalFlow]);
        assert_eq!(spec.plan.source_indices(), [3]);
    }
    #[test]
    fn every_annotation_subset_has_a_consistent_copy_plan() {
        let modes = [
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Position,
            RenderMode::Semantic,
            RenderMode::OpticalFlow,
            RenderMode::MotionVectors,
            RenderMode::CoVisibility,
        ];
        for mask in 1..(1_u16 << modes.len()) {
            let mut config = scene();
            config.depth_format = DepthFormat::Linear;
            config.render_modes = modes
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, mode)| mode.clone())
                .collect();
            let spec = CaptureSpec::from_scene(config.clone()).resolve().unwrap();
            assert!(!spec.plan.copied.is_empty());
            assert_eq!(
                spec.plan.index(Plane::Color).is_some(),
                config.render_modes.contains(&RenderMode::Color)
            );
            let indices = spec.plan.source_indices();
            assert!(indices.windows(2).all(|w| w[0] < w[1]));
            for (i, plane) in indices.iter().zip(&spec.plan.copied) {
                assert_eq!(spec.plan.rendered[*i], *plane);
            }
        }
    }
    #[test]
    fn invalid_policy_and_budget_fail_before_gpu_initialization() {
        let mut config = scene();
        config.width = 256.5;
        assert!(CaptureSpec::from_scene(config).resolve().is_err());
        let mut config = scene();
        config.render_modes = vec![RenderMode::Depth];
        assert!(CaptureSpec::from_scene(config).resolve().is_err());
        let mut config = scene();
        config.indoor_density = f32::NAN;
        assert!(CaptureSpec::from_scene(config).resolve().is_err());
        let spec = CaptureSpec::from_scene(scene()).resolve().unwrap();
        assert!(
            CaptureRuntimeOptions {
                max_capture_bytes: spec.room_bytes,
                ..Default::default()
            }
            .validate(&spec)
            .is_err()
        );
    }
    #[test]
    fn policy_identity_includes_color_but_not_request_seed() {
        let mut config = scene();
        config.indoor_seed = Some(42);
        let a = CaptureSpec::from_scene(config.clone()).resolve().unwrap();
        config.indoor_seed = Some(47);
        let b = CaptureSpec::from_scene(config.clone()).resolve().unwrap();
        assert_eq!(a.fingerprint, b.fingerprint);
        assert_ne!(
            a.fingerprint,
            CaptureSpec::from_scene(config)
                .color_output(ColorOutput::SrgbRgbF32)
                .resolve()
                .unwrap()
                .fingerprint
        );
    }
}
