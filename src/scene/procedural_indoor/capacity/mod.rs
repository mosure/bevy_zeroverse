//! Opt-in capacity planning and accepted-program observations. Normal generation
//! calls none of this code. No model, GPU, download or background audit is started.
mod descriptor;
mod rendered;
use super::{appearance::AppearanceSettings, cameras::CameraSettings};
use crate::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
pub use bevy_zeroverse_capture::capacity::*;
use bevy_zeroverse_capture::{provenance::sha, GeneratorIdentity};
pub use descriptor::observe;
pub use rendered::observe_rendered;
use serde_json::json;

// Ancestry is independent of quantization, observations and appearance/camera
// interventions. This identifier groups evidence; it is never a diversity bin.
pub(super) fn family_identity(
    scene: &super::layout::IndoorManifest,
    plan: &CapacityPlan,
) -> String {
    sha(format!(
        "{}/{}/{}",
        scene.generator_version, plan.population.generator.source_sha256, scene.seed
    )
    .as_bytes())
}

impl BevyZeroverseConfig {
    /// Resolve the *actual* capture schedule explicitly. Viewer playback defaults
    /// and the CLI's single captured timestep must never be confused.
    pub fn capacity_plan(
        &self,
        request: &CapacityRequest,
        capture: CapturePlan,
    ) -> Result<CapacityPlan, String> {
        if self.scene_type != ZeroverseSceneType::ProceduralIndoor {
            return Err("capacity descriptors currently require procedural_indoor".into());
        }
        if !self.width.is_finite()
            || !self.height.is_finite()
            || self.width != capture.image_size[0] as f32
            || self.height != capture.image_size[1] as f32
            || self.num_cameras != capture.cameras as usize
        {
            return Err(
                "capacity capture plan must match resolved image dimensions and camera count"
                    .into(),
            );
        }
        if !(0.0..=1.).contains(&self.indoor_density)
            || !(0.0..=1.).contains(&self.indoor_human_density)
        {
            return Err("invalid indoor population density".into());
        }
        let camera = self
            .indoor_camera
            .as_deref()
            .map(CameraSettings::parse)
            .transpose()?
            .unwrap_or_default();
        let appearance = self
            .indoor_appearance
            .as_deref()
            .map(AppearanceSettings::parse)
            .transpose()?
            .unwrap_or_default();
        let human_motion = self
            .human_motion
            .as_deref()
            .map(crate::human_motion::HumanMotionConfig::parse)
            .transpose()?;
        let object_motion = self
            .indoor_motion
            .as_deref()
            .map(super::motion::IndoorMotionConfig::parse)
            .transpose()?;
        let annotation_glass =
            serde_json::to_value(self.annotation_glass).map_err(|e| e.to_string())?;
        if annotation_glass.as_str().map(|s| s.to_lowercase())
            != Some(capture.annotation_glass.clone())
        {
            return Err("capacity annotation glass policy mismatch".into());
        }
        // This is the complete preparation/capture contract, not a hash of
        // arbitrary user JSON, its key order or UI-only controls.
        let mut runtime = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let runtime_fields = runtime
            .as_object_mut()
            .ok_or("configuration must be an object")?;
        runtime_fields.remove("indoor_seed");
        let defaults =
            serde_json::to_value(BevyZeroverseConfig::default()).map_err(|e| e.to_string())?;
        for key in [
            "editor",
            "viewer_state",
            "gizmos",
            "gizmos_alpha",
            "draw_obb_gizmo",
            "draw_pose_gizmos",
            "name",
            "keybinds",
            "orbit_smoothness",
            "pan_smoothness",
            "zoom_smoothness",
            "camera_grid",
            "room_schematic",
        ] {
            if let Some(value) = defaults.get(key) {
                runtime_fields.insert(key.into(), value.clone());
            }
        }
        runtime_fields.insert("indoor_camera".into(), json!(camera));
        runtime_fields.insert("indoor_appearance".into(), json!(appearance));
        runtime_fields.insert("human_motion".into(), json!(human_motion));
        runtime_fields.insert("indoor_motion".into(), json!(object_motion));
        let resolved = json!({
            "runtime": runtime,
            "layout":self.indoor_layout,"density":self.indoor_density,"human_density":self.indoor_human_density,
            "camera":camera,"appearance":appearance,"human_motion":human_motion,"object_motion":object_motion,
            "quality":self.indoor_quality,"gpu_materials":self.indoor_gpu_materials,"gi_rays":self.indoor_gi_rays,
            "rotation_augmentation":self.rotation_augmentation,"yaw_speed":self.yaw_speed,
            "headless":self.headless,"capture":capture,"seed_policy":request.seed_policy,
            "preparation_settings":"production-default-gi-v1","failure_policy":"retain-error-no-resampling-v1"
        });
        let resolved_config_json = serde_json::to_string(&resolved).map_err(|e| e.to_string())?;
        let temporal = capture.progress.len() > 1;
        let observed_cameras = request.unit != SampleUnit::DressedScene;
        use CapturedFactorState::*;
        let descriptions = [
            (
                "architecture",
                vec![],
                Observed,
                "uniform supported root profiles; conditional spatial programs",
                "accepted primary geometry",
            ),
            (
                "furniture",
                vec!["architecture"],
                Observed,
                "bounded transactional placement and correlated construction",
                "accepted objects and support/interaction relations",
            ),
            (
                "people",
                vec!["architecture", "furniture"],
                if self.indoor_human_density > 0. {
                    Observed
                } else {
                    Inactive
                },
                "accepted support-conditioned morphology, appearance and static poses",
                "accepted people; pixel visibility requires render evidence",
            ),
            (
                "materials",
                vec!["architecture", "furniture"],
                Observed,
                "assembly-correlated assigned PBR roles and continuous recipes",
                "assigned roles only; no unused recipe credit",
            ),
            (
                "lighting",
                vec!["architecture"],
                Observed,
                "correlated sun/sky/electric photometry and exposure",
                "physical parameters; transport accuracy is a separate gate",
            ),
            (
                "camera_initial",
                vec!["architecture", "furniture", "people"],
                if observed_cameras {
                    Observed
                } else {
                    AdmissionOnly
                },
                "accepted baseline, overlap, intrinsics and placement",
                "actual scheduled K/pose, not rejected proposals",
            ),
            (
                "camera_motion",
                vec!["camera_initial"],
                if camera.motion_fraction == 0. {
                    Inactive
                } else if temporal && observed_cameras {
                    Observed
                } else {
                    AdmissionOnly
                },
                "seeded motion fraction and metric paths",
                "only transforms at captured times enter the descriptor",
            ),
            (
                "human_motion",
                vec!["people", "furniture"],
                if human_motion.is_some() {
                    Unqualified
                } else {
                    Inactive
                },
                "opt-in upstream motion generation",
                "requires completed model trajectories and rendered delivery",
            ),
            (
                "object_motion",
                vec!["architecture", "furniture", "people"],
                if object_motion.is_some() {
                    Unqualified
                } else {
                    Inactive
                },
                "opt-in swept-volume admission",
                "requires completed admitted articulation and rendered delivery",
            ),
        ];
        let plan = CapacityPlan {
            schema_version: SCHEMA_VERSION,
            request: request.clone(),
            population: ResolvedPopulation {
                generator: GeneratorIdentity {
                    schema_version: 1,
                    crate_version: env!("CARGO_PKG_VERSION").into(),
                    generator_version: bevy_zeroverse_capture::GENERATOR_VERSION,
                    source_sha256: env!("ZEROVERSE_SOURCE_SHA256").into(),
                },
                capture_engine: crate::CAPTURE_ENGINE_IDENTITY.into(),
                platform: format!(
                    "{};web={};threads={};motion={}",
                    env!("ZEROVERSE_CAPTURE_TARGET"),
                    cfg!(feature = "web"),
                    cfg!(feature = "multi_threaded"),
                    cfg!(feature = "human_motion")
                ),
                config_sha256: sha(resolved_config_json.as_bytes()),
                resolved_config_json,
                capture,
                failure_policy: "retain-error-no-resampling-v1".into(),
            },
            factors: descriptions
                .into_iter()
                .map(
                    |(id, parents, state, distribution, evidence_scope)| FactorInventory {
                        id: id.into(),
                        parents: parents.into_iter().map(String::from).collect(),
                        state,
                        distribution: distribution.into(),
                        evidence_scope: evidence_scope.into(),
                    },
                )
                .collect(),
        };
        validate_plan(&plan)?;
        Ok(plan)
    }

    /// Pure evidence query: mismatched configuration/source is an error, and no
    /// receipt means an explicit unqualified result rather than implicit work.
    pub fn estimate_capacity(
        &self,
        request: &CapacityRequest,
        capture: CapturePlan,
        receipt: &CapacityReceipt,
    ) -> Result<CapacityEstimate, String> {
        let plan = self.capacity_plan(request, capture)?;
        validate_receipt(&plan, receipt)?;
        Ok(receipt.estimate.clone())
    }
}

/// Explicit conditional CLI default. This does not change viewer or dataset defaults.
pub fn default_capture() -> CapturePlan {
    CapturePlan {
        image_size: [256, 256],
        cameras: 4,
        progress: vec![0.],
        seconds: None,
        modes: vec!["color".into()],
        annotation_glass: "surface".into(),
    }
}

/// CPU construction uses the production ordering, including appearance before
/// camera admission. It cannot substitute static poses for requested motion.
pub fn generate(plan: &CapacityPlan, draw: u32) -> Result<super::layout::IndoorManifest, String> {
    validate_plan(plan)?;
    if plan.population.generator.source_sha256 != env!("ZEROVERSE_SOURCE_SHA256")
        || plan.population.capture_engine != crate::CAPTURE_ENGINE_IDENTITY
    {
        return Err("capacity plan was made by a different build; resolve a new plan".into());
    }
    let config: serde_json::Value =
        serde_json::from_str(&plan.population.resolved_config_json).map_err(|e| e.to_string())?;
    if sha(plan.population.resolved_config_json.as_bytes()) != plan.population.config_sha256 {
        return Err("capacity config hash mismatch".into());
    }
    let args: BevyZeroverseConfig =
        serde_json::from_value(config["runtime"].clone()).map_err(|e| e.to_string())?;
    if args.human_motion.is_some() || args.indoor_motion.is_some() {
        return Err("actor-motion qualification requires completed render captures".into());
    }
    let seed = seed_at(&plan.request.seed_policy, draw);
    let mut scene = super::layout::IndoorManifest::generate_with_humans(
        seed,
        args.indoor_layout,
        args.indoor_density,
        0,
        args.indoor_human_density,
    )?;
    if let Some(json) = args.indoor_appearance {
        scene.apply_appearance(AppearanceSettings::parse(&json)?)?;
    }
    let camera = args
        .indoor_camera
        .as_deref()
        .map(CameraSettings::parse)
        .transpose()?
        .unwrap_or_default();
    scene.resample_cameras(
        plan.population.capture.cameras as usize,
        camera,
        plan.population.capture.image_size[0] as f32 / plan.population.capture.image_size[1] as f32,
    )?;
    if args.rotation_augmentation {
        use rand::Rng;
        scene.world_yaw = super::layout::stream(seed, 11).random_range(0.0..std::f32::consts::TAU);
    }
    Ok(scene)
}

#[cfg(test)]
mod tests;
