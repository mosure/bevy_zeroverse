//! Optional text-conditioned motion. Planning and metadata never load a model.
//! Runtime inference is gated by the `human_motion` Cargo feature and configuration.
#[cfg(feature = "human_motion")]
mod contact;
mod navigation;
pub mod planning;
pub use navigation::NavigationRecipe;
mod prompts;
pub use prompts::{
    ActionPhase, Family as MotionFamily, PromptRecipe, PromptSamplingConfig, PROMPT_PROGRAM_VERSION,
};
#[cfg(feature = "human_motion")]
mod runtime;
#[cfg(feature = "human_motion")]
mod skin;
pub mod validation;
#[cfg(feature = "human_motion")]
pub use runtime::{HumanMotionClips, HumanMotionPlugin};

use bevy::prelude::*;
use burn_human_motion::{MotionRequest, Waypoint};
use serde::{Deserialize, Serialize};

/// Immutable policy captured when a scene is requested. Editing a viewer slider
/// must not cancel inference, reload models or replace the active actors.
#[derive(Component, Clone, Default)]
#[cfg_attr(not(feature = "human_motion"), allow(dead_code))]
pub(crate) struct SceneMotionPolicy(pub Option<String>);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HumanMotionConfig {
    /// Requested fraction of people (rounded up, capped by max_actors).
    pub fraction: f32,
    /// Share of automatically selected people staged in navigable standing space.
    /// Others keep their support and may sit, stand or gesture in a chair.
    pub locomotion_fraction: f32,
    /// Share of feasible walking clips composed with a timed action and resumption.
    pub sequence_fraction: f32,
    /// Share of travel-gait proposals using skipping/jogging when speed and space permit.
    pub energetic_fraction: f32,
    pub prompt_sampling: PromptSamplingConfig,
    pub max_actors: usize,
    pub frames: usize,
    pub diffusion_steps: usize,
    /// Context retained between ARDY autoregressive windows, at 20 Hz.
    pub history_frames: usize,
    /// Constrain the whole interpolated navigation path, not only key waypoints.
    pub dense_trajectory: bool,
    pub batch_size: usize,
    /// Bounded independent model samples for rejected clips; models stay cached.
    pub max_attempts: usize,
    pub text_guidance: f32,
    pub trajectory_guidance: f32,
    /// Fail capture instead of retaining the static person when a clip is rejected.
    pub strict: bool,
    /// A CDN or local mirror root, retaining the loaders' pinned artifact digests.
    pub model_root: Option<String>,
    pub trajectories: Vec<HumanTrajectory>,
}
impl Default for HumanMotionConfig {
    fn default() -> Self {
        Self {
            fraction: 0.35,
            locomotion_fraction: 0.85,
            sequence_fraction: 0.45,
            energetic_fraction: 0.15,
            prompt_sampling: PromptSamplingConfig::default(),
            max_actors: 8,
            frames: 160,
            diffusion_steps: 10,
            history_frames: 80,
            dense_trajectory: true,
            batch_size: 2,
            max_attempts: 2,
            text_guidance: 2.0,
            trajectory_guidance: 3.0,
            strict: false,
            model_root: None,
            trajectories: Vec::new(),
        }
    }
}
impl HumanMotionConfig {
    pub fn parse(json: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(json).map_err(|e| format!("human_motion: {e}"))?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), String> {
        self.prompt_sampling.validate()?;
        if !self.fraction.is_finite()
            || !(0.0..=1.0).contains(&self.fraction)
            || !self.locomotion_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.locomotion_fraction)
            || !(0.0..=1.0).contains(&self.sequence_fraction)
            || !(0.0..=1.0).contains(&self.energetic_fraction)
            || !(1..=16).contains(&self.max_actors)
            || !(1..=8).contains(&self.batch_size)
            || !(1..=3).contains(&self.max_attempts)
            || !(40..=640).contains(&self.frames)
            || !self.frames.is_multiple_of(4)
            || !(1..=10).contains(&self.diffusion_steps)
            || self.history_frames > 160
            || !self.history_frames.is_multiple_of(4)
            || [self.text_guidance, self.trajectory_guidance]
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=10.0).contains(v))
            || self.trajectories.len() > self.max_actors
        {
            return Err("invalid motion fraction, actor limit (1..16), batch limit (1..8), frames (40..640, multiple of 4), steps (1..10), history (0..160, multiple of 4), or guidance".into());
        }
        let mut ids = std::collections::HashSet::new();
        for trajectory in &self.trajectories {
            if !ids.insert(trajectory.actor_id)
                || trajectory.prompt.trim().is_empty()
                || trajectory.prompt.len() > 240
                || trajectory.waypoints.len() < 2
            {
                return Err("trajectories require unique actor IDs, a short prompt and at least two waypoints".into());
            }
            let request = self.request(trajectory.prompt.clone(), 0, trajectory.waypoints.clone());
            request.validate().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn request(&self, prompt: String, seed: u64, waypoints: Vec<Waypoint>) -> MotionRequest {
        MotionRequest {
            prompt,
            seed,
            frames: self.frames,
            history_frames: self.history_frames,
            diffusion_steps: self.diffusion_steps,
            text_guidance: self.text_guidance,
            trajectory_guidance: self.trajectory_guidance,
            waypoints,
            dense_trajectory: self.dense_trajectory,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HumanTrajectory {
    pub actor_id: usize,
    pub prompt: String,
    /// Optional contact furniture for custom sit/stand prompts. Otherwise the
    /// actor's assigned chair is used. Its actual mesh is still collision tested.
    #[serde(default)]
    pub support_chair: Option<usize>,
    /// Scene-local metres, +Y up. Frame numbers are at 20 Hz.
    pub waypoints: Vec<Waypoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionPlan {
    pub actor_id: usize,
    pub behavior: String,
    pub request: MotionRequest,
    pub support_chair: Option<usize>,
    /// Sampled grammar and physical action schedule; absent for explicit prompts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_recipe: Option<PromptRecipe>,
}

#[derive(Resource, Default, Clone, Debug, Serialize, Deserialize)]
pub struct HumanMotionReport {
    pub scene_seed: u64,
    pub pending: bool,
    pub stage: String,
    pub model_loads: usize,
    pub generated_batches: usize,
    pub embedding_hits: usize,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub model_artifacts: std::collections::BTreeMap<String, String>,
    pub requested: Vec<MotionPlan>,
    pub accepted: Vec<MotionPlan>,
    pub rejected: Vec<MotionRejection>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionRejection {
    pub actor_id: usize,
    pub reason: String,
}

#[cfg(not(feature = "human_motion"))]
pub struct HumanMotionPlugin;
#[cfg(not(feature = "human_motion"))]
impl Plugin for HumanMotionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::sample::CaptureFailure>();
        app.add_systems(
            PreUpdate,
            |config: Res<crate::app::BevyZeroverseConfig>,
             mut failure: ResMut<crate::sample::CaptureFailure>| {
                if config.human_motion.is_some() && failure.0.is_none() {
                    failure.0 = Some("human_motion requires cargo --features human_motion".into());
                    error!("{}", failure.0.as_ref().unwrap());
                }
            },
        );
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    #[test]
    fn sampler_controls_validate_and_reach_model_requests() {
        let policy = HumanMotionConfig::parse(r#"{"diffusion_steps":8,"history_frames":40,"frames":124,"text_guidance":1.5,"trajectory_guidance":2.5,"dense_trajectory":false}"#).unwrap();
        let request = policy.request("A person walks.".into(), 17, vec![]);
        assert_eq!(
            (
                request.diffusion_steps,
                request.history_frames,
                request.frames
            ),
            (8, 40, 124)
        );
        assert_eq!(
            (request.text_guidance, request.trajectory_guidance),
            (1.5, 2.5)
        );
        assert!(!request.dense_trajectory);
        for invalid in [
            r#"{"diffusion_steps":11}"#,
            r#"{"history_frames":3}"#,
            r#"{"history_frames":164}"#,
        ] {
            assert!(HumanMotionConfig::parse(invalid).is_err());
        }
        let legacy = HumanMotionConfig::parse("{}").unwrap();
        assert_eq!(legacy.history_frames, 80);
        assert!(legacy.dense_trajectory);
    }
}
