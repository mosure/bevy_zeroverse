//! UI state and validation are independent of widgets, rendering and the browser.
use super::super::BevyZeroverseConfig;
use crate::scene::procedural_indoor::{appearance::AppearanceSettings, cameras::CameraSettings};
use bevy::prelude::*;
use serde_json::Value;

pub use super::super::view_state::CameraPose;
pub use super::super::view_state::{Page, ViewState};

#[derive(Resource)]
pub struct EditorState {
    pub draft: Value,
    pub applied: Value,
    pub view: ViewState,
    pub revision: u64,
    pub error: Option<String>,
    pub input_errors: std::collections::BTreeMap<String, String>,
    pub invalid_text: std::collections::BTreeMap<String, String>,
    pub active_seed: Option<u64>,
    pub debug: bool,
    /// Explicitly imported evidence only. Never serialized into the scene URL.
    pub capacity_receipt: Option<bevy_zeroverse_capture::capacity::CapacityReceipt>,
}
impl EditorState {
    pub fn new(config: &BevyZeroverseConfig) -> Self {
        let mut error = None;
        let view = config
            .viewer_state
            .as_deref()
            .map(ViewState::parse)
            .transpose()
            .unwrap_or_else(|e| {
                error = Some(e);
                None
            })
            .unwrap_or_default();
        let applied = expand(config);
        let mut draft = applied.clone();
        if let Some(pending) = &view.pending {
            if let Some(fields) = pending.as_object() {
                for (key, value) in fields {
                    if applied.get(key).is_some() && key != "viewer_state" {
                        super::delta::merge(&mut draft[key], value);
                    } else {
                        error = Some(format!("Unknown pending setting: {key}"));
                    }
                }
                if let Err(e) = validate(&draft) {
                    error = Some(e);
                    draft = applied.clone();
                }
            } else {
                error = Some("Pending settings must be a JSON object".into());
            }
        }
        Self {
            applied,
            draft,
            view,
            revision: 0,
            error,
            input_errors: default(),
            invalid_text: default(),
            active_seed: None,
            debug: false,
            capacity_receipt: None,
        }
    }
    pub fn pending(&self) -> bool {
        self.draft != self.applied
    }
    pub fn set(&mut self, path: &str, value: Value) -> Result<(), String> {
        let slot = self
            .draft
            .pointer_mut(path)
            .ok_or_else(|| format!("Unknown control: {path}"))?;
        *slot = value;
        self.error = None;
        Ok(())
    }
    pub fn config(&self) -> Result<BevyZeroverseConfig, String> {
        validate(&self.draft)
    }
    pub fn capacity_status(&self) -> String {
        let Some(receipt) = &self.capacity_receipt else {
            return "No imported qualification receipt. Audit explicitly with indoor_capacity; changing controls never starts capture or a model.".into();
        };
        let plan = self.config().and_then(|c| {
            c.capacity_plan(
                &receipt.plan.request,
                receipt.plan.population.capture.clone(),
            )
        });
        let matches = plan.is_ok_and(|p| p == receipt.plan);
        let e = &receipt.estimate;
        let c = &receipt.plan.population.capture;
        format!("{} · {:?} · {}×{} · {} views × {} times · {}/{} accepted rooms, {} rendered. Forecast horizon: {} draws. Counts are descriptor-specific; no automatic capacity or learning-utility claim.",
            if matches {"Matching evidence"} else {"Evidence does not match these controls/build"}, receipt.plan.request.unit,c.image_size[0],c.image_size[1],c.cameras,c.progress.len(),e.accepted_rooms,e.requested_rooms,e.rendered_rooms,e.definitions.iter().map(|d|d.max_qualified_forecast_draws).min().unwrap_or(0))
    }
}
pub fn expand(config: &BevyZeroverseConfig) -> Value {
    super::super::query::expand(config)
}

pub(super) fn optics(draft: &Value) -> &Value {
    &draft["indoor_camera"]
}
pub(super) fn optics_mut(draft: &mut Value) -> &mut Value {
    &mut draft["indoor_camera"]
}

pub fn validate(draft: &Value) -> Result<BevyZeroverseConfig, String> {
    let config: BevyZeroverseConfig =
        serde_json::from_value(draft.clone()).map_err(|e| e.to_string())?;
    CameraSettings::parse(config.indoor_camera.as_deref().unwrap_or("{}"))?;
    AppearanceSettings::parse(config.indoor_appearance.as_deref().unwrap_or("{}"))?;
    crate::scene::procedural_indoor::items::distribution::ClutterSettings::parse(
        config.indoor_clutter.as_deref().unwrap_or("{}"),
    )?;
    if let Some(m) = &config.human_motion {
        crate::human_motion::HumanMotionConfig::parse(m)?;
    }
    if !config.width.is_finite()
        || !config.height.is_finite()
        || !(16.0..=8192.0).contains(&config.width)
        || !(16.0..=8192.0).contains(&config.height)
        || config.num_cameras > 16
    {
        return Err("Resolution must be 16–8192 pixels; camera count must be 0–16".into());
    }
    if !(0.0..=1.).contains(&config.indoor_density)
        || !(0.0..=1.).contains(&config.indoor_human_density)
    {
        return Err("Density must be between 0 and 1".into());
    }
    config.validate_ovoxel()?;
    crate::render::ssaa::validate_factor(config.ssaa_factor)?;
    if config.playback_steps == 0
        || !(0.0..=1.0).contains(&config.playback_step)
        || !config.playback_speed.is_finite()
        || config.playback_speed < 0.
        || !config.yaw_speed.is_finite()
    {
        return Err(
            "Capture needs at least one timestep, a normalized step and finite playback settings"
                .into(),
        );
    }
    Ok(config)
}
pub fn live(path: &str) -> bool {
    matches!(
        path,
        "/render_mode"
            | "/depth_format"
            | "/z_depth"
            | "/annotation_glass"
            | "/camera_grid"
            | "/room_schematic"
            | "/material_grid"
            | "/gizmos"
            | "/gizmos_alpha"
            | "/draw_obb_gizmo"
            | "/draw_pose_gizmos"
            | "/playback_mode"
            | "/playback_speed"
            | "/yaw_speed"
            | "/keybinds"
            | "/regenerate_ms"
            | "/orbit_smoothness"
            | "/pan_smoothness"
            | "/zoom_smoothness"
    )
}
