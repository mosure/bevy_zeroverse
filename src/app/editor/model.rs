//! UI state and validation are independent of widgets, rendering and the browser.
use super::super::BevyZeroverseConfig;
use crate::scene::procedural_indoor::{appearance::AppearanceSettings, cameras::CameraSettings};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Page {
    #[default]
    Scene,
    Cameras,
    Appearance,
    People,
    View,
    Advanced,
}
impl Page {
    pub const ALL: [Self; 6] = [
        Self::Scene,
        Self::Cameras,
        Self::Appearance,
        Self::People,
        Self::View,
        Self::Advanced,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Scene => "Scene",
            Self::Cameras => "Cameras",
            Self::Appearance => "Materials & light",
            Self::People => "People & motion",
            Self::View => "View & playback",
            Self::Advanced => "Advanced",
        }
    }
    pub fn intro(self) -> &'static str {
        match self {
            Self::Scene => "Build your next scene. Apply keeps the seed; Next explores a new room.",
            Self::Cameras => "Spacing separates views. Travel moves each view through the room.",
            Self::Appearance => {
                "Vary surfaces and lighting independently while preserving geometry."
            }
            Self::People => "Shape and clothing are sampled with the room. Motion is optional.",
            Self::View => "Preview annotations, overlays and time. These controls update live.",
            Self::Advanced => {
                "Exact configuration and diagnostics. Invalid values never reach generation."
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ViewState {
    pub version: u32,
    pub page: Page,
    pub collapsed: bool,
    pub expanded: std::collections::BTreeSet<String>,
    pub progress: f32,
    pub direction: f32,
    pub flow_interval: f32,
    pub flow_scale: f32,
    pub camera: Option<CameraPose>,
    pub scene_rotation: Option<[f32; 4]>,
    /// Unapplied controls travel with the link without changing the rendered scene.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraPose {
    pub focus: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub fov: f32,
}
impl Default for ViewState {
    fn default() -> Self {
        Self {
            version: 1,
            page: Page::Scene,
            collapsed: false,
            expanded: default(),
            progress: 0.,
            direction: 1.,
            flow_interval: 0.05,
            flow_scale: 32.,
            camera: None,
            scene_rotation: None,
            pending: None,
        }
    }
}
impl ViewState {
    pub fn parse(s: &str) -> Result<Self, String> {
        let v: Self = serde_json::from_str(s).map_err(|e| e.to_string())?;
        v.validate()?;
        Ok(v)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || !(0.0..=1.0).contains(&self.progress)
            || ![-1., 1.].contains(&self.direction)
            || !(0.005..=0.5).contains(&self.flow_interval)
            || !(1.0..=256.).contains(&self.flow_scale)
        {
            return Err("Invalid viewer state version, time or flow settings".into());
        }
        if self.scene_rotation.is_some_and(|r| {
            !Quat::from_array(r).is_finite()
                || (Quat::from_array(r).length_squared() - 1.).abs() > 0.01
        }) {
            return Err("Invalid scene rotation".into());
        }
        if self.camera.as_ref().is_some_and(|c| {
            !c.focus.iter().all(|f| f.is_finite())
                || !c.yaw.is_finite()
                || !c.pitch.is_finite()
                || !(0.01..=10000.).contains(&c.radius)
                || !(0.05..=3.05).contains(&c.fov)
        }) {
            return Err("Invalid editor camera".into());
        }
        Ok(())
    }
}

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
                        draft[key] = value.clone();
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
}
pub fn expand(config: &BevyZeroverseConfig) -> Value {
    let mut v = serde_json::to_value(config).expect("serializable config");
    v["indoor_camera"] = json!(config
        .indoor_camera
        .as_deref()
        .and_then(|s| CameraSettings::parse(s).ok())
        .unwrap_or_default());
    for key in ["duration_seconds", "handheld", "overlap_mixture"] {
        v["indoor_camera"]
            .as_object_mut()
            .unwrap()
            .entry(key)
            .or_insert(Value::Null);
    }
    v["indoor_appearance"] = json!(config
        .indoor_appearance
        .as_deref()
        .and_then(|s| AppearanceSettings::parse(s).ok())
        .unwrap_or_default());
    v["human_motion"] = config
        .human_motion
        .as_deref()
        .and_then(|s| crate::human_motion::HumanMotionConfig::parse(s).ok())
        .map(|v| json!(v))
        .unwrap_or(Value::Null);
    v["viewer_state"] = Value::Null;
    v
}
pub fn validate(draft: &Value) -> Result<BevyZeroverseConfig, String> {
    let config: BevyZeroverseConfig =
        serde_json::from_value(draft.clone()).map_err(|e| e.to_string())?;
    CameraSettings::parse(config.indoor_camera.as_deref().unwrap_or("{}"))?;
    AppearanceSettings::parse(config.indoor_appearance.as_deref().unwrap_or("{}"))?;
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
