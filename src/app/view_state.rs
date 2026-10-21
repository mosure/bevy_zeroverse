//! Serializable studio state, independent of widgets and render schedules.
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
            Self::Cameras => "Baseline separates views. Lenses control framing. Motion selects which cameras travel.",
            Self::Appearance => {
                "Vary surfaces and lighting independently while preserving geometry."
            }
            Self::People => "People, furniture and portable-item motion. Every animation is optional.",
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
