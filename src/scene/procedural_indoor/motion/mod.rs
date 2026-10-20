//! Optional deterministic articulated motion. Construction, clearance and mesh
//! splitting run only for an explicitly supplied policy. All-time reservations
//! deliberately trade some motion acceptance for safe, reversible playback.
pub(crate) mod construction;
mod planning;
mod proposals;
mod runtime;
#[cfg(test)]
mod tests;

use super::{geometry::Geometry, layout::stream, materials::Surface, objects::Assembly};
use bevy::prelude::*;
pub(crate) use planning::prepare;
use rand::Rng;
pub use runtime::IndoorMotionPlugin;
pub(crate) use runtime::{AnimatedBounds, AnimatedPart};
use serde::{Deserialize, Serialize};

type Bounds = (Vec3, Vec3);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndoorMotionConfig {
    pub chair_fraction: f32,
    pub door_fraction: f32,
    pub cabinet_fraction: f32,
    pub covering_fraction: f32,
    pub fan_fraction: f32,
    /// Constant-speed revolutions over the shared normalized scene timeline.
    pub fan_turns: [f32; 2],
    /// Maximum independently driven mechanisms, not individual mesh parts.
    pub max_mechanisms: usize,
    pub chair_travel_m: f32,
    pub chair_swivel_degrees: f32,
}
impl Default for IndoorMotionConfig {
    fn default() -> Self {
        Self {
            chair_fraction: 0.35,
            door_fraction: 0.65,
            cabinet_fraction: 0.5,
            covering_fraction: 0.5,
            fan_fraction: 0.8,
            fan_turns: [0.75, 12.0],
            max_mechanisms: 12,
            chair_travel_m: 0.65,
            chair_swivel_degrees: 70.,
        }
    }
}
impl IndoorMotionConfig {
    pub fn parse(json: &str) -> Result<Self, String> {
        let p: Self = serde_json::from_str(json).map_err(|e| format!("indoor_motion: {e}"))?;
        p.validate()?;
        Ok(p)
    }
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.chair_fraction,
            self.door_fraction,
            self.cabinet_fraction,
            self.covering_fraction,
            self.fan_fraction,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !(1..=32).contains(&self.max_mechanisms)
            || !self.chair_travel_m.is_finite()
            || !(0.0..=1.5).contains(&self.chair_travel_m)
            || !self.chair_swivel_degrees.is_finite()
            || !(0.0..=120.).contains(&self.chair_swivel_degrees)
            || self
                .fan_turns
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=120.).contains(v))
            || self.fan_turns[0] > self.fan_turns[1]
        {
            return Err("indoor_motion requires fractions in [0,1], 1..32 mechanisms, travel 0..1.5 m, swivel 0..120 degrees and ordered fan_turns in [0,120]".into());
        }
        Ok(())
    }
    fn fraction(&self, kind: MotionKind) -> f32 {
        match kind {
            MotionKind::Chair => self.chair_fraction,
            MotionKind::Door => self.door_fraction,
            MotionKind::Cabinet => self.cabinet_fraction,
            MotionKind::Covering => self.covering_fraction,
            MotionKind::Fan => self.fan_fraction,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionKind {
    Chair,
    Door,
    Cabinet,
    Covering,
    Fan,
}

/// Capture manifest metadata. Bounds are in scene-local metres; playback maps
/// the shared normalized timeline, including reverse/sinusoidal playback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndoorMotionReport {
    pub version: u32,
    pub policy: IndoorMotionConfig,
    pub mechanisms: Vec<MechanismReport>,
}
impl IndoorMotionReport {
    pub fn new(policy: IndoorMotionConfig) -> Self {
        Self {
            version: 2,
            policy,
            mechanisms: vec![],
        }
    }
    pub fn reservations(&self) -> impl Iterator<Item = Bounds> + '_ {
        self.mechanisms
            .iter()
            .filter(|m| m.rejection.is_none())
            .flat_map(|m| m.swept_bounds.iter().copied())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MechanismReport {
    pub object_id: Option<usize>,
    pub joint: usize,
    pub kind: MotionKind,
    pub interval: [f32; 2],
    pub rejection: Option<String>,
    pub swept_bounds: Vec<Bounds>,
    pub translation_m: Vec3,
    pub angle_radians: f32,
    pub pivot: Vec3,
    pub axis: Vec3,
    /// Number of clearance-tested candidates, including the original proposal.
    #[serde(default)]
    pub attempts: usize,
    /// Describes an admitted fallback; rejection still explains truly blocked motion.
    #[serde(default)]
    pub adaptation: Option<String>,
    #[serde(default)]
    pub requested_translation_m: Vec3,
    #[serde(default)]
    pub requested_angle_radians: f32,
}

/// Motion in the assembly coordinate system. Rigid motion is exact; cloth
/// endpoint interpolation preserves topology/UVs and fixed hem/rail attachments.
#[derive(Clone)]
pub(crate) struct Joint {
    pub kind: MotionKind,
    pub interval: [f32; 2],
    pub pivot: Vec3,
    /// Construction-aligned frame used only for tight swept collision bounds.
    pub frame: Transform,
    pub axis: Vec3,
    pub angle: f32,
    pub translation: Vec3,
    pub parts: Vec<(Surface, String)>,
    pub morph: std::collections::BTreeMap<(Surface, String), Geometry>,
    pub accepted: bool,
    /// A rotor or linked mechanism must be admitted as one rigid assembly.
    pub coupled: bool,
    /// Deliberate mechanical attachments belonging to this mechanism only.
    pub attachment: Option<String>,
    /// Admission changes gains rather than the original proposal or mesh endpoints.
    /// Re-running preparation is therefore deterministic and cannot shrink twice.
    pub adjustment: proposals::Adjustment,
}
impl Joint {
    fn new(kind: MotionKind, seed: u64) -> Self {
        let mut r = stream(seed, 0x54494d45);
        let start = r.random_range(0.0..0.2);
        Self {
            kind,
            interval: [start, r.random_range(0.7..1.0)],
            pivot: Vec3::ZERO,
            frame: Transform::IDENTITY,
            axis: Vec3::Y,
            angle: 0.,
            translation: Vec3::ZERO,
            parts: vec![],
            morph: default(),
            accepted: false,
            coupled: false,
            attachment: None,
            adjustment: default(),
        }
    }
    pub fn subset(&self, parts: &std::collections::BTreeMap<(Surface, String), Geometry>) -> Self {
        Self {
            kind: self.kind,
            interval: self.interval,
            pivot: self.pivot,
            frame: self.frame,
            axis: self.axis,
            angle: self.angle,
            translation: self.translation,
            coupled: self.coupled,
            accepted: self.accepted && self.parts.iter().any(|k| parts.contains_key(k)),
            attachment: self.attachment.clone(),
            adjustment: self.adjustment,
            parts: self
                .parts
                .iter()
                .filter(|k| parts.contains_key(*k))
                .cloned()
                .collect(),
            morph: self
                .morph
                .iter()
                .filter(|(k, _)| parts.contains_key(*k))
                .map(|(k, g)| (k.clone(), g.clone()))
                .collect(),
        }
    }
    pub fn amount(&self, t: f32) -> f32 {
        if self.kind == MotionKind::Fan {
            return t.clamp(0., 1.);
        }
        let t = ((t - self.interval[0]) / (self.interval[1] - self.interval[0])).clamp(0., 1.);
        t * t * (3. - 2. * t)
    }
    pub fn transform(&self, amount: f32) -> Transform {
        let q = Quat::from_axis_angle(self.axis, self.effective_angle() * amount);
        Transform::from_translation(
            self.pivot - q * self.pivot + self.effective_translation() * amount,
        )
        .with_rotation(q)
    }
    fn effective_angle(&self) -> f32 {
        self.angle * self.adjustment.rotation
    }
    fn effective_translation(&self) -> Vec3 {
        Quat::from_rotation_y(self.adjustment.heading)
            * self.translation
            * self.adjustment.translation
    }
}
pub(crate) fn tag(label: &str) -> Option<usize> {
    label
        .split('#')
        .find_map(|s| s.strip_prefix("joint").and_then(|s| s.parse().ok()))
}
fn tagged(label: &str, id: usize) -> String {
    let (class, rest) = label.split_once('#').unwrap_or((label, ""));
    if rest.is_empty() {
        format!("{class}#joint{id}")
    } else {
        format!("{class}#joint{id}#{rest}")
    }
}

impl MechanismReport {
    /// Scene-local rigid delta relative to the captured rest geometry. Coverings
    /// deform vertices instead; their exact positions are in render annotations.
    pub fn rigid_transform_at(&self, progress: f32) -> Option<Transform> {
        if self.rejection.is_some() || self.kind == MotionKind::Covering {
            return None;
        }
        let t = if self.kind == MotionKind::Fan {
            progress.clamp(0., 1.)
        } else {
            let t = ((progress - self.interval[0]) / (self.interval[1] - self.interval[0]))
                .clamp(0., 1.);
            t * t * (3. - 2. * t)
        };
        let q = Quat::from_axis_angle(self.axis, self.angle_radians * t);
        Some(
            Transform::from_translation(self.pivot - q * self.pivot + self.translation_m * t)
                .with_rotation(q),
        )
    }
}
