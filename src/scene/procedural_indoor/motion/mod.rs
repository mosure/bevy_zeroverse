//! Optional deterministic articulated motion. Construction, clearance and mesh
//! splitting run only for an explicitly supplied policy. All-time reservations
//! deliberately trade some motion acceptance for safe, reversible playback.
mod appearance;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod capture_tests;
mod cloth;
mod operation;
pub(crate) use appearance::{display, AppearancePlayback, DimmedLight};
pub use appearance::{AppearanceTimeline, DimmerCurve, LightChange, LightSource, ScreenChange};
pub use operation::OperationCycle;
pub(crate) mod construction;
pub(crate) use cloth::ClothMotion;
mod mobility;
mod planning;
mod proposals;
mod roller;
mod runtime;
pub(crate) use roller::RollerFeed;
#[cfg(test)]
mod tests;

use super::{geometry::Geometry, layout::stream, materials::Surface, objects::Assembly};
use bevy::prelude::*;
pub(crate) use planning::prepare;
use rand::Rng;
pub use runtime::IndoorMotionPlugin;
pub(crate) use runtime::{AnimatedBounds, AnimatedLight, AnimatedPart};
use serde::{Deserialize, Serialize};

type Bounds = (Vec3, Vec3);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndoorMotionConfig {
    pub chair_fraction: f32,
    pub door_fraction: f32,
    pub cabinet_fraction: f32,
    pub covering_fraction: f32,
    /// Probability of winding a spring or motorized projection screen.
    pub projection_screen_fraction: f32,
    /// Probability per hinged outer leaf of a folding teaching board.
    pub teaching_board_fraction: f32,
    /// Fraction of admitted doors, drawers, lids and board wings that pause and return.
    pub operation_return_fraction: f32,
    pub fan_fraction: f32,
    /// Probability of a desk fan running on its own spindle. Shares `fan_turns`.
    pub desk_fan_fraction: f32,
    pub appliance_fraction: f32,
    pub prop_fraction: f32,
    /// Among selected hand-held tabletop props, fraction lifted and set down.
    pub prop_pickup_fraction: f32,
    /// Maximum height above the original support plane during a pickup.
    pub prop_lift_m: f32,
    /// Probability of a page change on an awake optical display.
    pub screen_change_fraction: f32,
    /// Probability of a smooth power change per installed ceiling circuit or powered lamp.
    pub light_change_fraction: f32,
    pub cart_fraction: f32,
    /// Maximum supported sliding travel, in metres.
    pub prop_travel_m: f32,
    pub prop_rotation_degrees: f32,
    pub cart_travel_m: f32,
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
            projection_screen_fraction: 0.,
            teaching_board_fraction: 0.,
            operation_return_fraction: 0.,
            fan_fraction: 0.8,
            desk_fan_fraction: 0.,
            appliance_fraction: 0.5,
            prop_fraction: 0.35,
            prop_pickup_fraction: 0.,
            prop_lift_m: 0.16,
            screen_change_fraction: 0.,
            light_change_fraction: 0.,
            cart_fraction: 0.5,
            prop_travel_m: 0.25,
            prop_rotation_degrees: 60.,
            cart_travel_m: 0.8,
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
            self.projection_screen_fraction,
            self.teaching_board_fraction,
            self.operation_return_fraction,
            self.fan_fraction,
            self.desk_fan_fraction,
            self.appliance_fraction,
            self.prop_fraction,
            self.prop_pickup_fraction,
            self.screen_change_fraction,
            self.light_change_fraction,
            self.cart_fraction,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || !(1..=128).contains(&self.max_mechanisms)
            || !(0.0..=0.45).contains(&self.prop_lift_m)
            || !self.prop_travel_m.is_finite()
            || !(0.0..=0.75).contains(&self.prop_travel_m)
            || !self.prop_rotation_degrees.is_finite()
            || !(0.0..=180.).contains(&self.prop_rotation_degrees)
            || !self.cart_travel_m.is_finite()
            || !(0.0..=2.0).contains(&self.cart_travel_m)
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
            return Err("indoor_motion requires fractions in [0,1], 1..128 mechanisms, chair travel 0..1.5 m, swivel 0..120 degrees, prop travel 0..0.75 m, lift 0..0.45 m, prop rotation 0..180 degrees, cart travel 0..2 m and ordered fan_turns in [0,120]".into());
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
            MotionKind::Appliance => self.appliance_fraction,
            MotionKind::Prop => self.prop_fraction,
            MotionKind::Cart => self.cart_fraction,
            MotionKind::TeachingBoard => self.teaching_board_fraction,
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
    Appliance,
    Prop,
    Cart,
    TeachingBoard,
}

/// Capture manifest metadata. Bounds are in scene-local metres; playback maps
/// the shared normalized timeline, including reverse/sinusoidal playback.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndoorMotionReport {
    pub version: u32,
    pub policy: IndoorMotionConfig,
    pub mechanisms: Vec<MechanismReport>,
    #[serde(default, skip_serializing_if = "AppearanceTimeline::is_empty")]
    pub appearance: AppearanceTimeline,
}
impl IndoorMotionReport {
    pub fn new(policy: IndoorMotionConfig) -> Self {
        Self {
            version: 8,
            policy,
            mechanisms: vec![],
            appearance: Default::default(),
        }
    }
    pub fn reservations(&self) -> impl Iterator<Item = Bounds> + '_ {
        self.mechanisms
            .iter()
            .filter(|m| m.rejection.is_none())
            .flat_map(|m| m.swept_bounds.iter().copied())
    }

    pub(crate) fn mechanism_for_object(&self, id: usize) -> Option<&MechanismReport> {
        self.mechanisms.iter().find(|m| {
            m.rejection.is_none() && (m.object_id == Some(id) || m.carried_object_ids.contains(&id))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkedSlide {
    pub joint: usize,
    pub translation_m: Vec3,
    pub requested_translation_m: Vec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionScreenTravel {
    /// Exposed sheet length from cassette lip to the top of the weighted rail.
    pub start_drop_m: f32,
    pub requested_end_drop_m: f32,
    pub end_drop_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MechanismReport {
    /// Four absolute timeline instants for an admitted out-and-back operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation_cycle: Option<OperationCycle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_screen_travel: Option<ProjectionScreenTravel>,
    pub object_id: Option<usize>,
    /// Objects carried rigidly by this mechanism, retaining their own instance IDs.
    #[serde(default)]
    pub carried_object_ids: Vec<usize>,
    /// Semantic labels of the actual driven mesh parts, including carried cargo.
    #[serde(default)]
    pub semantic_labels: Vec<String>,
    pub joint: usize,
    /// Additional leaves of one telescoping transmission. They share its
    /// timeline, admission and swept reservation, with their own travel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_slides: Vec<LinkedSlide>,
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
    /// Peak vertical excursion. Both endpoints lie on the same support.
    #[serde(default)]
    pub lift_m: f32,
    #[serde(default)]
    pub requested_lift_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Contact {
    #[default]
    Free,
    Floor,
    Surface,
}

/// Motion in the assembly coordinate system. Rigid motion is exact; cloth
/// endpoint interpolation preserves topology/UVs and fixed hem/rail attachments.
#[derive(Clone)]
pub(crate) struct Joint {
    program_seed: u64,
    operation_cycle: Option<OperationCycle>,
    pub kind: MotionKind,
    pub interval: [f32; 2],
    pub pivot: Vec3,
    /// Construction-aligned frame used only for tight swept collision bounds.
    pub frame: Transform,
    pub cloth_motion: Option<ClothMotion>,
    pub axis: Vec3,
    pub angle: f32,
    pub translation: Vec3,
    pub lift_m: f32,
    pub parts: Vec<(Surface, String)>,
    pub morph: std::collections::BTreeMap<(Surface, String), Geometry>,
    pub accepted: bool,
    pub contact: Contact,
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
            program_seed: seed,
            operation_cycle: None,
            interval: [start, r.random_range(0.7..1.0)],
            pivot: Vec3::ZERO,
            frame: Transform::IDENTITY,
            cloth_motion: None,
            axis: Vec3::Y,
            angle: 0.,
            translation: Vec3::ZERO,
            lift_m: 0.,
            parts: vec![],
            morph: default(),
            accepted: false,
            contact: Contact::Free,
            coupled: false,
            attachment: None,
            adjustment: default(),
        }
    }
    pub fn subset(&self, parts: &std::collections::BTreeMap<(Surface, String), Geometry>) -> Self {
        Self {
            kind: self.kind,
            interval: self.interval,
            program_seed: self.program_seed,
            operation_cycle: self.operation_cycle,
            pivot: self.pivot,
            frame: self.frame,
            cloth_motion: self.cloth_motion,
            axis: self.axis,
            angle: self.angle,
            translation: self.translation,
            lift_m: self.lift_m,
            coupled: self.coupled,
            contact: self.contact,
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
        motion_amount(self.kind, self.interval, self.operation_cycle, t)
    }
    pub fn transform(&self, amount: f32) -> Transform {
        let q = Quat::from_axis_angle(self.axis, self.effective_angle() * amount);
        Transform::from_translation(
            self.pivot - q * self.pivot
                + self.effective_translation() * amount
                + lift_offset(self.effective_lift(), amount),
        )
        .with_rotation(q)
    }
    fn effective_angle(&self) -> f32 {
        self.angle * self.adjustment.rotation
    }
    fn effective_lift(&self) -> f32 {
        self.lift_m * self.adjustment.lift
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

fn motion_amount(
    kind: MotionKind,
    interval: [f32; 2],
    cycle: Option<OperationCycle>,
    progress: f32,
) -> f32 {
    if let Some(cycle) = cycle {
        return cycle.stroke_fraction_at(progress);
    }
    if kind == MotionKind::Fan {
        return progress.clamp(0., 1.);
    }
    let t = ((progress - interval[0]) / (interval[1] - interval[0])).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

impl MechanismReport {
    /// Admitted stroke fraction at a scene instant. Applies to rigid and deforming parts.
    pub fn stroke_fraction_at(&self, progress: f32) -> f32 {
        if self.rejection.is_some() {
            return 0.;
        }
        motion_amount(self.kind, self.interval, self.operation_cycle, progress)
    }

    /// Visible cloth length at normalized playback progress, including clearance-limited travel.
    pub fn projection_screen_drop_at(&self, progress: f32) -> Option<f32> {
        let travel = self.projection_screen_travel.as_ref()?;
        Some(
            travel
                .start_drop_m
                .lerp(travel.end_drop_m, self.stroke_fraction_at(progress)),
        )
    }

    /// Scene-local rigid delta relative to the captured rest geometry. Coverings
    /// deform vertices instead; their exact positions are in render annotations.
    pub fn rigid_transform_at(&self, progress: f32) -> Option<Transform> {
        self.rigid_delta(self.translation_m, progress)
    }

    /// Delta for the reported joint or a linked sliding leaf, in scene coordinates.
    pub fn rigid_transform_for_joint_at(&self, joint: usize, progress: f32) -> Option<Transform> {
        if joint == self.joint {
            return self.rigid_transform_at(progress);
        }
        let leaf = self.linked_slides.iter().find(|s| s.joint == joint)?;
        self.rigid_delta(leaf.translation_m, progress)
    }

    fn rigid_delta(&self, translation: Vec3, progress: f32) -> Option<Transform> {
        if self.rejection.is_some() || self.kind == MotionKind::Covering {
            return None;
        }
        let t = self.stroke_fraction_at(progress);
        let q = Quat::from_axis_angle(self.axis, self.angle_radians * t);
        Some(
            Transform::from_translation(
                self.pivot - q * self.pivot + translation * t + lift_offset(self.lift_m, t),
            )
            .with_rotation(q),
        )
    }
}

fn lift_offset(height: f32, t: f32) -> Vec3 {
    Vec3::Y * (4. * t * (1. - t) * height)
}

pub(super) const TEMPORAL_PROGRAM_VERSION: u32 = 275;
pub(super) const ROTOR_CARGO_VERSION: u32 = 276;
pub(super) const PROJECTION_MOTION_VERSION: u32 = 277;
pub(super) const OPERATION_CYCLE_VERSION: u32 = 278;
