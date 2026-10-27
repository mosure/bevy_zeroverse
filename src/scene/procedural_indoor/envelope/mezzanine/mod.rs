//! Mezzanine construction, admission and queries share the same recorded plan.
//! Stair support volumes remain conservative for camera/human motion planning;
//! exact annotation visibility continues to use the built meshes.
use super::*;
use crate::scene::procedural_indoor::materials::Surface;
mod geometry;
mod legacy;
#[cfg(test)]
mod qualification;
mod queries;
mod sampling;
#[cfg(test)]
mod tests;
mod validation;
pub(super) use geometry::build;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StairStructure {
    Solid,
    ClosedStringer,
    OpenStringer,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportSection {
    Square,
    Round,
    ISection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Landing {
    /// Number of risers climbed before the extended horizontal tread.
    pub after_riser: u32,
    pub extra_run: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Construction {
    /// Ascend min -> max rather than the legacy max -> min direction.
    pub ascending_positive: bool,
    pub landing: Option<Landing>,
    pub stair_structure: StairStructure,
    pub tread_thickness: f32,
    pub nosing_depth: f32,
    pub frame_surface: Surface,
    pub deck_surface: Surface,
    pub tread_surface: Surface,
    pub support_section: SupportSection,
    pub support_width: f32,
    pub support_depth: f32,
    pub support_inset: f32,
    pub support_pitch: f32,
    pub beam_depth: f32,
    pub guard: super::guardrail::GuardSection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mezzanine {
    pub deck: FloorPatch,
    pub thickness: f32,
    pub stair_min: Vec2,
    pub stair_max: Vec2,
    /// X/Z run axis. Without construction metadata, ascent is max -> min.
    pub stair_axis: usize,
    pub steps: u32,
    pub rail_height: f32,
    /// Absent archives keep their original geometry, materials and queries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<Construction>,
}

#[derive(Clone, Copy)]
pub(super) struct RailSpan {
    pub start: Vec3,
    pub end: Vec3,
    pub stair: bool,
}

pub(super) struct LandingFrame {
    pub columns: [(Vec3, Vec3); 2],
    pub beam: (Vec3, Vec3),
}

impl Construction {
    pub fn validate(&self, m: &Mezzanine) -> Result<(), String> {
        if m.stair_axis > 1 || !(6..=64).contains(&m.steps) {
            return Err("invalid mezzanine stair dimensions".into());
        }
        self.guard.validate()?;
        for (v, lo, hi) in [
            (self.tread_thickness, 0.03, 0.09),
            (self.nosing_depth, 0., 0.065),
            (self.support_width, 0.10, 0.26),
            (self.support_depth, 0.10, 0.30),
            (self.support_inset, 0.12, 0.32),
            (self.support_pitch, 1.8, 3.6),
            (self.beam_depth, 0.10, 0.28),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err("invalid mezzanine construction dimensions".into());
            }
        }
        if m.deck.height - m.thickness - self.beam_depth < 2.2
            || m.stair_width() < 0.9
            || m.stair_width() > 1.9
            || self.support_inset < self.support_width.max(self.support_depth) * 0.5 + 0.015
            || self.tread_thickness >= m.riser_height()
            || ![self.frame_surface, self.deck_surface, self.tread_surface]
                .into_iter()
                .all(|s| {
                    matches!(
                        s,
                        Surface::Wood
                            | Surface::Floor
                            | Surface::Metal
                            | Surface::Concrete
                            | Surface::Paint
                    )
                })
            || self.support_section == SupportSection::ISection
                && self.frame_surface != Surface::Metal
        {
            return Err("invalid mezzanine support or finish".into());
        }
        if let Some(l) = &self.landing {
            if l.after_riser < 3
                || l.after_riser + 3 > m.steps
                || !(0.8..=1.8).contains(&l.extra_run)
            {
                return Err("invalid intermediate stair landing".into());
            }
        }
        Ok(())
    }
}
