//! One architectural description for construction, support, camera clearance,
//! visibility and exports. It is a bounded procedural building grammar, not CAD.
pub mod construction;
pub mod polygon;
mod queries;
mod sampling;
#[cfg(test)]
mod tests;
use super::{architecture::facade::Facade, layout::IndoorManifest};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorPatch {
    pub min: Vec2,
    pub max: Vec2,
    pub height: f32,
}
impl FloorPatch {
    pub fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
    pub fn overlaps(&self, lo: Vec2, hi: Vec2) -> bool {
        lo.cmplt(self.max - Vec2::splat(1e-5)).all() && hi.cmpgt(self.min + Vec2::splat(1e-5)).all()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pillar {
    pub center: Vec2,
    pub radius: f32,
    pub sides: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ColumnProfile>,
}

/// Cross-section and taper fit within the pillar's conservative collision radius.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub aspect: f32,
    pub roundness: f32,
    pub taper: f32,
    pub rotation: f32,
    pub collar_height: f32,
}
impl ColumnProfile {
    pub fn sample(seed: u64, index: usize) -> Self {
        use rand::Rng;
        let mut rng = super::layout::stream(seed, 6120 + index as u64);
        Self {
            aspect: rng.random_range(0.62..1.0),
            roundness: rng.random_range(2.0..8.0),
            taper: rng.random_range(0.84..1.0),
            rotation: rng.random_range(0.0..std::f32::consts::PI),
            collar_height: rng.random_range(0.045..0.15),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.6..=1.0).contains(&self.aspect)
            || !(2.0..=8.0).contains(&self.roundness)
            || !(0.8..=1.0).contains(&self.taper)
            || !self.rotation.is_finite()
            || !(0.04..=0.16).contains(&self.collar_height)
        {
            return Err("invalid structural column profile".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mezzanine {
    pub deck: FloorPatch,
    pub thickness: f32,
    pub stair_min: Vec2,
    pub stair_max: Vec2,
    /// X/Z direction of the run. Stair rises from max[axis] to min[axis].
    pub stair_axis: usize,
    pub steps: u32,
    pub rail_height: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeWall {
    pub edge: usize,
    pub facade: Option<Facade>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeProgram {
    /// Simple CCW polygon in X/Z; the +Z edge connects to the neighboring room.
    pub footprint: Vec<Vec2>,
    /// Signed total height change across the room in X and Z; stored in metres.
    pub ceiling_drop: Vec2,
    /// Disjoint flat support surfaces, including each tread of level changes.
    pub floor_patches: Vec<FloorPatch>,
    pub pillars: Vec<Pillar>,
    pub mezzanine: Option<Mezzanine>,
    pub walls: Vec<EnvelopeWall>,
}

impl EnvelopeProgram {
    pub fn floor_height(&self, p: Vec2) -> f32 {
        self.floor_patches
            .iter()
            .find(|f| f.contains(p))
            .map_or(0., |f| f.height)
    }
    pub fn minimum_floor(&self) -> f32 {
        self.floor_patches
            .iter()
            .fold(0.0_f32, |a, p| a.min(p.height))
    }
    pub fn ceiling_height(&self, size: Vec3, p: Vec2) -> f32 {
        let uv = (p / size.xz() + Vec2::splat(0.5)).clamp(Vec2::ZERO, Vec2::ONE);
        size.y
            - (0..2)
                .map(|i| {
                    self.ceiling_drop[i].abs()
                        * if self.ceiling_drop[i] > 0. {
                            uv[i]
                        } else {
                            1. - uv[i]
                        }
                })
                .sum::<f32>()
    }
    pub fn support_height(&self, p: Vec3) -> f32 {
        self.mezzanine
            .as_ref()
            .filter(|m| m.deck.contains(p.xz()) && p.y >= m.deck.height - 0.01)
            .map_or_else(|| self.floor_height(p.xz()), |m| m.deck.height)
    }
    pub fn wall_transform(&self, edge: usize) -> Transform {
        let a = self.footprint[edge];
        let b = self.footprint[(edge + 1) % self.footprint.len()];
        let d = b - a;
        let c = (a + b) * 0.5;
        Transform::from_xyz(c.x, 0., c.y).with_rotation(Quat::from_rotation_y((-d.y).atan2(d.x)))
    }
    pub fn shared_edge(&self, edge: usize, size: Vec3) -> bool {
        self.footprint[edge].y > size.z * 0.5 - 0.001
            && self.footprint[(edge + 1) % self.footprint.len()].y > size.z * 0.5 - 0.001
    }
    pub fn support_clear(&self, lo: Vec3, hi: Vec3) -> bool {
        if self.mezzanine.as_ref().is_some_and(|m| {
            (lo.y - m.deck.height).abs() < 0.002
                && lo.xz().cmpge(m.deck.min + Vec2::splat(0.08)).all()
                && hi.xz().cmple(m.deck.max - Vec2::splat(0.08)).all()
        }) {
            return true;
        }
        let h = self.floor_height((lo + hi).xz() * 0.5);
        if (lo.y - h).abs() > 0.002 {
            return false;
        }
        if h.abs() > 0.001
            && !self.floor_patches.iter().any(|p| {
                (p.height - h).abs() < 0.001
                    && lo.xz().cmpge(p.min).all()
                    && hi.xz().cmple(p.max).all()
            })
        {
            return false;
        }
        self.floor_patches
            .iter()
            .all(|p| !p.overlaps(lo.xz(), hi.xz()) || (p.height - h).abs() < 0.001)
    }
}

impl IndoorManifest {
    pub fn floor_height(&self, p: Vec2) -> f32 {
        self.envelope.as_ref().map_or(0., |e| e.floor_height(p))
    }
    pub fn ceiling_height(&self, p: Vec2) -> f32 {
        self.envelope
            .as_ref()
            .map_or(self.room_size.y, |e| e.ceiling_height(self.room_size, p))
    }
    pub fn floor_support_clear(&self, lo: Vec3, hi: Vec3) -> bool {
        self.envelope
            .as_ref()
            .map_or(lo.y.abs() < 0.002, |e| e.support_clear(lo, hi))
    }
}
