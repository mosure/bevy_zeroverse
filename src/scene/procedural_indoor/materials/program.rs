//! Bounded, metric material programs. A recipe describes correlated colour,
//! roughness and relief layers; seed changes do not merely recolour one bitmap.
use super::{periodic_noise, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod prepared;
pub(super) use prepared::{PreparedMaterial, PreparedMineral, PreparedWood};
#[cfg(test)]
pub(super) mod replay_helpers;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialRecipe {
    /// Resolved construction-prior provenance; numeric substrate fields remain
    /// authoritative when replaying an archived recipe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<super::construction::ConstructionSample>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manufactured: Option<super::manufactured::ManufacturedFinish>,
    /// Appearance intervention inherited by regenerated finish structures.
    /// The stored base relief/contrast already include this multiplier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flooring: Option<super::flooring::FloorFinish>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub carpet: Option<super::carpet::CarpetDesign>,
    #[serde(default)]
    pub layers: Option<super::layers::FinishLayers>,
    #[serde(default)]
    pub textile: Option<super::textile::TextileRecipe>,
    #[serde(default)]
    pub leather: Option<super::leather::LeatherRecipe>,
    #[serde(default)]
    pub mineral: Option<super::mineral::MineralRecipe>,
    #[serde(default)]
    pub coating: Option<super::coating::CoatingRecipe>,
    #[serde(default)]
    pub wood: Option<super::timber::WoodFinish>,
    #[serde(default)]
    pub leaf: Option<super::botanical::LeafRecipe>,
    #[serde(default)]
    pub cuticle: Option<super::botanical::CuticleRecipe>,
    pub surface: Surface,
    pub seed: u64,
    pub color: [f32; 3],
    pub roughness: f32,
    /// Longitudinal repeat. Timber's transverse cut width is given by `period_uv`.
    pub period_m: f32,
    pub relief_m: f32,
    pub grain_frequency: f32,
    pub cross_frequency: f32,
    pub warp: f32,
    pub contrast: f32,
    pub weathering: f32,
    pub mineral_mix: f32,
    pub weave_mix: f32,
    pub phase: [f32; 2],
    pub panel_count: [u32; 2],
    pub joint_width: f32,
}
/// Compatibility alias for the authoritative binding catalog.
pub const SURFACES: [Surface; Surface::COUNT] = Surface::ALL;
/// sRGB reflectance (absolute or relative per recipe), metric relief, roughness, AO.
pub(super) struct Texel {
    pub color: [f32; 3],
    pub height: f32,
    pub roughness: f32,
    pub occlusion: f32,
}

mod evaluation;
mod palette;
mod sampling;
pub use palette::PaletteProgram;
pub(crate) use sampling::floor_finish;
#[cfg(test)]
pub(super) use sampling::{floor_finish_base, sample_base};
pub use sampling::{sample, sample_with_floor};
#[cfg(test)]
mod tests;
