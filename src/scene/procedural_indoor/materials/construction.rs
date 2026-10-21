//! Correlated construction priors, resolved once into the existing substrate
//! programs. Four distinct process regions per category have continuous blends,
//! metric scale and finish variation. Reuses existing atlases and substrate
//! shaders; the priors do not add per-fragment material layers.
// Retain independent within-process parameters from the original seeded
// substrate. Correlated process priors must not collapse dozens of continuous
// controls onto only the five-dimensional construction mixture.
macro_rules! retain_variation {
    ($target:expr, $source:expr, $($field:ident),+ $(,)?) => {{
        let source = $source;
        $( $target.$field = $target.$field * 0.80 + source.$field * 0.20; )+
    }};
}
mod coatings;
mod fibers;
mod mineral;
mod organic;
#[cfg(test)]
mod tests;

use super::{program::MaterialRecipe, Surface};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Sampling provenance. The recipe's resolved substrate parameters are
/// authoritative on replay; use `apply` to resolve an authored construction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstructionSample {
    pub weights: [f32; 4],
    /// Multiplier on physical feature sizes, not texture resolution.
    pub scale: f32,
    pub variation: f32,
}
impl ConstructionSample {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4352414654505249);
        let anchor = rng.random_range(0..4);
        // Continuous concentration retains recognizable process poles while
        // filling the intervening finishes. Fixed tiny off-pole weights would
        // collapse wax/polish distributions into four narrow islands.
        let spread = 0.04 + 0.80 * rng.random::<f32>().powi(2);
        let mut weights = std::array::from_fn(|_| rng.random_range(0.0..spread));
        weights[anchor] = rng.random_range(0.65..1.);
        let sum: f32 = weights.iter().sum();
        Self {
            weights: weights.map(|x| x / sum),
            scale: rng.random_range(0.65_f32.ln()..1.65_f32.ln()).exp(),
            variation: rng.random_range(0.10..0.95),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self
            .weights
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
            || (self.weights.iter().sum::<f32>() - 1.).abs() > 1e-5
            || !(0.5..=2.).contains(&self.scale)
            || !(0.0..=1.).contains(&self.variation)
        {
            return Err("construction requires normalized finite weights, scale in [0.5,2], variation in [0,1]".into());
        }
        Ok(())
    }
    pub fn dominant(&self) -> usize {
        (0..4)
            .max_by(|&a, &b| self.weights[a].total_cmp(&self.weights[b]))
            .unwrap()
    }
    fn mix(&self, values: [f32; 4]) -> f32 {
        self.weights
            .into_iter()
            .zip(values)
            .map(|(w, v)| w * v)
            .sum()
    }
    fn count(&self, values: [f32; 4], bounds: [u32; 2]) -> u32 {
        (self.mix(values) / self.scale)
            .round()
            .clamp(bounds[0] as f32, bounds[1] as f32) as u32
    }
}

/// Process names describe the four poles, not four fixed material presets.
pub fn labels(surface: Surface, floor: u32) -> Option<[&'static str; 4]> {
    Some(match surface {
        Surface::Fabric | Surface::FabricAlt => {
            ["linen weave", "twill", "chenille pile", "loop knit"]
        }
        Surface::Leather => ["aniline", "pebbled", "nubuck", "sealed hide"],
        Surface::Wood | Surface::WoodEdge => {
            ["fine veneer", "open pore", "limed timber", "dark lacquer"]
        }
        Surface::Floor if floor == 0 => [
            "fine timber",
            "open-grain timber",
            "washed timber",
            "sealed timber",
        ],
        Surface::Floor if floor == 1 => [
            "fine heather",
            "striated pile",
            "modular weave",
            "dense felt",
        ],
        Surface::Floor => [
            "honed limestone",
            "exposed aggregate",
            "veined stone",
            "polished mineral",
        ],
        Surface::Paint | Surface::Ceiling => [
            "smooth rolled",
            "orange peel",
            "knockdown",
            "trowelled plaster",
        ],
        Surface::Accent => [
            "mineral wash",
            "long fiber",
            "woven covering",
            "printed covering",
        ],
        Surface::Ceramic => [
            "porcelain",
            "speckled stoneware",
            "reactive glaze",
            "crazed glaze",
        ],
        Surface::Concrete => [
            "smooth cast",
            "board formed",
            "trowelled",
            "ground aggregate",
        ],
        Surface::Metal => [
            "fine coating",
            "peened coating",
            "directional satin",
            "coarse powder coat",
        ],
        Surface::Chrome => [
            "mirror polish",
            "satin polish",
            "brushed alloy",
            "bead blasted alloy",
        ],
        Surface::Plastic => [
            "gloss molded",
            "pebbled polymer",
            "soft touch",
            "satin molded",
        ],
        Surface::Rubber => [
            "smooth elastomer",
            "directional grip",
            "coarse molded",
            "matte foam",
        ],
        Surface::Paper => [
            "smooth paper",
            "coated stock",
            "matte card",
            "fibrous stock",
        ],
        Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => [
            "waxy blade",
            "matte blade",
            "pale variegation",
            "pronounced venation",
        ],
        Surface::Succulent => [
            "matte cuticle",
            "wax cuticle",
            "glaucous bloom",
            "stomatal texture",
        ],
        Surface::Soil => [
            "organic soil",
            "gritty substrate",
            "sandy substrate",
            "clay-rich substrate",
        ],
        Surface::Terracotta => [
            "fine earthenware",
            "grogged clay",
            "burnished clay",
            "coarse earthenware",
        ],
        Surface::Bark => [
            "fine texture",
            "coarse furrows",
            "shallow furrows",
            "rough bark",
        ],
        _ => return None,
    })
}

/// Recipe-aware labels distinguish whole-rug pile from furniture upholstery.
pub fn recipe_labels(recipe: &MaterialRecipe, floor: u32) -> Option<[&'static str; 4]> {
    if recipe.carpet.as_ref().is_some_and(|c| c.rug.is_some()) {
        Some(["short heather pile", "cut pile", "woven pile", "dense felt"])
    } else {
        labels(recipe.surface, floor)
    }
}

/// Resolve an authored blend without changing the recipe seed, geometry or
/// allocation budget. Invalid requests leave the original recipe untouched.
pub fn apply(
    recipe: &mut MaterialRecipe,
    sample: ConstructionSample,
    floor: u32,
) -> Result<(), String> {
    sample.validate()?;
    if floor > 2 || !has_substrate(recipe, floor) {
        return Err(
            "construction requires a supported resolved substrate and matching floor style 0..2"
                .into(),
        );
    }
    let mut candidate = recipe.clone();
    resolve(&mut candidate, &sample, floor);
    candidate.construction = Some(sample);
    candidate.validate()?;
    *recipe = candidate;
    Ok(())
}

fn has_substrate(r: &MaterialRecipe, floor: u32) -> bool {
    match r.surface {
        Surface::Fabric | Surface::FabricAlt => r.carpet.is_some() || r.textile.is_some(),
        Surface::Leather => r.leather.is_some(),
        Surface::Wood | Surface::WoodEdge => r.wood.is_some(),
        Surface::Floor => match floor {
            0 => r.wood.is_some() && r.carpet.is_none(),
            1 => r.carpet.is_some(),
            _ => r.mineral.is_some() && r.carpet.is_none(),
        },
        Surface::Paint | Surface::Accent | Surface::Ceiling => r.coating.is_some(),
        Surface::Ceramic => r.coating.as_ref().is_some_and(|c| c.glaze.is_some()),
        Surface::Concrete => r.mineral.as_ref().is_some_and(|m| m.casting.is_some()),
        Surface::Soil | Surface::Terracotta => r.mineral.is_some(),
        Surface::Metal | Surface::Chrome | Surface::Plastic | Surface::Rubber => {
            r.manufactured.is_some()
        }
        Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated => r.leaf.is_some(),
        Surface::Succulent => r.cuticle.is_some(),
        Surface::Paper | Surface::Bark => true,
        _ => false,
    }
}

pub(super) fn sample(recipe: &mut MaterialRecipe, floor: u32) {
    if labels(recipe.surface, floor).is_some() {
        // A piece's end grain and face grain share finishing stock, including
        // lazily regenerated finish slots, while their cut seeds stay distinct.
        let seed = if matches!(recipe.surface, Surface::Wood | Surface::WoodEdge) {
            recipe.wood.as_ref().map_or(recipe.seed, |w| w.seed)
        } else {
            recipe.seed
        };
        let sample = ConstructionSample::sample(seed);
        resolve(recipe, &sample, floor);
        recipe.construction = Some(sample);
    }
}

fn resolve(r: &mut MaterialRecipe, p: &ConstructionSample, floor: u32) {
    if r.carpet.is_some() {
        fibers::carpet(r, p);
    } else {
        match r.surface {
            Surface::Fabric | Surface::FabricAlt | Surface::Leather => fibers::apply(r, p),
            Surface::Paint
            | Surface::Accent
            | Surface::Ceiling
            | Surface::Ceramic
            | Surface::Metal
            | Surface::Chrome
            | Surface::Plastic
            | Surface::Rubber
            | Surface::Paper => coatings::apply(r, p),
            Surface::Concrete | Surface::Terracotta | Surface::Soil => mineral::apply(r, p, floor),
            Surface::Floor if floor == 1 => fibers::carpet(r, p),
            Surface::Floor if floor == 2 => mineral::apply(r, p, floor),
            _ => organic::apply(r, p),
        }
    }
    // All construction programs replace the absolute relief. Apply a recorded
    // appearance intervention once, including authored overrides and variants.
    r.relief_m *= r.detail_scale.unwrap_or(1.);
}
