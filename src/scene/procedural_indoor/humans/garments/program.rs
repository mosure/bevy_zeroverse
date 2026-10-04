//! Sewn cuts and fit in rest-body metres, independent of colour and phenotype.
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GarmentProgram {
    pub neckline_width: f32,
    pub neckline_depth: f32,
    pub neckline_roundness: f32,
    pub drape: f32,
    pub section_roundness: f32,
    /// Knee-to-ankle coverage; zero is a knee-length cut, one a full trouser.
    pub trouser_coverage: f32,
    pub trouser_ease: f32,
    /// Zero follows the calf; one hangs straighter from the knee.
    pub leg_straightness: f32,
    /// Ankle width relative to the fitted calf (taper through modest flare).
    pub hem_width: f32,
    pub collar_width: f32,
    pub placket_width: f32,
    pub pocket_width: f32,
    pub pocket_height: f32,
    pub pocket: bool,
}
impl Default for GarmentProgram {
    fn default() -> Self {
        Self {
            neckline_width: 0.075,
            neckline_depth: 0.012,
            neckline_roundness: 2.0,
            drape: 0.55,
            section_roundness: 2.05,
            trouser_coverage: 0.96,
            trouser_ease: 0.009,
            leg_straightness: 0.5,
            hem_width: 0.92,
            collar_width: 0.055,
            placket_width: 0.014,
            pocket_width: 0.095,
            pocket_height: 0.10,
            pocket: false,
        }
    }
}
impl GarmentProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = super::super::stream(seed, 0x5345574e435554);
        Self {
            neckline_width: rng.random_range(0.058..0.115),
            neckline_depth: rng.random_range(0.008..0.115),
            neckline_roundness: rng.random_range(1.05..3.4),
            drape: rng.random_range(0.18..0.92),
            section_roundness: rng.random_range(1.9..2.2),
            trouser_coverage: if rng.random_bool(0.88) {
                rng.random_range(0.82..0.99)
            } else {
                rng.random_range(0.08..0.55)
            },
            trouser_ease: rng.random_range(0.005..0.023),
            collar_width: rng.random_range(0.036..0.072),
            placket_width: rng.random_range(0.009..0.022),
            pocket_width: rng.random_range(0.065..0.105),
            pocket_height: rng.random_range(0.065..0.11),
            pocket: rng.random_bool(0.38),
            leg_straightness: rng.random_range(0.15..0.98),
            hem_width: rng.random_range(0.70..1.35),
        }
    }
    pub(super) fn neckline(&self, p: Vec3, neck: f32, depth_center: f32) -> f32 {
        let width = (1.0 - p.x.abs() / self.neckline_width).max(0.0);
        let front = 1.0 - super::fit::smooth(depth_center - 0.045, depth_center + 0.025, p.z);
        p.y - neck + self.neckline_depth * width.powf(self.neckline_roundness) * front
    }
}
