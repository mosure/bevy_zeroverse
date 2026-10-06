//! Application-scale wall finishes with a fine paint film over plaster relief.
use super::{
    coating::CoatingRecipe,
    field::*,
    periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod prepared;
pub(super) use prepared::PreparedPaint;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaintApplication {
    pub spray_spacing_m: f32,
    pub roller_spacing_m: f32,
    pub roller_stretch: f32,
    pub orange_peel_m: f32,
    pub orange_peel: f32,
    pub brush_spacing_m: f32,
    pub brush: f32,
    pub trowel_scale_m: f32,
    pub lap_variation: f32,
    pub repair_mix: f32,
}
impl PaintApplication {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x5041494e54);
        Self {
            spray_spacing_m: rng.random_range(0.0025..0.009),
            roller_spacing_m: rng.random_range(0.0018..0.0055),
            roller_stretch: rng.random_range(1.4..4.0),
            orange_peel_m: rng.random_range(0.0007..0.0020),
            orange_peel: rng.random_range(0.0_f32..1.).powi(2),
            brush_spacing_m: rng.random_range(0.0006..0.0018),
            brush: rng.random_range(0.0_f32..1.).powi(3) * 0.75,
            trowel_scale_m: rng.random_range(0.018..0.075),
            lap_variation: rng.random_range(0.008..0.065),
            repair_mix: rng.random_range(0.0_f32..1.).powi(3) * 0.22,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if [
            self.spray_spacing_m,
            self.roller_spacing_m,
            self.orange_peel_m,
            self.brush_spacing_m,
        ]
        .iter()
        .any(|v| !v.is_finite() || !(0.0003..=0.020).contains(v))
            || !self.roller_stretch.is_finite()
            || !(1.0..=6.).contains(&self.roller_stretch)
            || !self.trowel_scale_m.is_finite()
            || !(0.008..=0.15).contains(&self.trowel_scale_m)
            || [
                self.orange_peel,
                self.brush,
                self.lap_variation,
                self.repair_mix,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid paint/plaster application program".into());
        }
        Ok(())
    }
    pub(super) fn evaluate(&self, c: &CoatingRecipe, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        self.prepare(r).evaluate(self, c, r, uv)
    }
    pub(super) fn prepare(&self, r: &MaterialRecipe) -> PreparedPaint {
        PreparedPaint::new(self, r)
    }
}

#[cfg(test)]
mod replay_tests;
