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
        let [u, v] = uv;
        let spray_n = frequency(r.period_m, self.spray_spacing_m);
        let spray = cellular(uv, [spray_n, spray_n], r.seed.wrapping_add(503));
        let aa = spray_n as f32 / r.map_size(0) as f32;
        let splat = disk(spray.radius, 0.20 + spray.dye * 0.28, aa);
        // Rounded droplets become broad, flattened islands when knocked down.
        let plateau = smooth(splat * 1.4).min(1. - c.knockdown * 0.55);
        let roller_n = frequency(r.period_m, self.roller_spacing_m);
        let roller = periodic_noise(
            u,
            v,
            roller_n,
            (roller_n as f32 / self.roller_stretch).round().max(1.) as u32,
            r.seed.wrapping_add(509),
        ) - 0.5;
        let fine_n = frequency(r.period_m, self.orange_peel_m);
        let film = periodic_noise(u, v, fine_n, fine_n, r.seed.wrapping_add(521)) - 0.5;
        let brush_n = frequency(r.period_m, self.brush_spacing_m);
        let brush = periodic_noise(u, v, brush_n, 5, r.seed.wrapping_add(523)) - 0.5;
        // Unresolved brush strokes contribute scattering, not aliasing relief.
        let resolved_brush = (r.map_size(0) as f32 / brush_n as f32 * 0.30).min(1.);
        let trowel_n = frequency(r.period_m, self.trowel_scale_m);
        let passes = deposit(uv, [trowel_n, trowel_n], 0.12, r.seed.wrapping_add(541));
        let lap = smooth((passes - 0.35) * 1.5);
        let repair = smooth((passes - 0.65) * 6.) * self.repair_mix;
        let holes = disk(spray.radius, 0.08, aa) * smooth((c.pinholes - spray.dye) * 10.);
        let texture = (plateau - 0.28) * c.texture_mix * (1. - repair);
        Texel {
            color: tint(
                r.color,
                1. + (passes - 0.5) * c.pigment_variation - holes * 0.035,
            ),
            height: r.relief_m * (texture * 0.80 + (lap - 0.5) * c.trowel * 0.22 - holes * 0.40)
                + 0.000025 * (roller * c.roller + film * self.orange_peel)
                + 0.000016 * brush * self.brush * resolved_brush,
            roughness: (r.roughness
                + film * 0.025
                + roller * c.roller * 0.035
                + (lap - 0.5) * self.lap_variation
                + holes * 0.04
                - repair * c.gloss * 0.10)
                .clamp(0.16, 0.99),
            occlusion: (1. - holes * 0.025).max(0.97),
        }
    }
}
