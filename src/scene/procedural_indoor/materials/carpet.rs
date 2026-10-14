//! Millimetre pile is filtered into roughness; room-scale dye and installation
//! patterns survive distant views. CPU and one-shot GPU bake use the same field.
use super::{
    hash, periodic_noise,
    program::{MaterialRecipe, Texel},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarpetDesign {
    /// 0 heather, 1 striae, 2 broken lattice, 3 woven diamonds, 4 color blocks.
    pub pattern: u32,
    pub repeats: [u32; 2],
    pub palette: [[f32; 3]; 3],
    pub contrast: f32,
    pub heather: f32,
    pub pile: f32,
    pub line_width: f32,
    pub edge_variation: f32,
}
impl CarpetDesign {
    pub fn sample(seed: u64, base: [f32; 3], rug: bool) -> Self {
        let mut rng = stream(seed, 0x434152504554);
        let mut palette = [base; 3];
        let pale = rng.random_range(0.48..0.85);
        let warmth = rng.random_range(-0.04..0.08);
        palette[1] = [pale, pale * (1. - warmth), pale * (1. - warmth * 1.6)];
        let shade = rng.random_range(0.28..0.65);
        palette[2] = base.map(|c| c * shade);
        Self {
            pattern: rng.random_range(0..if rug { 5 } else { 3 }),
            repeats: [rng.random_range(2..=7), rng.random_range(2..=9)],
            palette,
            contrast: rng.random_range(if rug { 0.30..0.95 } else { 0.08..0.40 }),
            heather: rng.random_range(0.03..0.18),
            pile: rng.random_range(0.15..0.95),
            line_width: rng.random_range(0.07..0.23),
            edge_variation: rng.random_range(0.002..0.02),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.pattern > 4
            || self.repeats.iter().any(|n| !(1..=16).contains(n))
            || self
                .palette
                .iter()
                .flatten()
                .copied()
                .chain([
                    self.contrast,
                    self.heather,
                    self.pile,
                    self.line_width,
                    self.edge_variation,
                ])
                .any(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
        {
            return Err("invalid carpet dye/pile program".into());
        }
        Ok(())
    }
    pub(super) fn evaluate(&self, r: &MaterialRecipe, uv: [f32; 2]) -> Texel {
        let [u, v] = uv;
        let nx = self.repeats[0] as f32;
        let ny = self.repeats[1] as f32;
        let x = (u * nx).fract();
        let y = (v * ny).fract();
        let width = nx.max(ny) / r.map_size(1) as f32;
        let edge =
            (periodic_noise(u, v, 17, 11, r.seed.wrapping_add(271)) - 0.5) * self.edge_variation;
        let line = self.line_width + edge;
        let smooth = |t: f32| {
            let t = t.clamp(0., 1.);
            t * t * (3. - 2. * t)
        };
        let mask = match self.pattern {
            1 => smooth((line - (x - 0.5).abs()) / width + 0.5),
            2 => {
                let a = (x - 0.5).abs();
                let b = (y - 0.5).abs();
                smooth((line * 0.4 - a.min(b)) / width + 0.5)
            }
            3 => smooth(
                (line * 0.7 - ((x - 0.5).abs() + (y - 0.5).abs() - 0.55).abs()) / width + 0.5,
            ),
            4 => hash(
                (u * nx).floor() as u32,
                (v * ny).floor() as u32,
                r.seed.wrapping_add(251),
            ),
            _ => periodic_noise(u, v, 7, 9, r.seed.wrapping_add(241)),
        };
        let nap = periodic_noise(u, v, 173, 191, r.seed.wrapping_add(257)) - 0.5;
        let slub = periodic_noise(u, v, 11, 83, r.seed.wrapping_add(263)) - 0.5;
        let fleck = periodic_noise(u, v, 43, 47, r.seed.wrapping_add(269)) - 0.5;
        let dye = (mask * self.contrast).clamp(0., 1.);
        let tone = self.heather * (slub * 0.6 + fleck * 0.4) + nap * 0.035;
        let color = [0, 1, 2].map(|i| {
            let base = self.palette[0][i] * (1. - dye) + self.palette[1][i] * dye;
            let stipple = (fleck * 2.).max(0.) * self.heather;
            (base * (1. - stipple) + self.palette[2][i] * stipple + tone * base).clamp(0.01, 0.95)
        });
        let mut t = Texel {
            color,
            height: r.relief_m * (nap * 0.65 + fleck * self.pile * 0.35),
            roughness: (r.roughness + nap * 0.055 + fleck * 0.025).clamp(0.80, 1.),
            occlusion: 0.97 + (nap + 0.5) * 0.03,
        };
        if let Some(f) = &r.flooring {
            let before = t.color;
            f.apply(r, uv, &mut t);
            for (c, b) in t.color.iter_mut().zip(before) {
                *c = b + (*c - b) * f.tile_definition;
            }
            let cut = f.cut(r, uv);
            let seam = f.seam(r, &cut, 1) * f.tile_definition;
            t.color = t.color.map(|c| c * (1. - seam * 0.12));
            t.height -= seam * 0.00035;
        }
        t
    }
}
