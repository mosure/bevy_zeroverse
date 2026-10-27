//! Floor installation and cut-piece variation, independent of the substrate.
//! One bounded atlas describes many pieces; no extra geometry or draw calls.
use super::{hash, periodic_noise, program::MaterialRecipe};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloorLayout {
    Straight,
    BrokenBond,
    Basket,
    Herringbone,
    Chevron,
    Hexagonal,
    Seamless,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorFinish {
    pub layout: FloorLayout,
    /// Optional construction parameters; old recorded installations are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub craft: Option<installation::Installation>,
    /// Product/installation variation. Omitted recipes retain their original cuts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stock: Option<stock::FloorStock>,
    /// Installations close exactly at atlas boundaries. Basket cells are square.
    pub cells: [u32; 2],
    pub basket_strips: u32,
    pub piece_tone: f32,
    pub piece_warmth: f32,
    pub piece_roughness: f32,
    pub mottling: f32,
    pub accent_fraction: f32,
    pub accent_tint: [f32; 3],
    /// Broadloom (zero) through clearly articulated modular carpet tiles.
    pub tile_definition: f32,
}

pub(super) struct Cut {
    pub id: [u32; 2],
    pub uv: [f32; 2],
    pub distance_m: f32,
}

impl FloorFinish {
    pub fn sample(seed: u64, style: u32, period: f32) -> Self {
        let mut rng = stream(seed, 0x464c4f4f52494e47);
        let basket = style == 0 && rng.random_bool(0.30);
        let layout = if basket {
            FloorLayout::Basket
        } else if rng.random_bool(if style == 0 { 0.85 } else { 0.40 }) {
            FloorLayout::BrokenBond
        } else {
            FloorLayout::Straight
        };
        let width = if style == 0 && !basket {
            rng.random_range(0.12..0.24)
        } else {
            rng.random_range(0.40..0.85)
        };
        let nx = (period / width).round().clamp(2., 48.) as u32;
        let ny = if basket {
            nx
        } else {
            let length = if style == 0 {
                rng.random_range(0.85..1.9)
            } else {
                rng.random_range(0.40..0.95)
            };
            (period / length).round().clamp(2., 24.) as u32
        };
        let tint = rng.random_range(0.60_f32..0.91);
        let warm = rng.random_range(-0.08..0.12);
        let mut result = Self {
            layout,
            craft: None,
            stock: None,
            cells: [nx, ny],
            basket_strips: rng.random_range(3..=5),
            piece_tone: rng.random_range(0.12..if style == 0 { 0.48 } else { 0.30 }),
            piece_warmth: rng.random_range(0.012..0.075),
            piece_roughness: rng.random_range(0.018..0.10),
            mottling: rng.random_range(0.015..0.07),
            accent_fraction: if style != 0 && rng.random_bool(0.28) {
                rng.random_range(0.06..0.24)
            } else {
                0.
            },
            accent_tint: [tint, tint * (1. - warm), tint * (1. - warm * 1.5)].map(|v| v.min(0.97)),
            tile_definition: if style == 1 && rng.random_bool(0.30) {
                0.
            } else {
                rng.random_range(0.35..1.)
            },
        };
        installation::sample(&mut result, seed, style, period);
        result
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(stock) = &self.stock {
            stock.validate()?;
        }
        if let Some(c) = &self.craft {
            c.validate(self)?;
        } else if matches!(
            self.layout,
            FloorLayout::Herringbone
                | FloorLayout::Chevron
                | FloorLayout::Hexagonal
                | FloorLayout::Seamless
        ) {
            return Err("floor construction requires installation parameters".into());
        }
        if self.cells.iter().any(|n| !(2..=48).contains(n))
            || !(3..=5).contains(&self.basket_strips)
            || self.layout == FloorLayout::Basket && self.cells[0] != self.cells[1]
            || [
                self.piece_tone,
                self.piece_warmth,
                self.piece_roughness,
                self.mottling,
                self.accent_fraction,
                self.tile_definition,
            ]
            .into_iter()
            .chain(self.accent_tint)
            .any(|v| !v.is_finite() || !(0.0..=1.).contains(&v))
        {
            return Err("invalid floor installation program".into());
        }
        Ok(())
    }

    pub(super) fn cut(&self, r: &MaterialRecipe, uv: [f32; 2]) -> Cut {
        if let Some(cut) = self.craft.as_ref().and_then(|c| c.cut(self, r, uv)) {
            return cut;
        }
        if let Some(stock) = &self.stock {
            if matches!(self.layout, FloorLayout::Straight | FloorLayout::BrokenBond)
                && stock.cut_jitter != [0.; 2]
            {
                return stock.cut(self, r, uv);
            }
        }
        let [nx, ny] = self.cells;
        let u = uv[0].rem_euclid(1.) * nx as f32;
        let x = u.floor() as u32 % nx;
        // Independent row offsets, including fractional board ends, not a
        // single alternating seam repeated along every lane in the room.
        let stagger = if self.layout == FloorLayout::BrokenBond {
            hash(x, 7, r.seed.wrapping_add(211))
        } else {
            0.
        };
        let v = uv[1].rem_euclid(1.) * ny as f32 + stagger;
        let y = v.floor() as u32 % ny;
        let mut local = [u.fract(), v.fract()];
        let mut id = [x, y];
        let mut metric = [r.period_m / nx as f32, r.period_m / ny as f32];
        if self.layout == FloorLayout::Basket {
            if (x + y).is_multiple_of(2) {
                local.swap(0, 1);
            }
            let strips = self.basket_strips as f32;
            let index = (local[0] * strips).floor() as u32;
            local[0] = (local[0] * strips).fract();
            id[0] = x * self.basket_strips + index;
            metric[0] /= strips;
        }
        Cut {
            id,
            uv: local,
            distance_m: (local[0].min(1. - local[0]) * metric[0])
                .min(local[1].min(1. - local[1]) * metric[1]),
        }
    }

    pub(super) fn seam(&self, r: &MaterialRecipe, cut: &Cut, style: u32) -> f32 {
        if self.layout == FloorLayout::Seamless {
            return 0.;
        }
        let width = r.joint_width + r.period_m / r.map_size(style) as f32;
        (1. - cut.distance_m / width).clamp(0., 1.) * r.joint_width / width
    }

    pub(super) fn apply(&self, r: &MaterialRecipe, uv: [f32; 2], t: &mut super::program::Texel) {
        if self.layout == FloorLayout::Seamless {
            return;
        }
        let cut = self.cut(r, uv);
        let [x, y] = cut.id;
        let [mx, my] = self.stock.as_ref().map_or([3, 3], |s| s.mottle_cells);
        let gain = 1.
            + (hash(x, y, r.seed.wrapping_add(223)) - 0.5) * self.piece_tone
            + (periodic_noise(uv[0], uv[1], mx, my, r.seed.wrapping_add(227)) - 0.5)
                * self.mottling;
        let warm = (hash(x, y, r.seed.wrapping_add(229)) - 0.5) * self.piece_warmth;
        let accent = hash(x, y, r.seed.wrapping_add(233)) < self.accent_fraction;
        for (i, c) in t.color.iter_mut().enumerate() {
            // Small reflectance differences between cuts of the same species
            // or production lot, rather than unrelated checkerboard colors.
            *c = (*c
                * gain
                * [1. + warm, 1., 1. - warm][i]
                * if accent { self.accent_tint[i] } else { 1. })
            .clamp(0., 0.97);
        }
        t.roughness = (t.roughness
            + (hash(x, y, r.seed.wrapping_add(239)) - 0.5) * self.piece_roughness)
            .clamp(0.12, 1.);
    }
}

pub mod installation;
pub(super) mod sampling;
pub mod stock;

#[cfg(test)]
mod diversity_tests;
#[cfg(test)]
mod tests;
