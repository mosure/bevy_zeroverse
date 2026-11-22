//! Stock and veneer cuts in metres on the actual tabletop. Cuts change the
//! tangent/UV charts for all PBR maps together without adding a second surface.
use super::*;
use crate::scene::procedural_indoor::{envelope::polygon, geometry::Geometry};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Stock {
    Continuous,
    BookmatchedVeneer,
    EdgeGlued,
}

#[derive(Clone, Copy)]
pub(crate) struct StockProgram {
    pub stock: Stock,
    pub pitch_m: Option<f32>,
}
impl StockProgram {
    pub(crate) fn sample(seed: u64) -> Self {
        // Keep construction consistent across manufactured leaves. Each leaf can
        // use a different cut from the same species and the same cached PBR stock.
        let mut stock_rng = stream(seed, 0x5449_4d42_5354_4f43);
        let stock = match stock_rng.random::<f32>() {
            t if t < 0.30 => Stock::Continuous,
            t if t < 0.59 => Stock::BookmatchedVeneer,
            _ => Stock::EdgeGlued,
        };
        let pitch = match stock {
            Stock::Continuous => None,
            Stock::BookmatchedVeneer => Some(stock_rng.random_range(0.21..0.48)),
            Stock::EdgeGlued => Some(stock_rng.random_range(0.095..0.235)),
        };
        Self {
            stock,
            pitch_m: pitch,
        }
    }

    pub(super) fn build(
        self,
        g: &mut Geometry,
        o: &IndoorObject,
        p: &TableProgram,
        outline: &[Vec2],
        tf: Transform,
        leaf: usize,
    ) {
        let across = if o.size.x > o.size.z {
            Vec2::Y
        } else {
            Vec2::X
        };
        let along = Vec2::new(-across.y, across.x);
        let mut rng = stream(o.seed, 0x5449_4d42_4348_4152 ^ leaf as u64);
        let start = g.positions.len();
        g.profile_slab_with_top(
            outline,
            p.top_thickness,
            p.edge_bevel,
            tf,
            |g, outline, inset, y, tf| {
                let phase = Vec2::new(rng.random_range(0.0..2.7), rng.random_range(0.0..3.9));
                let Some(pitch) = self.pitch_m else {
                    g.slab_cap(outline, inset, Vec2::ZERO, y, tf, |v| {
                        Vec2::new(v.dot(across), v.dot(along)) + phase
                    });
                    return;
                };
                let cap: Vec<_> = outline.iter().map(|p| *p * inset).collect();
                let lo = cap
                    .iter()
                    .map(|v| v.dot(across))
                    .fold(f32::INFINITY, f32::min);
                let hi = cap
                    .iter()
                    .map(|v| v.dot(across))
                    .fold(f32::NEG_INFINITY, f32::max);
                let count = ((hi - lo) / pitch).round().clamp(1., 32.) as usize;
                let mut previous = lo;
                let mut remaining = cap.clone();
                let mut offset = phase;
                let mut skew = 0.;
                for i in 0..count {
                    let end = if i + 1 == count {
                        hi
                    } else {
                        let proposed = lo
                            + (hi - lo) * (i as f32 + 1. + rng.random_range(-0.14..0.14))
                                / count as f32;
                        // A seam grazing an existing polygon vertex can leave an
                        // almost zero-length edge after world-space quantization.
                        cap.iter()
                            .map(|v| v.dot(across))
                            .find(|s| (*s - proposed).abs() < 0.00045)
                            .unwrap_or(proposed)
                    };
                    // A cut only subdivides the still uncharted part of the cap.
                    // The final strip already has both boundaries.
                    let strip = if i + 1 == count {
                        std::mem::take(&mut remaining)
                    } else {
                        let strip = polygon::clip(&remaining, -across, -end);
                        remaining = polygon::clip(&remaining, across, end);
                        strip
                    };
                    if strip.len() >= 3 {
                        let center = strip.iter().sum::<Vec2>() / strip.len() as f32;
                        if self.stock == Stock::EdgeGlued || i.is_multiple_of(2) {
                            offset =
                                Vec2::new(rng.random_range(0.0..2.7), rng.random_range(0.0..3.9));
                            skew = rng.random_range(-0.025..0.025);
                        }
                        g.slab_cap(&strip, 1., center, y, tf, |v| {
                            let length = v.dot(along);
                            // Consecutive veneer sheets open like a book and share
                            // a growth-ring intersection at their common seam.
                            let width =
                                if self.stock == Stock::BookmatchedVeneer && i.is_multiple_of(2) {
                                    end - v.dot(across)
                                } else {
                                    v.dot(across) - previous
                                };
                            Vec2::new(width + skew * length, length) + offset
                        });
                    }
                    previous = end;
                }
            },
        );
        for i in start..g.positions.len() {
            let ny = g.normals[i][1];
            if ny < -0.99999 {
                let v = Vec2::from_array(g.uvs[i]);
                g.uvs[i] = [v.dot(across), v.dot(along)];
            } else if ny < 0.99999 {
                // The material's fibre axis is V. Edge banding follows the
                // perimeter at any table orientation, including the curved ends.
                let [distance, height] = g.uvs[i];
                g.uvs[i] = [height, -distance];
            }
        }
    }
}
