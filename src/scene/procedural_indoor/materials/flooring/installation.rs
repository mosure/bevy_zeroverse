//! Closed, periodic installations. Piece IDs, grain coordinates and metric joints
//! come from one tessellation; they cannot disagree at material boundaries.
use super::{Cut, FloorFinish, FloorLayout, MaterialRecipe};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Installation {
    /// Herringbone plank length in units of its width.
    pub plank_ratio: u32,
    /// Rise across a chevron lane, in course-height units.
    pub chevron_rise: f32,
}

pub(super) fn sample(f: &mut FloorFinish, seed: u64, style: u32, period: f32) {
    // Carpets keep their existing supported GPU installation. Hard substrates
    // already use CPU synthesis; no extra atlas, texture or draw call is added.
    if style == 1 {
        return;
    }
    let mut rng = stream(seed, 0x494e5354414c4c);
    let choice: f32 = rng.random();
    f.layout = if style == 0 {
        if choice < 0.23 {
            FloorLayout::Herringbone
        } else if choice < 0.40 {
            FloorLayout::Chevron
        } else {
            f.layout
        }
    } else if choice < 0.23 {
        FloorLayout::Hexagonal
    } else if choice < 0.40 {
        FloorLayout::Seamless
    } else if choice < 0.55 {
        FloorLayout::Herringbone
    } else {
        f.layout
    };
    let craft = Installation {
        plank_ratio: rng.random_range(2..=5),
        chevron_rise: rng.random_range(0.65..2.40),
    };
    match f.layout {
        FloorLayout::Herringbone => {
            let unit = 2 * craft.plank_ratio;
            let width = rng.random_range(if style == 0 { 0.10..0.20 } else { 0.13..0.30 });
            let blocks = (period / width / unit as f32)
                .round()
                .clamp(1., (48 / unit) as f32) as u32;
            f.cells = [blocks * unit; 2];
        }
        FloorLayout::Chevron => {
            // An even number of opposing lanes closes the V at the atlas edge.
            let lanes = (period / rng.random_range(0.38..0.85) / 2.)
                .round()
                .clamp(1., 12.) as u32
                * 2;
            f.cells = [
                lanes,
                (period / rng.random_range(0.10..0.23))
                    .round()
                    .clamp(2., 48.) as u32,
            ];
        }
        FloorLayout::Hexagonal => {
            let nx = (period / rng.random_range(0.22..0.65))
                .round()
                .clamp(4., 40.) as u32;
            f.cells = [nx, (nx as f32 / 3_f32.sqrt()).round().max(2.) as u32];
        }
        FloorLayout::Seamless => f.cells = [2; 2],
        _ => {}
    }
    f.craft = Some(craft);
}

impl Installation {
    pub(super) fn validate(&self, f: &FloorFinish) -> Result<(), String> {
        if !(2..=5).contains(&self.plank_ratio)
            || !self.chevron_rise.is_finite()
            || !(0.25..=3.).contains(&self.chevron_rise)
            || f.layout == FloorLayout::Herringbone
                && (f.cells[0] != f.cells[1] || !f.cells[0].is_multiple_of(2 * self.plank_ratio))
            || f.layout == FloorLayout::Chevron && !f.cells[0].is_multiple_of(2)
        {
            return Err("invalid floor tessellation".into());
        }
        Ok(())
    }

    pub(super) fn cut(&self, f: &FloorFinish, r: &MaterialRecipe, uv: [f32; 2]) -> Option<Cut> {
        let [nx, ny] = f.cells;
        let [x, y] = [
            uv[0].rem_euclid(1.) * nx as f32,
            uv[1].rem_euclid(1.) * ny as f32,
        ];
        let a = r.period_m / nx as f32;
        let b = r.period_m / ny as f32;
        let wrap = |i: i32, n: u32| i.rem_euclid(n as i32) as u32;
        let rectangle = |id, uv: [f32; 2], metric: [f32; 2]| Cut {
            id,
            uv,
            distance_m: (uv[0].min(1. - uv[0]) * metric[0])
                .min(uv[1].min(1. - uv[1]) * metric[1])
                .max(0.),
        };
        Some(match f.layout {
            FloorLayout::Herringbone => {
                let (ix, iy) = (x.floor() as i32, y.floor() as i32);
                let n = self.plank_ratio as i32;
                let k = (ix - iy).rem_euclid(2 * n);
                if k < n {
                    rectangle(
                        [wrap(ix - k, nx), wrap(iy, ny)],
                        [y.fract(), (k as f32 + x.fract()) / n as f32],
                        [a, a * n as f32],
                    )
                } else {
                    let along = 2 * n - 1 - k;
                    rectangle(
                        [wrap(ix, nx), wrap(iy - along, ny) + ny],
                        [x.fract(), (along as f32 + y.fract()) / n as f32],
                        [a, a * n as f32],
                    )
                }
            }
            FloorLayout::Chevron => {
                let ix = x.floor() as u32;
                let along = if ix.is_multiple_of(2) {
                    x.fract()
                } else {
                    1. - x.fract()
                };
                let course = y - self.chevron_rise * along;
                // The seam distance is measured perpendicular to the inclined
                // cut, rather than in sheared UV units.
                let across = b / (1. + (self.chevron_rise * b / a).powi(2)).sqrt();
                rectangle(
                    [ix, wrap(course.floor() as i32, ny)],
                    [course.rem_euclid(1.), along],
                    [across, a],
                )
            }
            FloorLayout::Hexagonal => {
                // Two rectangular sublattices form a hexagonal Voronoi tiling.
                // Nearest-site selection and six bisectors are closed form.
                let c0 = [x.round(), y.round()];
                let c1 = [(x - 0.5).round() + 0.5, (y - 0.5).round() + 0.5];
                let distance = |c: [f32; 2]| ((x - c[0]) * a).powi(2) + ((y - c[1]) * b).powi(2);
                let second = distance(c1) < distance(c0);
                let c = if second { c1 } else { c0 };
                let dx = (x - c[0]) * a;
                let dy = (y - c[1]) * b;
                let diagonal = (a * a + b * b).sqrt();
                let distance_m = (a * 0.5 - dx.abs())
                    .min(b * 0.5 - dy.abs())
                    .min(diagonal * 0.25 - (dx * a + dy * b).abs() / diagonal)
                    .min(diagonal * 0.25 - (dx * a - dy * b).abs() / diagonal)
                    .max(0.);
                let half_height = (diagonal * diagonal / (4. * b)).min(b * 0.5);
                Cut {
                    id: [
                        wrap(c[0].floor() as i32, nx) + u32::from(second) * nx,
                        wrap(c[1].floor() as i32, ny),
                    ],
                    uv: [
                        (0.5 + dx / a).clamp(0., 1.),
                        (0.5 + dy / (2. * half_height)).clamp(0., 1.),
                    ],
                    distance_m,
                }
            }
            FloorLayout::Seamless => Cut {
                id: [0; 2],
                uv: uv.map(|v| v.rem_euclid(1.)),
                distance_m: r.period_m,
            },
            _ => return None,
        })
    }
}
