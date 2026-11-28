//! Merchandise on low shop tables. Positions are in fixture-local metres.
use super::*;
use serde::Serialize;

mod garment;
mod pattern;
mod stand;
pub use pattern::{FoldedCut, FoldedGarment};
pub use stand::DisplayTableStand;
#[cfg(test)]
mod tests;

pub(crate) const FOLDED_APPAREL_VERSION: u32 = 310;
pub(crate) const SINGLE_BROWSING_FACE: u32 = 1 << 21;
/// This optional shop layout can be decided before any garments are built.
pub(crate) fn wants_opposite_face(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::ApparelDisplay
        && o.size.y < super::super::RAIL_MIN_HEIGHT
        && o.size.z >= 0.73
        && stream(o.seed, u64::from_le_bytes(*b"ShopSide")).random_bool(0.53)
}
const MAX_GRADE: f32 = 1.04;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FoldedLayer {
    pub base_m: Vec3,
    pub yaw_rad: f32,
    pub size_grade: f32,
}

#[derive(Debug, Serialize)]
pub struct FoldedPile {
    /// Column increases along local X and row along local Z.
    pub column: usize,
    pub row: usize,
    pub garment: FoldedGarment,
    pub layers: Vec<FoldedLayer>,
}

#[derive(Debug, Serialize)]
pub struct FoldedDisplay {
    pub deck_y_m: f32,
    pub frame: Surface,
    pub deck_surface: Surface,
    pub deck_thickness_m: f32,
    pub stand: DisplayTableStand,
    pub columns: usize,
    pub rows: usize,
    pub piles: Vec<FoldedPile>,
}

/// Recover the cuts, dye lots and supported placements without building meshes.
pub fn folded_apparel(o: &IndoorObject, generator_version: u32) -> Option<FoldedDisplay> {
    if generator_version < FOLDED_APPAREL_VERSION || o.kind != ObjectKind::ApparelDisplay {
        return None;
    }
    let Installation::Folded {
        piles,
        layers,
        frame,
    } = parameters(o)
    else {
        return None;
    };
    Some(FoldedDisplay::sample(o, piles, layers, frame))
}

impl FoldedDisplay {
    pub(in super::super) fn sample(
        o: &IndoorObject,
        bays: usize,
        max_layers: usize,
        frame: Surface,
    ) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"ShopFold"));
        let two_sided = o.variant & SINGLE_BROWSING_FACE == 0 && wants_opposite_face(o);
        let rows = if two_sided { 2 } else { 1 };
        let columns = if two_sided {
            (bays * 2).div_ceil(3)
        } else {
            bays
        };
        let cells = columns * rows;
        let cell = (o.size.xz() - Vec2::splat(0.055)) / Vec2::new(columns as f32, rows as f32);
        // Opposite browsing faces share the same merchandise budget.
        let capacity = (bays * max_layers / cells).max(1);
        let full = r.random_bool(0.46);
        let min_layers = if full { capacity.div_ceil(2) } else { 1 };
        let fill = r.random_range(if full { 0.86..0.99 } else { 0.58..0.91 });
        let anchor = r.random_range(0..cells);
        let department = FoldedCut::sample(&mut r);
        let share = r.random_range(0.32..0.81);
        let yaw_spread = r.random_range(0.045_f32..0.23);
        let shift = r.random_range(0.003_f32..0.011);
        let mut piles = Vec::with_capacity(cells);
        let mut highest = 0f32;
        for i in 0..cells {
            if i != anchor && !r.random_bool(fill) {
                continue;
            }
            let column = i % columns;
            let row = i / columns;
            let cut = if r.random_bool(share) {
                department
            } else {
                FoldedCut::sample(&mut r)
            };
            let heading = r.random_range(-yaw_spread..yaw_spread);
            let mut garment = FoldedGarment::sample(cut, &mut r);
            let footprint = garment.fit(
                cell - Vec2::splat(2. * shift + 0.009),
                heading.abs() + 0.038,
            );
            let slack = (cell - footprint) * 0.5 - Vec2::splat(shift);
            let center = Vec2::new(
                cell.x * (column as f32 + 0.5 - columns as f32 * 0.5),
                cell.y * (row as f32 + 0.5 - rows as f32 * 0.5),
            ) + slack
                * Vec2::new(r.random_range(-0.91..0.91), r.random_range(-0.91..0.91));
            let back = two_sided && row == 0;
            let yaw = heading + if back { PI } else { 0. };
            let count = r.random_range(min_layers..=capacity);
            let mut layers = Vec::with_capacity(count);
            let mut h = 0.;
            for j in 0..count {
                let size_grade = r.random_range(0.94..=MAX_GRADE);
                layers.push(FoldedLayer {
                    base_m: Vec3::new(
                        center.x + r.random_range(-shift..shift),
                        h,
                        center.y + r.random_range(-shift..shift),
                    ),
                    yaw_rad: yaw + r.random_range(-0.038..0.038),
                    size_grade,
                });
                h += garment.size_m.y * size_grade;
                if j + 1 == count {
                    highest = highest.max(h + garment.detail_rise_m * size_grade);
                }
            }
            piles.push(FoldedPile {
                column,
                row,
                garment,
                layers,
            });
        }
        let deck_y_m = o.size.y - highest;
        for layer in piles.iter_mut().flat_map(|pile| &mut pile.layers) {
            layer.base_m.y += deck_y_m;
        }
        let mut frame_rng = stream(o.seed, u64::from_le_bytes(*b"FoldDeck"));
        let deck_surface = if frame_rng.random_bool(0.48) {
            Surface::Wood
        } else {
            frame
        };
        let deck_thickness_m = frame_rng.random_range(0.024..0.047);
        let stand = DisplayTableStand::sample(o.size, frame, &mut frame_rng);
        Self {
            deck_surface,
            deck_thickness_m,
            stand,
            deck_y_m,
            frame,
            columns,
            rows,
            piles,
        }
    }

    pub(in super::super) fn build(&self, a: &mut Assembly, size: Vec3) {
        self.stand.build(a, size, self);
        for pile in &self.piles {
            if let Some((top, buried)) = pile.layers.split_last() {
                if !buried.is_empty() {
                    pile.garment.install(a, buried, false);
                }
                pile.garment.install(a, std::slice::from_ref(top), true);
            }
        }
    }
}
