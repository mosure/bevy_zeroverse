//! Metric cuts from the room's shared sheet stock. The material maps, including
//! their normal and roughness registration, stay shared across furniture.
use super::*;

pub(crate) const REGISTERED_SHEET_VERSION: u32 = 203;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SheetEdge {
    ExposedSlab,
    FormedFoil,
    BondedBand,
    TimberLip,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct SheetCut {
    pub face_origin_m: Vec2,
    pub back_origin_m: Vec2,
    pub band_origin_m: Vec2,
    /// Rotation from the tabletop's local XZ plane to the stock's UV axes.
    pub heading_rad: f32,
    pub edge: SheetEdge,
    /// Joined leaves either continue the print or use separate cuts of the same stock.
    pub match_leaves: bool,
}

impl SheetCut {
    pub(super) fn for_version(o: &IndoorObject, surface: Surface, version: u32) -> Self {
        let lot = o.finish_seed.unwrap_or(o.seed);
        let timber_lip =
            surface == Surface::Plastic && stream(lot, 0x4544_4745_4c49_5053).random_bool(0.36);
        if version < REGISTERED_SHEET_VERSION {
            let origin = if o.variant == SHARED_ROW_VARIANT {
                let mut r = stream(o.seed, u64::from_le_bytes(*b"SheetCut"));
                Vec2::new(r.random_range(0.0..2.4), r.random_range(0.0..2.4))
            } else {
                Vec2::ZERO
            };
            return Self {
                face_origin_m: origin,
                back_origin_m: origin,
                band_origin_m: Vec2::ZERO,
                heading_rad: 0.,
                edge: if timber_lip {
                    SheetEdge::TimberLip
                } else {
                    SheetEdge::BondedBand
                },
                match_leaves: true,
            };
        }

        let mut stock = stream(lot, u64::from_le_bytes(*b"SheetLay"));
        let edge = match surface {
            Surface::Plastic if timber_lip => SheetEdge::TimberLip,
            Surface::Plastic if stock.random_bool(0.33) => SheetEdge::FormedFoil,
            Surface::Plastic => SheetEdge::BondedBand,
            _ => SheetEdge::ExposedSlab,
        };
        let axis = if o.size.x >= o.size.z { 0. } else { -FRAC_PI_2 };
        let direction = if stock.random_bool(0.5) { PI } else { 0. };
        let heading = if surface == Surface::Plastic {
            // Roll goods are normally cut square, often lengthwise. Pieces of
            // one order retain that direction even when installed at different yaws.
            if stock.random_bool(0.76) {
                axis
            } else {
                axis + FRAC_PI_2
            }
        } else if stock.random_bool(0.58) {
            axis + stock.random_range(-0.55..0.55)
        } else {
            stock.random_range(0.0..PI)
        };
        let mut r = stream(o.seed, u64::from_le_bytes(*b"SheetCut"));
        let mut origin = || Vec2::new(r.random_range(0.0..2.4), r.random_range(0.0..2.4));
        let face_origin_m = origin();
        Self {
            face_origin_m,
            back_origin_m: if surface == Surface::Plastic {
                origin()
            } else {
                face_origin_m
            },
            band_origin_m: origin(),
            heading_rad: (heading + direction).rem_euclid(TAU),
            edge,
            match_leaves: stock.random_bool(if surface == Surface::Plastic {
                0.36
            } else {
                0.64
            }),
        }
    }

    pub(super) fn for_leaf(self, seed: u64, leaf: usize) -> Self {
        if leaf == 0 || self.match_leaves {
            return self;
        }
        let mut r = stream(seed, u64::from_le_bytes(*b"LeafCuts") ^ leaf as u64);
        let offset = Vec2::new(r.random_range(-1.2..1.2), r.random_range(-1.2..1.2));
        Self {
            face_origin_m: self.face_origin_m + offset,
            back_origin_m: self.back_origin_m + offset,
            band_origin_m: self.band_origin_m + offset,
            ..self
        }
    }

    pub(super) fn edge_uvs(
        &self,
        g: &mut Geometry,
        outline: &[Vec2],
        height: f32,
        profile: (f32, f32),
        tf: &Transform,
        rotation: Mat2,
    ) {
        let (bevel, inset) = profile;
        let (origin, backing) = (self.face_origin_m, self.band_origin_m);
        match self.edge {
            SheetEdge::TimberLip | SheetEdge::BondedBand => {
                g.slab_edge_with_uv(outline, height, profile, tf, |_, along, rings| {
                    rings.map(|(y, _)| {
                        backing
                            + if self.edge == SheetEdge::TimberLip {
                                Vec2::new(y, -along)
                            } else {
                                Vec2::new(along, y)
                            }
                    })
                });
            }
            SheetEdge::FormedFoil | SheetEdge::ExposedSlab => {
                g.slab_edge_with_uv(outline, height, profile, tf, |i, _, _| {
                    let n = outline.len();
                    let p = outline[i];
                    // CCW outline in XZ. The miter gives both adjoining faces
                    // the same corner chart and one metre per metre of vertical
                    // reveal on straight runs. Bound the miter at sharp tips.
                    let prev = -(p - outline[(i + n - 1) % n]).normalize_or_zero().perp();
                    let next = -(outline[(i + 1) % n] - p).normalize_or_zero().perp();
                    let miter = (prev + next) / (1. + prev.dot(next)).max(0.25);
                    let cap = p * inset;
                    let run = ((p - cap).dot(miter) / miter.length_squared().max(1e-6)).max(0.);
                    let slope = run.hypot(bevel);
                    let upper = p + miter * (slope - run);
                    let side = height - 2. * bevel;
                    [
                        cap + miter * (2. * slope + side),
                        upper + miter * side,
                        upper,
                        cap,
                    ]
                    .map(|v| rotation * (v + tf.translation.xz()) + origin)
                });
            }
        }
    }
}

/// Reproducible physical cut for a current-generator table. Timber has its own
/// veneer and plank program. Coordinates are in metres before material tiling.
pub fn sheet_cut(o: &IndoorObject, p: &TableProgram) -> Option<SheetCut> {
    (p.top_surface != Surface::Wood)
        .then(|| SheetCut::for_version(o, p.top_surface, REGISTERED_SHEET_VERSION))
}

#[cfg(test)]
mod tests;
