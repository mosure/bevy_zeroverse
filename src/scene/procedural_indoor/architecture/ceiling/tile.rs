//! Mineral-fiber board and its compatible exposed suspension grid.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileEdge {
    SquareLayIn,
    SquareTegular,
    BeveledTegular,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileFacing {
    FineFiber,
    Pinhole,
    Fissured,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CeilingTile {
    pub edge: TileEdge,
    pub facing: TileFacing,
    pub flange_width_m: f32,
    pub thickness_m: f32,
    /// Distance from the bearing shoulder to the visible lower face.
    pub drop_m: f32,
    /// Each board edge clears the vertical web by this distance.
    pub clearance_m: f32,
}

impl CeilingTile {
    pub(super) fn sample(room_seed: u64, finish: u8) -> Self {
        // Rafts sharing a finish in a zone are assembled from the same stock.
        let mut r = stream(
            room_seed,
            u64::from_le_bytes(*b"TileSpec") ^ u64::from(finish),
        );
        let edge = match r.random_range(0..10) {
            0..=4 => TileEdge::SquareLayIn,
            5..=7 => TileEdge::SquareTegular,
            _ => TileEdge::BeveledTegular,
        };
        let thickness_m = r.random_range(0.015..0.025);
        Self {
            edge,
            facing: [
                TileFacing::FineFiber,
                TileFacing::Pinhole,
                TileFacing::Fissured,
            ][r.random_range(0..3)],
            flange_width_m: if r.random_bool(0.65) {
                r.random_range(0.022..0.025)
            } else {
                r.random_range(0.014..0.016)
            },
            thickness_m,
            drop_m: if edge == TileEdge::SquareLayIn {
                0.
            } else {
                r.random_range(0.004_f32..0.012).min(thickness_m - 0.009)
            },
            clearance_m: r.random_range(0.001..0.0028),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.014..=0.025).contains(&self.flange_width_m)
            && (0.015..=0.025).contains(&self.thickness_m)
            && (0.001..=0.0028).contains(&self.clearance_m)
            && match self.edge {
                TileEdge::SquareLayIn => self.drop_m == 0.,
                _ => {
                    (0.004..=0.012).contains(&self.drop_m)
                        && self.thickness_m - self.drop_m >= 0.0089
                }
            }
    }
}

pub(super) fn vary_cut(
    g: &mut crate::scene::procedural_indoor::geometry::Geometry,
    start: usize,
    cuts: &mut impl Rng,
) {
    // Each piece comes from a different cut of the same stock. Quarter turns
    // and translation preserve the physical scale and tangent handedness.
    let quarter = cuts.random_range(0..4);
    let phase = Vec2::new(cuts.random_range(0.0..2.), cuts.random_range(0.0..2.));
    for uv in &mut g.uvs[start..] {
        let value = Vec2::from_array(*uv);
        let rotated = match quarter {
            0 => value,
            1 => value.perp(),
            2 => -value,
            _ => -value.perp(),
        };
        *uv = (rotated + phase).to_array();
    }
}
