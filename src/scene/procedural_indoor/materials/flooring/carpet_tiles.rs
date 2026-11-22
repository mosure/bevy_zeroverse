//! Coordinates on the carpet roll from which an installed tile was cut.
use super::{hash, stream, FloorFinish, FloorLayout};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PileDirection {
    Parallel,
    OpposingCourses,
    QuarterTurn,
    NonDirectional,
}

impl PileDirection {
    pub(in super::super) fn gpu_code(self) -> u32 {
        match self {
            Self::Parallel => 1,
            Self::OpposingCourses => 2,
            Self::QuarterTurn => 3,
            Self::NonDirectional => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarpetTiles {
    pub direction: PileDirection,
    /// Maximum difference between two cut origins, in fractions of the dye repeat atlas.
    pub registration: f32,
}

impl CarpetTiles {
    pub(super) fn install(floor: &mut FloorFinish, seed: u64, directional_dye: f32) {
        if floor.tile_definition == 0. {
            return;
        }
        let mut rng = stream(seed, 0x4341525054494c45);
        let even_courses = floor.cells[0].is_multiple_of(2);
        let direction = if even_courses && floor.cells[0] == floor.cells[1] && rng.random_bool(0.63)
        {
            // A checkerboard of 90-degree turns requires square stock and
            // uninterrupted cross joints. Even counts also close across the atlas wrap.
            floor.layout = FloorLayout::Straight;
            PileDirection::QuarterTurn
        } else if even_courses && rng.random_bool(0.31) {
            PileDirection::OpposingCourses
        } else if rng.random_bool(0.50 - 0.42 * f64::from(directional_dye)) {
            // Flecked and mottled products tolerate random orientations more
            // often than deliberately striped yarn.
            PileDirection::NonDirectional
        } else {
            PileDirection::Parallel
        };
        floor.carpet_tiles = Some(Self {
            direction,
            registration: rng.random_range(0.08_f32..1.).powf(1.35),
        });
    }

    pub(super) fn validate(self, f: &FloorFinish) -> Result<(), String> {
        if !(0.0..=1.).contains(&self.registration)
            || f.tile_definition == 0.
            || !matches!(f.layout, FloorLayout::Straight | FloorLayout::BrokenBond)
            || self.direction == PileDirection::OpposingCourses && !f.cells[0].is_multiple_of(2)
            || self.direction == PileDirection::QuarterTurn
                && !(square_stock(f)
                    && f.cells[0].is_multiple_of(2)
                    && f.layout == FloorLayout::Straight)
        {
            return Err("invalid modular carpet installation".into());
        }
        Ok(())
    }

    pub(in super::super) fn yarn_uv(
        self,
        uv: [f32; 2],
        id: [u32; 2],
        f: &FloorFinish,
        seed: u64,
    ) -> [f32; 2] {
        let [x, y] = id;
        let turns = match self.direction {
            PileDirection::Parallel => 0,
            PileDirection::OpposingCourses => (x & 1) * 2,
            PileDirection::QuarterTurn => (x + y) & 1,
            PileDirection::NonDirectional => {
                let pick = ((hash(x, y, seed.wrapping_add(0x43540001)) * 4.) as u32).min(3);
                if square_stock(f) {
                    pick
                } else {
                    pick & 2
                }
            }
        };
        let [u, v] = uv;
        // Rotate in the metric atlas frame. Unitizing each individual tile
        // would stretch the yarn on planks and change its spatial frequency.
        let rotated = match turns {
            0 => [u, v],
            1 => [-v, u],
            2 => [-u, -v],
            _ => [v, -u],
        };
        std::array::from_fn(|i| {
            let origin = (hash(x, y, seed.wrapping_add([0x43540002, 0x43540003][i])) - 0.5)
                * self.registration;
            // A binary-grid origin keeps CPU and GPU on the same side of dye
            // mask cell boundaries, including for negative coordinates.
            (rotated[i] + (origin * 4096.).round_ties_even() / 4096.).rem_euclid(1.)
        })
    }
}

fn square_stock(f: &FloorFinish) -> bool {
    f.cells[0] == f.cells[1] && f.stock.as_ref().is_none_or(|s| s.cut_jitter == [0.; 2])
}

#[cfg(test)]
mod tests;
