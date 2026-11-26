//! Deployed freestanding spectator banks. Local +Z faces the playing area.
use super::*;
use crate::scene::procedural_indoor::materials::sport;

mod mesh;
#[cfg(test)]
mod tests;
pub(super) use mesh::build;

pub(crate) const VERSION: u32 = 290;
pub(super) const CLASS: &str = "bleacher";
const GUARD_HEIGHT: f32 = 1.02;
const NOMINAL_PITCH: f32 = 0.65;

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub enum SeatStock {
    Timber,
    Moulded,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BleacherBank {
    pub rows: usize,
    pub width: f32,
    pub pitch: f32,
    pub rise: f32,
    pub aisle_center: f32,
    pub aisle_width: f32,
    pub seat_depth: f32,
    pub seat_height: f32,
    pub module_pitch: f32,
    pub seat_stock: SeatStock,
    pub seat_slot: usize,
    pub deck_stock: Surface,
}

/// The geometry and envelope agree on the number of rows, even after serialization.
pub(crate) fn dimensions(width: f32, rows: usize, pitch: f32, rise: f32) -> Vec3 {
    Vec3::new(
        width,
        rows as f32 * rise + GUARD_HEIGHT,
        rows as f32 * pitch,
    )
}

fn tier_count(depth: f32) -> usize {
    ((depth / NOMINAL_PITCH).round() as usize).clamp(2, 5)
}
pub(crate) fn fits_stock(size: Vec3) -> bool {
    let rows = tier_count(size.z) as f32;
    (3.54..=8.31).contains(&size.x)
        && (0.604..=0.701).contains(&(size.z / rows))
        && (0.234..=0.301).contains(&((size.y - GUARD_HEIGHT) / rows))
}

pub(super) fn catalog_dimensions(r: &mut impl Rng) -> Vec3 {
    dimensions(
        r.random_range(3.55..8.30),
        r.random_range(2..=5),
        r.random_range(0.605..0.700),
        r.random_range(0.235..0.300),
    )
}

impl BleacherBank {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"Bleachrs"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"SeatStok"),
        );
        let rows = tier_count(o.size.z);
        let aisle_width = r.random_range(0.90..1.13);
        // Leave usable seating either side of an internal stair aisle.
        let eccentricity = ((o.size.x - aisle_width) * 0.5 - 0.92).clamp(0., 1.35);
        Self {
            rows,
            width: o.size.x,
            pitch: o.size.z / rows as f32,
            rise: (o.size.y - GUARD_HEIGHT) / rows as f32,
            aisle_center: eccentricity * r.random_range(-0.88..0.88),
            aisle_width,
            seat_depth: r.random_range(0.25..0.315),
            seat_height: r.random_range(0.166..0.207),
            module_pitch: r.random_range(0.454..0.535),
            seat_stock: if finish.random_bool(0.46) {
                SeatStock::Timber
            } else {
                SeatStock::Moulded
            },
            seat_slot: sport::pad_slot(&mut finish),
            deck_stock: if finish.random_bool(0.52) {
                Surface::Wood
            } else {
                Surface::Rubber
            },
        }
    }

    pub(super) fn spans(self) -> [(f32, f32); 2] {
        [
            (
                -self.width * 0.5 + 0.045,
                self.aisle_center - self.aisle_width * 0.5,
            ),
            (
                self.aisle_center + self.aisle_width * 0.5,
                self.width * 0.5 - 0.045,
            ),
        ]
    }
    pub fn seats(self) -> usize {
        self.rows
            * self
                .spans()
                .iter()
                .map(|&(l, r)| self.modules(l, r))
                .sum::<usize>()
    }
    fn modules(self, lo: f32, hi: f32) -> usize {
        ((hi - lo) / self.module_pitch).floor() as usize
    }
    fn front(self, i: usize) -> f32 {
        self.pitch * (self.rows as f32 * 0.5 - i as f32)
    }
    fn deck_height(self, i: usize) -> f32 {
        self.rise * (i + 1) as f32
    }
    fn seat_z(self, i: usize) -> f32 {
        self.front(i) - 0.045 - self.seat_depth * 0.5
    }

    /// Four treads form one tier: the deck spans the back; two wings and a
    /// half-rise aisle tread meet its front edge. Dimensions here drive both
    /// the mesh and proposal visibility so the stair opening remains usable.
    fn treads(self, i: usize, mut emit: impl FnMut(Vec3, Vec3)) {
        let y = self.deck_height(i);
        let front = self.front(i);
        let nose = front - self.pitch * 0.46;
        let left = self.aisle_center - self.aisle_width * 0.5;
        let right = self.aisle_center + self.aisle_width * 0.5;
        let half = self.width * 0.5;
        for (lo, hi, near, far, top) in [
            (-half, half, nose, front - self.pitch, y),
            (-half, left, front, nose, y),
            (right, half, front, nose, y),
            (left, right, front, nose, y - self.rise * 0.5),
        ] {
            emit(
                Vec3::new((lo + hi) * 0.5, top - 0.026, (near + far) * 0.5),
                Vec3::new(hi - lo, 0.052, near - far),
            );
        }
    }

    /// Camera proposals see the tiers and seats. The open guard and underframe
    /// stay permeable. The renderer and exported overlap use the full mesh.
    pub(crate) fn view_boxes(self, mut emit: impl FnMut(Vec3, Vec3)) {
        for i in 0..self.rows {
            let y = self.deck_height(i);
            self.treads(i, |center, size| emit(center, size * 0.5));
            for (lo, hi) in self.spans() {
                emit(
                    Vec3::new(
                        (lo + hi) * 0.5,
                        y + self.seat_height - 0.032,
                        self.seat_z(i),
                    ),
                    Vec3::new((hi - lo) * 0.5, 0.032, self.seat_depth * 0.5),
                );
            }
        }
    }
}
