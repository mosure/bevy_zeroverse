//! Seeded furniture/support contract for shared lecture desks and dining rows.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableRowSeating {
    pub lecture_tables: bool,
    pub cafeteria_rows: bool,
    pub lecture_depth_metres: f32,
    pub dining_depth_metres: f32,
    /// Clear passage behind a seated row, excluding the chair footprint.
    pub aisle_metres: f32,
    /// Fraction of each seating bank occupied by its shared table.
    pub length_fraction: f32,
    pub chair_yaw_radians: f32,
}
impl Default for TableRowSeating {
    fn default() -> Self {
        Self {
            lecture_tables: false,
            cafeteria_rows: false,
            lecture_depth_metres: 0.56,
            dining_depth_metres: 0.82,
            aisle_metres: 0.75,
            length_fraction: 0.9,
            chair_yaw_radians: 0.16,
        }
    }
}
impl TableRowSeating {
    pub(super) fn sample(seed: u64, zone: usize) -> Self {
        let mut r = stream(seed, 0x5441424c45524f57 + zone as u64);
        Self {
            lecture_tables: r.random_bool(0.55),
            cafeteria_rows: r.random_bool(0.60),
            lecture_depth_metres: r.random_range(0.50..0.64),
            dining_depth_metres: r.random_range(0.72..0.94),
            aisle_metres: r.random_range(0.65..0.95),
            length_fraction: r.random_range(0.76..0.98),
            chair_yaw_radians: r.random_range(0.09..0.26),
        }
    }
}

#[cfg(test)]
mod tests;
