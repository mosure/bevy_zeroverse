//! A continuous spatial field for furniture groups, independent of their assets.
use super::super::layout::{stream, IndoorLayout};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FurnishingField {
    pub offset: Vec2,
    pub stagger: f32,
    pub curvature: f32,
    pub fan_radians: f32,
    pub jitter: f32,
    pub occupancy_gradient: Vec2,
    pub opposing_probability: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub banks: Option<WorkstationBanks>,
}

/// Locally coordinated desk headings, lateral offsets and optional paired backs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorkstationBanks {
    pub turn_radians: f32,
    pub lateral_pitch_fraction: f32,
    pub phase_radians: f32,
    pub row_frequency: f32,
    pub back_gap_m: Option<f32>,
}

pub(crate) struct DeskStation {
    pub center: Vec2,
    pub turn: f32,
    pub occupancy: f32,
    pub opposite: Option<bool>,
    pub paired: bool,
}

impl FurnishingField {
    /// An opposed pair shares its rear gap and needs chair clearance only at
    /// the two outside edges. A narrow room can fit one pair even when it
    /// cannot fit two rows of separately spaced desks.
    pub(crate) fn desk_rows(
        &self,
        ordinary: usize,
        depth: f32,
        pitch: f32,
        desk_depth: f32,
        teaching: bool,
    ) -> usize {
        let Some(gap) = self
            .banks
            .and_then(|b| b.back_gap_m)
            .filter(|_| !teaching && !ordinary.is_multiple_of(2))
        else {
            return ordinary;
        };
        // The furnishing envelope already reserves a perimeter margin. Allow
        // a chair centre 0.62 m past either desk and 0.36 m behind it.
        let occupied = (ordinary - 1) as f32 * pitch + 2. * (desk_depth + 0.98) + gap;
        ordinary + usize::from(depth >= occupied)
    }
    pub fn sample(seed: u64, zone: usize, activity: IndoorLayout) -> Self {
        let mut rng = stream(seed, 270 + zone as u64);
        let mut banks = stream(seed, 0x4445_534b_4241_4e4b ^ zone as u64);
        Self {
            offset: Vec2::new(rng.random_range(-0.07..0.07), rng.random_range(-0.07..0.07)),
            stagger: rng.random_range(-0.65..0.65),
            curvature: rng.random_range(-0.8..0.8),
            fan_radians: rng.random_range(-0.7..0.7),
            jitter: rng.random_range(0.0..0.13),
            occupancy_gradient: Vec2::new(rng.random_range(-0.7..0.7), rng.random_range(-0.7..0.7)),
            opposing_probability: if activity == IndoorLayout::Training {
                0.0
            } else {
                rng.random_range(0.0..1.0)
            },
            banks: Some(WorkstationBanks {
                turn_radians: if banks.random_bool(0.18) {
                    0.
                } else {
                    banks.random_range(0.08..0.48)
                },
                lateral_pitch_fraction: banks.random_range(0.02..0.18),
                phase_radians: banks.random_range(0.0..std::f32::consts::TAU),
                row_frequency: banks.random_range(0.75..2.35),
                back_gap_m: banks
                    .random_bool(0.52)
                    .then(|| banks.random_range(0.11..0.32)),
            }),
        }
    }

    pub(crate) fn desk_station(
        &self,
        cell: [usize; 2],
        count: [usize; 2],
        pitch: Vec2,
        depth: f32,
        teaching: bool,
    ) -> DeskStation {
        let (center, turn, occupancy) = self.station(cell, count, pitch);
        let mut s = DeskStation {
            center,
            turn,
            occupancy,
            opposite: None,
            paired: false,
        };
        let Some(banks) = self.banks else { return s };
        let row = cell[1];
        let first = row / 2 * 2;
        let gap = banks
            .back_gap_m
            .filter(|_| !teaching && first + 1 < count[1]);
        let bank = if gap.is_some() { first / 2 } else { row };
        let phase = bank as f32 * banks.row_frequency + banks.phase_radians;
        let (bend, shift) = phase.sin_cos();
        s.turn += banks.turn_radians * bend * if teaching { 0.3 } else { 1. };
        s.center.x += shift * banks.lateral_pitch_fraction * pitch.x;
        if let Some(gap) = gap {
            // Desks meet back to back around one joint. The chairs remain on
            // the outside and the regular inter-bank pitch becomes an aisle.
            let side = if row == first { -1. } else { 1. };
            s.center -= side * 0.5 * Vec2::new(self.stagger * pitch.x, pitch.y);
            // A row facing a single direction reserves extra space at the
            // front. An opposed bank has identical chair clearance at each end.
            s.center.y += 0.45;
            let (sin, cos) = s.turn.sin_cos();
            s.center += Vec2::new(sin, cos) * (side * 0.5 * (depth + gap));
            s.opposite = Some(row == first);
            s.paired = true;
        } else if teaching {
            s.opposite = Some(false);
        }
        s
    }

    /// Coordinates are local to the zone. Chairs use this same rotation, so
    /// fan-shaped and opposing workstations retain their interaction geometry.
    pub fn station(&self, cell: [usize; 2], count: [usize; 2], pitch: Vec2) -> (Vec2, f32, f32) {
        let [col, row] = cell;
        let [cols, rows] = count;
        let centered = Vec2::new(
            col as f32 - (cols - 1) as f32 * 0.5,
            row as f32 - (rows - 1) as f32 * 0.5,
        );
        let uv = Vec2::new(
            if cols > 1 {
                centered.x * 2.0 / (cols - 1) as f32
            } else {
                0.0
            },
            if rows > 1 {
                centered.y * 2.0 / (rows - 1) as f32
            } else {
                0.0
            },
        );
        let point = centered * pitch
            + Vec2::new(
                (row as f32 % 2.0 - 0.5) * self.stagger * pitch.x,
                self.curvature * uv.x * uv.x * pitch.y * 0.4 - 0.45,
            );
        (
            point,
            self.fan_radians * uv.x,
            (1.0 + self.occupancy_gradient.dot(uv)).clamp(0.25, 1.75),
        )
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.offset.is_finite()
            || self.offset.abs().max_element() > 0.1
            || !self.occupancy_gradient.is_finite()
            || self.occupancy_gradient.abs().max_element() > 1.0
            || !self.stagger.is_finite()
            || self.stagger.abs() > 1.0
            || !self.curvature.is_finite()
            || self.curvature.abs() > 1.0
            || !self.fan_radians.is_finite()
            || self.fan_radians.abs() > 1.0
            || !(0.0..=0.2).contains(&self.jitter)
            || !(0.0..=1.0).contains(&self.opposing_probability)
            || self.banks.is_some_and(|b| {
                !(0.0..=0.5).contains(&b.turn_radians)
                    || !(0.0..=0.20).contains(&b.lateral_pitch_fraction)
                    || !(0.0..=std::f32::consts::TAU).contains(&b.phase_radians)
                    || !(0.5..=2.5).contains(&b.row_frequency)
                    || b.back_gap_m.is_some_and(|g| !(0.10..=0.35).contains(&g))
            })
        {
            return Err("invalid continuous furnishing field".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_workstations_share_a_joint_and_leave_aisles_and_teaching_rows_open() {
        let pitch = Vec2::new(2.3, 3.2);
        let depth = 0.82;
        let count = [3, 5];
        let mut pairs = 0;
        for seed in 0..64 {
            let field = FurnishingField::sample(seed, 1, IndoorLayout::OpenOffice);
            field.validate().unwrap();
            let Some(gap) = field.banks.unwrap().back_gap_m else {
                continue;
            };
            for rows in [1, 3, 5] {
                let occupied = (rows - 1) as f32 * pitch.y + 2. * (depth + 0.98) + gap;
                assert_eq!(
                    field.desk_rows(rows, occupied + 0.01, pitch.y, depth, false),
                    rows + 1
                );
                assert_eq!(
                    field.desk_rows(rows, occupied - 0.01, pitch.y, depth, false),
                    rows
                );
                assert_eq!(
                    field.desk_rows(rows, occupied + 0.01, pitch.y, depth, true),
                    rows
                );
            }
            pairs += 1;
            for col in 0..count[0] {
                for row in [0, 2] {
                    let a = field.desk_station([col, row], count, pitch, depth, false);
                    let b = field.desk_station([col, row + 1], count, pitch, depth, false);
                    let normal = Vec2::new(a.turn.sin(), a.turn.cos());
                    assert_eq!((a.opposite, b.opposite), (Some(true), Some(false)));
                    assert_eq!(a.turn, b.turn);
                    assert!((b.center - a.center - normal * (gap + depth)).length() < 2e-6);
                }
                assert!(
                    !field
                        .desk_station([col, 4], count, pitch, depth, false)
                        .paired
                );
                for row in 0..count[1] {
                    let s = field.desk_station([col, row], count, pitch, depth, true);
                    assert!(!s.paired && s.opposite == Some(false));
                }
            }
        }
        assert!((16..50).contains(&pairs), "{pairs}");
    }

    #[test]
    fn archived_fields_replay_their_placement_and_invalid_bank_values_are_rejected() {
        let mut f = FurnishingField::sample(31, 2, IndoorLayout::OpenOffice);
        let mut archived = serde_json::to_value(&f).unwrap();
        archived.as_object_mut().unwrap().remove("banks");
        let legacy: FurnishingField = serde_json::from_value(archived).unwrap();
        let pitch = Vec2::new(2., 3.);
        let s = legacy.desk_station([1, 1], [3, 4], pitch, 0.75, false);
        assert_eq!(
            (s.center, s.turn, s.occupancy),
            legacy.station([1, 1], [3, 4], pitch)
        );
        assert_eq!(s.opposite, None);
        f.banks.as_mut().unwrap().back_gap_m = Some(f32::NAN);
        assert!(f.validate().is_err());
    }
}
