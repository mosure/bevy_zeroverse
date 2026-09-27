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
}

impl FurnishingField {
    pub fn sample(seed: u64, zone: usize, activity: IndoorLayout) -> Self {
        let mut rng = stream(seed, 270 + zone as u64);
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
        }
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
        {
            return Err("invalid continuous furnishing field".into());
        }
        Ok(())
    }
}
