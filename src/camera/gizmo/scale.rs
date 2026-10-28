//! Presentation-only bounds shared by editor, schematic, CLI and URL settings.
use serde::{Deserialize, Deserializer};

pub const MIN: f32 = 0.1;
pub const MAX: f32 = 4.;

pub const fn default() -> f32 {
    1.
}

pub fn validate(value: f32) -> Result<f32, String> {
    if value.is_finite() && (MIN..=MAX).contains(&value) {
        Ok(value)
    } else {
        Err("camera gizmo scale must be finite and in 0.1..=4 (1 = default size)".into())
    }
}

pub fn parse(value: &str) -> Result<f32, String> {
    validate(value.parse::<f32>().map_err(|e| e.to_string())?)
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    validate(f32::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}
