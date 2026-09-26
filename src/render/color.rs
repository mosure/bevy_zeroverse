//! The float render target contains tone-mapped **linear** RGB. Encoding it for
//! display needs the sRGB transfer function, not another tone map or min/max fit.
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorEncoding {
    /// Historical exporter behavior, retained for existing scene modes.
    #[default]
    Legacy,
    TonemappedLinear,
    Srgb,
}

pub fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_srgb_transfer_preserves_absolute_exposure_and_neutral_colors() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!((linear_to_srgb(0.18) - 0.4613561).abs() < 1e-6);
        assert!((linear_to_srgb(0.0031308) - 0.04044994).abs() < 1e-6);
        assert!(linear_to_srgb(0.36) > linear_to_srgb(0.18));
    }
}
