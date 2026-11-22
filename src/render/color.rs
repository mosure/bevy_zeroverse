//! RGB tone curves and the transfer function for eight-bit sRGB previews.
use bevy::{prelude::Reflect, render::view::Tonemapping};
use bevy_args::ValueEnum;
use serde::{Deserialize, Serialize};

/// RGB camera tone curve. Auto preserves the original SDR look and chooses
/// exposed, display-linear light when the destination supports HDR.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum RgbTonemapping {
    #[default]
    Auto,
    None,
    Linear,
    Reinhard,
    ReinhardLuminance,
    AcesFitted,
    #[serde(rename = "agx")]
    #[value(name = "agx")]
    AgX,
    SomewhatBoringDisplayTransform,
    #[serde(rename = "tony-mcmapface")]
    #[value(name = "tony-mcmapface")]
    TonyMcMapface,
    BlenderFilmic,
    KhronosPbrNeutral,
}

impl RgbTonemapping {
    /// Keep the display-linear floating-point result of the selected curve.
    /// Even SDR-fitted operators can produce out-of-range saturated channels.
    pub fn preserves_hdr(self, hdr: bool) -> bool {
        hdr || matches!(self, Self::Linear | Self::None)
    }

    pub fn resolve(self, hdr: bool) -> Tonemapping {
        match self {
            Self::Auto if hdr => Tonemapping::Linear,
            Self::Auto | Self::TonyMcMapface => Tonemapping::TonyMcMapface,
            Self::None => Tonemapping::None,
            Self::Linear => Tonemapping::Linear,
            Self::Reinhard => Tonemapping::Reinhard,
            Self::ReinhardLuminance => Tonemapping::ReinhardLuminance,
            Self::AcesFitted => Tonemapping::AcesFitted,
            Self::AgX => Tonemapping::AgX,
            Self::SomewhatBoringDisplayTransform => Tonemapping::SomewhatBoringDisplayTransform,
            Self::BlenderFilmic => Tonemapping::BlenderFilmic,
            Self::KhronosPbrNeutral => Tonemapping::KhronosPbrNeutral,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorEncoding {
    /// Historical exporter behavior, retained for existing scene modes.
    #[default]
    Legacy,
    TonemappedLinear,
    Srgb,
    /// Unbounded linear RGB. Floating-point capture preserves values above one.
    LinearHdr,
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
    fn hdr_storage_is_independent_of_the_chosen_operator() {
        for &curve in RgbTonemapping::value_variants() {
            assert!(curve.preserves_hdr(true));
            assert_eq!(
                curve.preserves_hdr(false),
                matches!(curve, RgbTonemapping::None | RgbTonemapping::Linear)
            );
        }
    }

    #[test]
    fn fixed_srgb_transfer_preserves_absolute_exposure_and_neutral_colors() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!((linear_to_srgb(0.18) - 0.4613561).abs() < 1e-6);
        assert!((linear_to_srgb(0.0031308) - 0.04044994).abs() < 1e-6);
        assert!(linear_to_srgb(0.36) > linear_to_srgb(0.18));
    }
}
