//! White points for room illumination. Bevy uses linear sRGB for transport.
//! The source spectrum is reduced to the CIE 1931 standard observer's tristimulus.
use super::{kelvin_rgb, tinted_white};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IlluminantModel {
    /// Archives without an illuminant model used an interpolation in encoded sRGB.
    #[default]
    LegacySrgb,
    /// Planckian lamp/direct-beam white points and CIE D-series diffuse daylight.
    Cie1931,
}

impl IlluminantModel {
    pub(crate) fn is_legacy(&self) -> bool {
        *self == Self::LegacySrgb
    }

    /// Encoded light chromaticity for a lamp specified in kelvin and the
    /// dimensionless tint recorded in `SourcePower` or a ceiling circuit.
    pub fn lamp_srgb(self, kelvin: f32, tint: f32) -> Vec3 {
        match self {
            Self::LegacySrgb => tinted_white(kelvin, tint),
            // The existing dimensionless lamp/batch tint spans roughly ±0.05.
            // A positive Duv moves toward green and a negative Duv toward magenta.
            Self::Cie1931 => {
                encoded_chromaticity(planckian_rgb(kelvin, 0.10 * tint.clamp(-0.2, 0.2)))
            }
        }
    }

    pub(crate) fn untinted_srgb(self, kelvin: f32) -> Vec3 {
        match self {
            Self::LegacySrgb => kelvin_rgb(kelvin),
            Self::Cie1931 => encoded_chromaticity(planckian_rgb(kelvin, 0.)),
        }
    }

    /// Linear RGB chromaticity before scaling to the measured sky luminance.
    pub fn diffuse_sky_linear(self, kelvin: f32) -> Vec3 {
        match self {
            Self::LegacySrgb => {
                let rgb = kelvin_rgb(kelvin);
                let c = Color::srgb(rgb.x, rgb.y, rgb.z).to_linear();
                Vec3::new(c.red, c.green, c.blue)
            }
            Self::Cie1931 => daylight_rgb(kelvin),
        }
    }
}

/// Krystek (1985), doi:10.1002/col.5080100109. The two rational fits return
/// CIE 1960 u,v. Differentiate in kilokelvin to offset perpendicular to the
/// locus without coupling the sampled lamp temperature to its tint.
fn planckian_rgb(kelvin: f32, duv: f32) -> Vec3 {
    let t = kelvin.clamp(1800., 10_000.) * 0.001;
    let offset = Vec2::new(0.860_117_73, 0.317_398_73);
    let linear = Vec2::new(0.154_118_25, 0.042_280_626);
    let quadratic = Vec2::new(0.128_641_22, 0.042_048_17);
    let divisor_linear = Vec2::new(0.842_420_2, -0.028_974_181);
    let divisor_quadratic = Vec2::new(0.708_145_14, 0.161_456_05);
    let divisor = Vec2::ONE + t * (divisor_linear + t * divisor_quadratic);
    let mut uv = (offset + t * (linear + t * quadratic)) / divisor;
    if duv != 0. {
        let tangent = (linear + 2. * t * quadratic
            - uv * (divisor_linear + 2. * t * divisor_quadratic))
            / divisor;
        uv += duv * Vec2::new(tangent.y, -tangent.x).normalize();
    }
    xyz_to_linear_rgb(1.5 * uv.x / uv.y, (2. - 0.5 * uv.x) / uv.y - 5.)
}

/// CIE D-series daylight locus (CIE 015), defined from 4000 K. Diffuse
/// skies sample this interval. Low-angle direct sun follows its own warm CCT.
fn daylight_rgb(kelvin: f32) -> Vec3 {
    let inverse = 1000. / kelvin.clamp(4000., 25_000.);
    let x = if kelvin <= 7000. {
        0.244_063 + inverse * (0.099_11 + inverse * (2.9678 - 4.6070 * inverse))
    } else {
        0.237_040 + inverse * (0.247_48 + inverse * (1.9018 - 2.0064 * inverse))
    };
    let y = x * (2.870 - 3.000 * x) - 0.275;
    xyz_to_linear_rgb(x / y, (1. - x - y) / y)
}

/// XYZ with Y=1 to linear sRGB (D65). Warm sources at the extreme of the
/// sampled range can lie just outside the gamut. Keep radiance nonnegative;
/// the consumer scales this chromaticity to its requested photopic luminance.
fn xyz_to_linear_rgb(x: f32, z: f32) -> Vec3 {
    let rgb = Color::xyz(x, 1., z).to_linear();
    Vec3::new(rgb.red, rgb.green, rgb.blue).max(Vec3::ZERO)
}

fn encoded_chromaticity(linear: Vec3) -> Vec3 {
    let c = linear / linear.max_element();
    let rgb = Color::linear_rgb(c.x, c.y, c.z).to_srgba();
    Vec3::new(rgb.red, rgb.green, rgb.blue)
}
