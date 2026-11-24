//! Fitted spectacles. The transparent optical face and opaque frame retain the
//! existing person materials, annotation policy and rigid head binding.
mod fit;
mod mesh;
use super::stream;
use bevy::prelude::*;
pub(super) use fit::Fit;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rim {
    Full,
    Upper,
    DrillMounted,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyewearProgram {
    pub rim: Rim,
    /// Nominal unscaled dimensions. The lens centres are fitted to the two eye bones.
    pub bridge_width_m: f32,
    pub lens_height_m: f32,
    pub decentration_m: f32,
    pub corner_exponent: f32,
    pub lower_taper: f32,
    pub upsweep_m: f32,
    pub tilt: f32,
    pub wrap: f32,
    pub lens_curve_radius_m: f32,
    pub cornea_clearance_m: f32,
    pub rim_radius_m: f32,
    pub brow_weight: f32,
    pub bridge_rise_m: f32,
    pub ear_hook: f32,
    pub frame_roughness: f32,
    pub lens_tint: [f32; 3],
    pub lens_opacity: f32,
}
impl EyewearProgram {
    pub(super) fn sample(seed: u64, metallic: f32) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"FrameFit"));
        let mut finish = stream(seed, u64::from_le_bytes(*b"SpecCoat"));
        let metal = metallic > 0.5;
        let draw: f32 = r.random();
        let rim = if metal && draw < 0.21 {
            Rim::DrillMounted
        } else if draw < if metal { 0.51 } else { 0.22 } {
            Rim::Upper
        } else {
            Rim::Full
        };
        let filter: f32 = finish.random_range(-1.0..1.);
        Self {
            rim,
            bridge_width_m: r.random_range(0.012..0.023),
            lens_height_m: r.random_range(0.029..0.048),
            decentration_m: r.random_range(-0.0007..0.0032),
            corner_exponent: r.random_range(0.45..1.12),
            lower_taper: r.random_range(0.02..0.20),
            upsweep_m: r.random_range(-0.0012..0.0053),
            tilt: r.random_range(0.045..0.21),
            wrap: r.random_range(0.06..0.24),
            lens_curve_radius_m: r.random_range(0.076..0.19),
            cornea_clearance_m: r.random_range(0.009..0.015),
            rim_radius_m: if metal {
                r.random_range(0.00065..0.00135)
            } else {
                r.random_range(0.0013..0.0028)
            },
            brow_weight: r.random_range(0.15..0.70),
            bridge_rise_m: r.random_range(0.002..0.008),
            ear_hook: r.random_range(0.06..0.96),
            frame_roughness: if metal {
                finish.random_range(0.21..0.44)
            } else {
                finish.random_range(0.22..0.40)
            },
            lens_tint: [0.96 + 0.036 * filter, 0.97, 0.96 - 0.032 * filter],
            lens_opacity: 0.027 + 0.050 * finish.random::<f32>().powi(2),
        }
    }
    pub(super) fn valid(&self) -> bool {
        [
            (self.bridge_width_m, 0.010, 0.025),
            (self.lens_height_m, 0.026, 0.051),
            (self.decentration_m, -0.002, 0.005),
            (self.corner_exponent, 0.42, 1.20),
            (self.lower_taper, 0., 0.23),
            (self.upsweep_m, -0.002, 0.006),
            (self.tilt, 0., 0.24),
            (self.wrap, 0., 0.28),
            (self.lens_curve_radius_m, 0.070, 0.21),
            (self.cornea_clearance_m, 0.008, 0.018),
            (self.rim_radius_m, 0.0006, 0.003),
            (self.brow_weight, 0., 0.8),
            (self.bridge_rise_m, 0.001, 0.009),
            (self.ear_hook, 0., 1.),
            (self.frame_roughness, 0.15, 0.50),
            (self.lens_opacity, 0.015, 0.10),
        ]
        .iter()
        .all(|&(v, lo, hi)| (lo..=hi).contains(&v))
            && self.lens_tint.iter().all(|v| (0.80..=1.).contains(v))
    }
    pub(super) fn lens_color(&self) -> Color {
        let [r, g, b] = self.lens_tint;
        Color::srgba(r, g, b, self.lens_opacity)
    }
}

/// Polished frames include brushed steel, warm alloys and dark coated wire.
/// The independent finish stream leaves all other actor properties untouched.
pub(super) fn metal_tint(seed: u64, original: [f32; 3]) -> [f32; 3] {
    let mut r = stream(seed, u64::from_le_bytes(*b"SpecWire"));
    if r.random_bool(0.30) {
        return original;
    }
    let warmth = r.random::<f32>().powi(2);
    (Vec3::new(0.76, 0.78, 0.80).lerp(Vec3::new(0.81, 0.62, 0.37), warmth)
        * r.random_range(0.40..0.98))
    .to_array()
}
#[cfg(test)]
mod tests;
