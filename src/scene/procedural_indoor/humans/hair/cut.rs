//! Scissor and clipper cuts measured on the fitted head. The deposited hair,
//! exposed scalp and garment below it share the same person instance.
use super::HairStyle;
use crate::scene::procedural_indoor::layout::stream;
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalpCut {
    pub temple_recession_m: f32,
    pub temple_asymmetry: f32,
    pub peak_depth_m: f32,
    pub nape_raise_m: f32,
    /// Full width at the front of a combed part, tapering at the crown.
    pub part_width_m: f32,
    pub part_end: f32,
    pub part_definition: f32,
    pub clip_length_m: f32,
    pub fibre_density: f32,
}

impl ScalpCut {
    pub(super) fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"scalpcut"));
        Self {
            temple_recession_m: 0.002 + r.random::<f32>().powi(2) * 0.038,
            temple_asymmetry: r.random_range(-0.26..0.26),
            peak_depth_m: r.random::<f32>().powi(2) * 0.012,
            nape_raise_m: r.random_range(-0.010..0.016),
            part_width_m: r.random_range(0.0014..0.0056),
            part_end: r.random_range(0.16..0.66),
            part_definition: r.random(),
            clip_length_m: r.random_range(0.0008_f32.ln()..0.008_f32.ln()).exp(),
            fibre_density: r.random_range(0.44..0.98),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.0..=0.045).contains(&self.temple_recession_m)
            && (-0.3..=0.3).contains(&self.temple_asymmetry)
            && (0.0..=0.015).contains(&self.peak_depth_m)
            && (-0.014..=0.020).contains(&self.nape_raise_m)
            && (0.001..=0.006).contains(&self.part_width_m)
            && (0.1..=0.75).contains(&self.part_end)
            && (0.0..=1.0).contains(&self.part_definition)
            && (0.0005..=0.009).contains(&self.clip_length_m)
            && (0.3..=1.0).contains(&self.fibre_density)
    }

    pub(super) fn hairline_offset(self, local: Vec3, front: f32, side: f32) -> f32 {
        let lobe = |u: f32| (1. - u * u).max(0.).powi(2);
        let temple = lobe((side - 0.61) / 0.40);
        front
            * (self.temple_recession_m * temple * (1. + self.temple_asymmetry * local.x)
                - self.peak_depth_m * lobe(local.x / 0.31))
            + self.nape_raise_m * local.z.clamp(0., 1.).powi(2)
    }

    pub fn part_width(self, style: HairStyle) -> f32 {
        let threshold = match style {
            HairStyle::SidePart => return self.part_width_m,
            HairStyle::StraightLong | HairStyle::Bob | HairStyle::AsymmetricBob => 0.20,
            HairStyle::Swept | HairStyle::WavyLong | HairStyle::LayeredLong => 0.40,
            HairStyle::Pixie => 0.60,
            _ => return 0.,
        };
        self.part_width_m * ((self.part_definition - threshold) / 0.18).clamp(0., 1.)
    }

    /// Area fraction covered by unresolved short fibres. A longer clip hides
    /// more scalp even on the same person, in linear reflected light.
    pub fn buzz_coverage(self) -> f32 {
        1. - (-self.fibre_density * (0.42 + 220. * self.clip_length_m)).exp()
    }
}

/// The two sides of a combed part follow the same curved line. The signed
/// lateral distance and half width can each be interpolated across a triangle.
#[derive(Clone, Copy)]
pub(super) struct Part {
    pub width: f32,
    pub crown: f32,
    pub start_x: f32,
    pub curve: f32,
}

impl Part {
    pub fn coordinates(self, local: Vec3, half_head_width: f32) -> Vec2 {
        let t = ((local.z + 1.) / (1. + self.crown)).clamp(0., 1.);
        let mid = self.start_x * (1. - 0.35 * t) + self.curve * 0.26 * t * (2. - t);
        let taper = smooth((self.crown - local.z) / 0.36) * smooth((local.y - 0.05) / 0.27);
        Vec2::new((local.x - mid) * half_head_width, 0.5 * self.width * taper)
    }
}

pub(super) fn smooth(x: f32) -> f32 {
    let t = x.clamp(0., 1.);
    t * t * (3. - 2. * t)
}

#[cfg(test)]
mod tests;
