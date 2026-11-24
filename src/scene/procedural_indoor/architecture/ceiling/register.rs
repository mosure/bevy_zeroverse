//! Air outlet stock is shared along a fabricated duct. Branch stations and
//! handedness fit that stock between the two support saddles in room metres.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegisterPattern {
    Directional,
    Opposed,
    CrossedReturn,
}

#[derive(Clone, Copy)]
pub(crate) struct RegisterStock {
    pub pattern: RegisterPattern,
    pub coated: bool,
    pub pitch: f32,
    pub tilt: f32,
    pub depth: f32,
    pub flange: f32,
    length: f32,
    aspect: f32,
    lateral: f32,
    first_side: f32,
    alternate: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct RegisterSite {
    /// Station on the trunk centreline, from zero to one.
    pub station: f32,
    /// Room-local transverse displacement from the module centre.
    pub lateral_m: f32,
    pub size: Vec2,
}

impl CeilingModule {
    pub(crate) fn duct_register_stock(&self, generator_version: u32) -> Option<RegisterStock> {
        if generator_version < 192
            || !matches!(self.detail.as_ref()?.shape, CeilingShape::Duct { .. })
        {
            return None;
        }
        let mut r = stream(self.seed, u64::from_le_bytes(*b"AirStock"));
        let pattern = match r.random_range(0..10) {
            0..=3 => RegisterPattern::Directional,
            4..=6 => RegisterPattern::Opposed,
            _ => RegisterPattern::CrossedReturn,
        };
        Some(RegisterStock {
            pattern,
            coated: r.random_bool(0.64),
            pitch: r.random_range(0.014..0.036),
            tilt: r.random_range(0.30..0.86) * if r.random_bool(0.5) { 1. } else { -1. },
            depth: r.random_range(0.029..0.075),
            flange: r.random_range(0.005..0.014),
            length: r.random_range(0.19..0.53),
            aspect: r.random_range(1.2..3.7),
            lateral: r.random_range(0.265..0.355),
            first_side: if r.random_bool(0.5) { 1. } else { -1. },
            alternate: r.random_bool(0.69),
        })
    }
}

impl RegisterStock {
    pub fn site(self, m: &CeilingModule, branches: u8, index: u8) -> RegisterSite {
        let mut r = stream(m.seed, u64::from_le_bytes(*b"AirSite ") ^ u64::from(index));
        let length = m.size.x.max(m.size.z);
        let width = m.size.x.min(m.size.z);
        let bay = 0.72 / f32::from(branches.max(1));
        let offset = r.random_range(-0.20_f32..0.20);
        let station = 0.14 + bay * (f32::from(index) + 0.5 + offset);
        let side = if self.alternate && index % 2 == 1 {
            -self.first_side
        } else {
            self.first_side
        };
        let lateral_m = if branches == 0 {
            0.
        } else {
            side * self.lateral * width
        };
        // The belly outlet follows the bent trunk (at most 0.2 widths off centre).
        let width_limit = if branches == 0 {
            width * 0.55
        } else {
            2. * (width * 0.49 - lateral_m.abs())
        };
        // Adjacent outlets and 26 mm saddle bars retain an installation gap.
        let bay_clear = length * 0.94 * bay * (1. - 2. * offset.abs()) - 0.032;
        let face_length = (self.length * r.random_range(0.94..1.06))
            .min(width_limit * self.aspect)
            .min(bay_clear);
        RegisterSite {
            station,
            lateral_m,
            size: Vec2::new(face_length, face_length / self.aspect),
        }
    }
}
