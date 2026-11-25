//! User-adjustable room thermostats. The setpoint drives both the numeric
//! readout and the mechanical index. All digital units share the fascia atlas.
use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorManifest,
    materials::{
        appliance_panel::{Film, Glyph},
        input_housing::Housing,
    },
};

mod face;
mod mesh;
pub(crate) use mesh::build;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    Dial,
    PushButton,
    Encoder,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Thermostat {
    pub style: Style,
    pub setpoint_c: f32,
    pub fahrenheit: bool,
    pub film: Film,
    pub casing: Surface,
    pub case_tone: &'static str,
    pub shoulder: f32,
    pub vent_gap_m: f32,
    pub vents: usize,
    pub screen_fraction: Vec2,
    pub mirrored_controls: bool,
}

pub(crate) fn fit(o: &mut IndoorObject) {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"TstatSku"));
    o.variant = match r.random_range(0..10) {
        0..=2 => 0,
        3..=7 => 1,
        _ => 2,
    };
    let height = r.random_range(0.077..0.132);
    let aspect = match o.variant {
        0 => r.random_range(0.93..1.16),
        1 => r.random_range(0.86..1.54),
        _ => r.random_range(1.18..1.82),
    };
    o.size = Vec3::new(height * aspect, height, r.random_range(0.023..0.048));
}

pub fn parameters(o: &IndoorObject) -> Thermostat {
    sample(o, &Housing::for_object(o))
}

fn sample(o: &IndoorObject, case: &Housing) -> Thermostat {
    let style = [Style::Dial, Style::PushButton, Style::Encoder][(o.variant % 3) as usize];
    let mut r = stream(o.seed, u64::from_le_bytes(*b"TstatCtl"));
    let mut occupancy = stream(o.seed, u64::from_le_bytes(*b"SetPoint"));
    let setpoint_c =
        (occupancy.random_range(16.5..27.5) + occupancy.random_range(18.0..26.0)) * 0.5;
    Thermostat {
        style,
        setpoint_c,
        fahrenheit: occupancy.random_bool(0.38),
        film: match style {
            Style::Dial if r.random_bool(0.76) => Film::LightKey,
            Style::Dial => Film::DarkKey,
            _ if r.random_bool(0.84) => Film::ReflectiveLcd,
            _ => Film::Led,
        },
        casing: case.surface,
        case_tone: case.tone(),
        shoulder: if style == Style::Dial {
            r.random_range(2.0..2.75)
        } else {
            r.random_range(3.0..6.2)
        },
        vent_gap_m: o.size.z * r.random_range(0.105..0.19),
        vents: r.random_range(4..=7),
        screen_fraction: Vec2::new(r.random_range(0.59..0.77), r.random_range(0.31..0.43)),
        mirrored_controls: r.random_bool(0.40),
    }
}
impl Thermostat {
    pub fn display_value(&self) -> u8 {
        (if self.fahrenheit {
            self.setpoint_c * 1.8 + 32.
        } else {
            self.setpoint_c
        })
        .round() as u8
    }
    /// Positive Z rotation turns the index left. Higher temperatures turn right.
    pub fn pointer_angle(&self) -> f32 {
        (0.5 - (self.setpoint_c - 15.) / 15.) * 5.0
    }
    /// Active glass aperture in local coordinates, from the wall-facing mount.
    pub fn screen(&self, o: &IndoorObject) -> Option<(Vec3, Vec2)> {
        let s = o.size;
        let side = if self.mirrored_controls { -1. } else { 1. };
        let (xy, scale) = match self.style {
            Style::Dial => return None,
            Style::PushButton => (Vec2::new(0., 0.63), self.screen_fraction),
            Style::Encoder => (
                Vec2::new(-side * 0.143, 0.52),
                self.screen_fraction * Vec2::new(0.69, 1.),
            ),
        };
        Some((
            (xy * s.truncate()).extend(s.z * 0.323),
            scale * s.truncate(),
        ))
    }
}
pub fn height_above_floor(scene: &IndoorManifest, o: &IndoorObject) -> f32 {
    o.position.y + o.size.y * 0.5 - scene.floor_height(standing_point(o).xz())
}

pub(crate) fn standing_point(o: &IndoorObject) -> Vec3 {
    o.transform().transform_point(Vec3::new(0., 0., 0.40))
}

#[cfg(test)]
mod tests;
