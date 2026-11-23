//! Seeded source output, independent of geometry, material content and frame time.
//! Direct lights, visible apertures and diffuse transport consume these samples.
use super::{
    layout::{stream, IndoorManifest, IndoorObject, ObjectKind},
    materials::kelvin_rgb,
};
use bevy::prelude::*;
use rand::Rng;

mod illuminants;
pub use illuminants::IlluminantModel;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SourcePower {
    pub kelvin: f32,
    /// Small LED/fluorescent green-magenta departure from the temperature locus.
    pub tint: f32,
    pub dimmer: f32,
    pub lumens: f32,
    /// sRGB chromaticity, in the same convention as Bevy's light color.
    pub color: Vec3,
}

/// World-space geometry shared by direct lighting, diffuse transport and capture metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FixtureBeam {
    pub position: Vec3,
    pub direction: Vec3,
    pub radius: f32,
    pub inner: f32,
    pub outer: f32,
}

impl FixtureBeam {
    pub(crate) fn tilt_from_down_degrees(self) -> f32 {
        self.direction
            .xz()
            .length()
            .atan2(-self.direction.y)
            .to_degrees()
    }
}

pub fn lighting_seed(scene: &IndoorManifest) -> u64 {
    scene
        .appearance
        .as_ref()
        .and_then(|a| a.lighting_seed)
        .unwrap_or(scene.seed)
}

/// The encoded RGB palette used by displays, decorative bulbs and older rooms.
pub fn tinted_white(kelvin: f32, tint: f32) -> Vec3 {
    let c = kelvin_rgb(kelvin.clamp(1800., 10000.))
        * Vec3::new(1. - tint * 0.45, 1. + tint, 1. - tint * 0.45);
    c / c.max_element()
}

/// Light intensity is specified in photopic lumens/lux. Preserve its luminance
/// when changing the source spectrum; the input is bounded display chromaticity.
pub(crate) fn source_color(scene: &IndoorManifest, chromaticity: Vec3) -> Color {
    let srgb = Color::srgb(chromaticity.x, chromaticity.y, chromaticity.z);
    if scene.domain().is_none_or(|d| d.photometry.meter.is_none()) {
        return srgb;
    }
    Color::LinearRgba(photopic_emission(srgb, 1.))
}

fn photopic_emission(color: Color, luminance: f32) -> LinearRgba {
    let c = color.to_linear();
    let y = c.red * 0.2126 + c.green * 0.7152 + c.blue * 0.0722;
    let scale = luminance / y.max(0.001);
    LinearRgba::new(c.red * scale, c.green * scale, c.blue * scale, 1.)
}

pub fn is_emitter(kind: ObjectKind) -> bool {
    is_lamp(kind)
        || matches!(
            kind,
            ObjectKind::ExitSign
                | ObjectKind::FireAlarm
                | ObjectKind::FireDetector
                | ObjectKind::StringLights
        )
}

/// Number of distinct visible emission bindings on this fixture.
pub fn emission_channels(o: &IndoorObject) -> usize {
    if o.kind == ObjectKind::StringLights {
        super::objects::equipment::signals::string_program(o)
            .colors_srgb
            .len()
    } else if o.kind == ObjectKind::ExitSign {
        1 + usize::from(super::objects::equipment::signals::exit::appearance(o, 1).is_some())
    } else {
        1
    }
}

pub fn decorative_emission(o: &IndoorObject) -> Option<LinearRgba> {
    decorative_emission_channel(o, 0)
}

pub fn decorative_emission_channel(o: &IndoorObject, channel: usize) -> Option<LinearRgba> {
    Some(match o.kind {
        ObjectKind::ExitSign => super::objects::equipment::signals::exit::appearance(o, channel)?.1,
        ObjectKind::FireAlarm | ObjectKind::FireDetector => LinearRgba::new(1.8, 0.01, 0.005, 1.),
        ObjectKind::StringLights => {
            let p = super::objects::equipment::signals::string_program(o);
            p.emission(channel, 1.)?
        }
        _ => return None,
    })
}

pub fn is_lamp(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::FloorLamp
            | ObjectKind::DeskLamp
            | ObjectKind::CeilingPendant
            | ObjectKind::WallSconce
    )
}

/// Static probes represent the whole capture sequence. A lamp moved directly
/// or carried on a cart can contribute its live direct light at playback time.
pub(crate) fn stationary_source(scene: &IndoorManifest, object: &IndoorObject) -> bool {
    scene
        .indoor_motion
        .as_ref()
        .is_none_or(|m| m.mechanism_for_object(object.id).is_none())
}

/// Independently operated task/decorative sources. The common component keeps
/// most fixtures coherent with the room; replacement bulbs and tunable LEDs
/// fill the interval between warm residential and cool work lighting.
pub fn lamp(scene: &IndoorManifest, object: &IndoorObject) -> SourcePower {
    let mut r = stream(
        object.seed ^ lighting_seed(scene).rotate_left(23),
        0x4c414d50504f5745,
    );
    let (lo, hi): (f32, f32) = match object.kind {
        ObjectKind::DeskLamp => (120., 1000.),
        ObjectKind::FloorLamp => (400., 2400.),
        ObjectKind::WallSconce => (150., 1600.),
        ObjectKind::CeilingPendant => (400., 3000.),
        _ => (0., 0.),
    };
    let rated = if lo > 0. {
        r.random_range(lo.ln()..hi.ln()).exp()
    } else {
        0.
    };
    let blend = r.random_range(0.15..0.85);
    let kelvin = scene.light_kelvin * (1. - blend) + r.random_range(1900.0..8200.0) * blend;
    let tint = r.random_range(-0.055..0.055);
    // Mostly lit and near rated output, with a continuous dim tail and genuine
    // switched-off lamps. This never switches the whole ceiling circuit off.
    let dimmer = if r.random_bool(0.09) {
        0.
    } else {
        r.random_range(0.08_f32..1.).sqrt()
    };
    let scale = scene
        .appearance
        .as_ref()
        .map_or(1., |a| a.illumination_scale);
    // Preserve intentionally low-light rooms without letting dimming change
    // the rated bulb sample or applying appearance attenuation twice.
    let room_dimmer = ((scene.target_lux / scale.max(0.001)) / 300.).clamp(0.05, 1.);
    let dimmer = dimmer * room_dimmer;
    SourcePower {
        kelvin,
        tint,
        dimmer,
        lumens: rated * dimmer * scale,
        color: scene.illuminant_model().lamp_srgb(kelvin, tint),
    }
}

/// Exit-aperture radiance shared by the visible lamp and reflection probes.
pub(crate) fn lamp_emission(scene: &IndoorManifest, power: SourcePower, area: f32) -> LinearRgba {
    let c = source_color(scene, power.color).to_linear();
    let radiance = power.lumens / (std::f32::consts::PI * area.max(0.0001));
    LinearRgba::new(c.red * radiance, c.green * radiance, c.blue * radiance, 1.)
}

/// White-pixel emission normalized to photopic luminance. Display content then
/// modulates this radiance; RGB-only color changes never alter annotation maps.
pub fn display_emission(luminance: f32, kelvin: f32, tint: f32) -> LinearRgba {
    let c = tinted_white(kelvin, tint);
    photopic_emission(Color::srgb(c.x, c.y, c.z), luminance)
}
