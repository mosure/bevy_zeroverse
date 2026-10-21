//! Seeded source output, independent of geometry, material content and frame time.
//! Direct lights, visible apertures and diffuse transport consume these samples.
use super::{
    layout::{stream, IndoorManifest, IndoorObject, ObjectKind},
    materials::kelvin_rgb,
};
use bevy::prelude::*;
use rand::Rng;

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

pub fn lighting_seed(scene: &IndoorManifest) -> u64 {
    scene
        .appearance
        .as_ref()
        .and_then(|a| a.lighting_seed)
        .unwrap_or(scene.seed)
}

pub fn tinted_white(kelvin: f32, tint: f32) -> Vec3 {
    let c = kelvin_rgb(kelvin.clamp(1800., 10000.))
        * Vec3::new(1. - tint * 0.45, 1. + tint, 1. - tint * 0.45);
    c / c.max_element()
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

/// Small indicator/decorative sources are emissive geometry, not new shadow lights.
pub fn emission_channels(o: &IndoorObject) -> usize {
    if o.kind == ObjectKind::StringLights {
        super::objects::equipment::signals::string_program(o)
            .colors_srgb
            .len()
    } else {
        1
    }
}

pub fn decorative_emission(o: &IndoorObject) -> Option<LinearRgba> {
    decorative_emission_channel(o, 0)
}

pub fn decorative_emission_channel(o: &IndoorObject, channel: usize) -> Option<LinearRgba> {
    Some(match o.kind {
        ObjectKind::ExitSign => {
            if o.variant.is_multiple_of(2) {
                LinearRgba::new(3., 180., 38., 1.)
            } else {
                LinearRgba::new(220., 5., 2., 1.)
            }
        }
        ObjectKind::FireAlarm | ObjectKind::FireDetector => LinearRgba::new(1.8, 0.01, 0.005, 1.),
        ObjectKind::StringLights => {
            let p = super::objects::equipment::signals::string_program(o);
            let [r, g, b] = *p.colors_srgb.get(channel)?;
            let c = Color::srgb(r, g, b).to_linear();
            // Bound peak radiance as well as brightness; saturated blue must not
            // become a pathological source by photopic normalization alone.
            let scale = (p.luminance
                / (0.2126 * c.red + 0.7152 * c.green + 0.0722 * c.blue).max(0.1))
            .min(2500.);
            LinearRgba::new(c.red * scale, c.green * scale, c.blue * scale, 1.)
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
        color: tinted_white(kelvin, tint),
    }
}

/// White-pixel emission normalized to photopic luminance. Display content then
/// modulates this radiance; RGB-only color changes never alter annotation maps.
pub fn display_emission(luminance: f32, kelvin: f32, tint: f32) -> LinearRgba {
    let c = tinted_white(kelvin, tint);
    let c = Color::srgb(c.x, c.y, c.z).to_linear();
    let y = c.red * 0.2126 + c.green * 0.7152 + c.blue * 0.0722;
    let scale = luminance / y.max(0.001);
    LinearRgba::new(c.red * scale, c.green * scale, c.blue * scale, 1.)
}
