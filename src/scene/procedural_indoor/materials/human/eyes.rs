//! A single mipmapped color chart per actor, shared by both eyes and every
//! timestep. Pigment is below the wet interface; highlights come from scene lights.
use super::super::{mip_image, ImageAddressMode, ImageSampler, MapType};
use crate::scene::procedural_indoor::{
    humans::eyes::{smoothstep, CHART_EXTENT},
    layout::{stream, IndoorManifest},
    preparation::AssetStore,
};
use bevy::prelude::*;
use rand::Rng;
use std::collections::BTreeMap;

const SIZE: u32 = 64;

struct PigmentChart {
    iris: Vec3,
    sclera: Vec3,
    pupil: f32,
    limbus: f32,
    fibres: [f32; 64],
}

impl PigmentChart {
    fn new(seed: u64, iris: [f32; 3], pupil: f32) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"IrisDye1"));
        let value = rng.random_range(0.67..0.81);
        let warmth = rng.random_range(0.012..0.050);
        Self {
            iris: Vec3::from_array(iris),
            sclera: Vec3::new(value, value - warmth, value - warmth * 1.30),
            pupil,
            limbus: rng.random_range(0.075..0.175),
            fibres: std::array::from_fn(|_| rng.random_range(-1.0..1.0)),
        }
    }

    fn angular_noise(&self, turn: f32, stations: usize) -> f32 {
        let at = turn.rem_euclid(1.) * stations as f32;
        let index = at.floor() as usize % stations;
        let t = at.fract();
        self.fibres[index]
            + t * t * (3. - 2. * t) * (self.fibres[(index + 1) % stations] - self.fibres[index])
    }

    fn color(&self, point: Vec2) -> Vec3 {
        let pupil_color = Vec3::new(0.040, 0.034, 0.030);
        let r = point.length();
        if r >= 1.035 {
            return self.sclera;
        }
        if r < self.pupil - 0.025 {
            return pupil_color;
        }
        let turn = point.y.atan2(point.x) / std::f32::consts::TAU;
        let broad = self.angular_noise(turn, 13);
        let fibre = self.angular_noise(turn + 0.017 * (1. - r) * broad, 64);
        let collarette = self.pupil + (1. - self.pupil) * (0.34 + broad * 0.09);
        let inner = 1. - smoothstep(collarette - 0.07, collarette + 0.06, r);
        let stroma = self.iris
            * (1.0 + 0.21 * fibre)
            * Vec3::new(1. + 0.15 * inner, 1. - 0.02 * inner, 1. - 0.15 * inner);
        let rim = smoothstep(1. - self.limbus, 0.986, r);
        let iris = stroma * (1. - 0.46 * rim);
        pupil_color
            .lerp(iris, smoothstep(self.pupil - 0.025, self.pupil + 0.025, r))
            .lerp(self.sclera, smoothstep(0.986, 1.035, r))
    }

    fn image(&self) -> Image {
        let mut bytes = Vec::with_capacity((SIZE * SIZE * 4) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let uv = (Vec2::new(x as f32, y as f32) + 0.5) / SIZE as f32;
                let color = self.color((uv - 0.5) * (2. * CHART_EXTENT));
                bytes.extend(
                    color
                        .to_array()
                        .map(|c| (255. * c.clamp(0., 1.)).round() as u8),
                );
                bytes.push(255);
            }
        }
        let mut image = mip_image(bytes, SIZE, MapType::Color);
        if let ImageSampler::Descriptor(sampler) = &mut image.sampler {
            sampler.address_mode_u = ImageAddressMode::ClampToEdge;
            sampler.address_mode_v = ImageAddressMode::ClampToEdge;
        }
        image
    }
}

pub(in crate::scene::procedural_indoor::materials) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
) -> BTreeMap<usize, Handle<Image>> {
    let illuminance = scene
        .domain()
        .and_then(|d| d.photometry.meter)
        .map_or(scene.target_lux, |meter| meter.incident_lux());
    scene
        .humans
        .iter()
        .filter_map(|person| {
            let face = &person.appearance.as_ref()?.face;
            let pupil = face.optics?.adapted_pupil(illuminance);
            let chart = PigmentChart::new(person.seed, face.iris, pupil);
            Some((person.id, images.add(chart.image())))
        })
        .collect()
}

#[cfg(test)]
mod tests;
