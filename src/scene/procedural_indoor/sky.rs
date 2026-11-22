//! Room-local, distant skylight shared by photography, reflections and transport.
//! A square unfolded upper hemisphere avoids an azimuth seam and pole singularity.
use super::{
    architecture,
    layout::{stream, IndoorManifest},
};
use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use noise::{NoiseFn, OpenSimplex};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
#[cfg(test)]
mod tests;

pub(crate) const SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkyDistribution {
    /// Direct beam survival, shared with the solar and diffuse illuminance sampler.
    pub beam_transmission: f32,
    pub haze: f32,
    pub cloud_frequency: f32,
    pub cloud_seed: u32,
}

impl SkyDistribution {
    pub(super) fn sample(seed: u64, beam_transmission: f32) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"SkyField"));
        Self {
            beam_transmission,
            haze: rng.random_range(0.08..0.95),
            cloud_frequency: rng.random_range(1.5..5.2),
            cloud_seed: rng.random(),
        }
    }

    pub(super) fn valid(self) -> bool {
        (0.0..=1.).contains(&self.beam_transmission)
            && (0.0..=1.).contains(&self.haze)
            && (0.5..=10.).contains(&self.cloud_frequency)
    }

    fn relative(self, direction: Vec3, sun: Vec3, weather: &OpenSimplex) -> Vec3 {
        let h = direction.y;
        let horizon = (1. - h).powi(3);
        // Scattered light around the solar direction. The direct disk remains
        // the directional light's responsibility.
        let aureole = direction.dot(sun).max(0.).powi(12);
        let clear = Vec3::new(0.27, 0.54, 1.09)
            + Vec3::new(1.13, 0.89, 0.52) * (horizon * (0.56 + self.haze))
            + Vec3::new(2.70, 2.11, 1.26) * (aureole * (0.65 + self.haze));
        let coord = direction.as_dvec3() * self.cloud_frequency as f64;
        let density = (0.5
            + 0.75 * weather.get(coord.to_array()) as f32
            + 0.25 * weather.get((coord * 2.43 + bevy::math::DVec3::splat(5.7)).to_array()) as f32)
            .clamp(0., 1.);
        let cover = 1. - self.beam_transmission;
        let broken =
            (0.10 + 2.65 * cover * (1. - cover)) * (density * density * (3. - 2. * density));
        // The standard overcast luminance varies as 1 + 2 sin(altitude).
        // Broken cloud has a cooler shaded base and broad sunlit shoulders.
        let cloud = Vec3::new(1.0, 1.015, 1.04) * ((1. + 2. * h) * (0.59 + 0.26 * density));
        clear.lerp(cloud, cover + (1. - cover) * broken)
    }
}

/// Node directions and the cosine-weighted solid-angle Jacobian on the folded square.
fn quadrature() -> &'static [(Vec3, f32)] {
    static NODES: OnceLock<Vec<(Vec3, f32)>> = OnceLock::new();
    NODES.get_or_init(|| {
        (0..SIZE * SIZE)
            .map(|i| {
                let uv =
                    Vec2::new((i % SIZE) as f32, (i / SIZE) as f32) * (2. / (SIZE - 1) as f32) - 1.;
                let xz = Vec2::new(uv.x + uv.y, uv.x - uv.y) * 0.5;
                let p = Vec3::new(xz.x, (1. - xz.abs().element_sum()).max(0.), xz.y);
                let inv = p.length_recip();
                let d = p * inv;
                (d, d.y * inv.powi(3))
            })
            .collect()
    })
}

/// Empty for archived isotropic skies. Realized texels contain decoded linear
/// radiance in the same units and row order as the RGB emissive texture.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SkyField(pub(crate) Arc<[Vec3]>);

pub(super) struct SkyMap {
    pub field: SkyField,
    pub texture: Image,
    pub emission: Vec3,
}

pub(super) fn bake(scene: &IndoorManifest) -> Option<SkyMap> {
    let p = &scene.domain()?.photometry;
    let distribution = p.sky_distribution?;
    let sun = architecture::sun_direction(scene);
    let noise = OpenSimplex::new(distribution.cloud_seed);
    let nodes = quadrature();
    let mut integral = Vec3::ZERO;
    let mut peak = Vec3::ZERO;
    let mut values = Vec::with_capacity(nodes.len());
    for &(direction, weight) in nodes {
        let v = distribution.relative(direction, sun, &noise);
        integral += v * weight;
        peak = peak.max(v);
        values.push(v);
    }
    let weight: f32 = nodes.iter().map(|v| v.1).sum();
    let emission = p.sky_radiance * (peak * weight / integral);
    let mut pixels = Vec::with_capacity(values.len() * 4);
    for v in &mut values {
        let rgb = *v / peak;
        let encoded = Color::linear_rgb(rgb.x, rgb.y, rgb.z)
            .to_srgba()
            .to_u8_array();
        pixels.extend(encoded);
        let linear = Color::srgb_u8(encoded[0], encoded[1], encoded[2]).to_linear();
        *v = emission * Vec3::new(linear.red, linear.green, linear.blue);
    }
    let mut texture = Image::new(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    texture.sampler = ImageSampler::linear();
    Some(SkyMap {
        field: SkyField(values.into()),
        texture,
        emission,
    })
}

impl SkyField {
    /// Called only for escaping upper-hemisphere transport rays. Below the
    /// geometric horizon RGB continues the horizon until rendered grade takes over.
    pub(crate) fn radiance(&self, direction: Vec3, uniform: Vec3) -> Vec3 {
        if self.0.is_empty() {
            return uniform;
        }
        let d = direction / (direction.x.abs() + direction.y.max(0.) + direction.z.abs()).max(1e-8);
        let uv = Vec2::new(d.x + d.z, d.x - d.z) * 0.5 + 0.5;
        let grid = uv.clamp(Vec2::ZERO, Vec2::ONE) * (SIZE - 1) as f32;
        let lo = grid.floor().as_uvec2();
        let hi = (lo + 1).min(UVec2::splat((SIZE - 1) as u32));
        let frac = grid - lo.as_vec2();
        let at = |x, y| self.0[y as usize * SIZE + x as usize];
        at(lo.x, lo.y)
            .lerp(at(hi.x, lo.y), frac.x)
            .lerp(at(lo.x, hi.y).lerp(at(hi.x, hi.y), frac.x), frac.y)
    }
}
