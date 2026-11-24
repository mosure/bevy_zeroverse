//! Installation-coherent glazing with metric privacy fields and filtered
//! roughness. Opaque ceramic frit is geometry, so all annotation policies agree.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::facade::WindowOpening, layout::stream, IndoorQuality,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlazingProgram {
    pub identity: u64,
    pub installation: u64,
    pub thickness_m: f32,
    /// Clear, uniform frost, privacy band, gradient, dots, stripes, ceramic frit.
    pub treatment: u8,
    pub band_m: Vec2,
    pub feather_m: f32,
    pub pitch_m: f32,
    pub duty: f32,
    pub frost_roughness: f32,
    /// Normalized hinge state. The constructed sweep never exceeds 14 cm.
    pub opening: f32,
}
impl GlazingProgram {
    pub(crate) fn treatment_name(&self) -> &'static str {
        match self.treatment {
            0 => "Clear",
            1 => "UniformFrost",
            2 => "PrivacyBand",
            3 => "Gradient",
            4 => "Dots",
            5 => "EtchedStripes",
            6 => "CeramicFrit",
            _ => "Unknown",
        }
    }
    pub(crate) fn glass_recipe(&self) -> glass::GlassRecipe {
        let mut recipe = glass::GlassRecipe::sample(self.installation, Surface::Glass);
        recipe.thickness_m = self.thickness_m;
        recipe
    }
    pub fn sample(seed: u64, index: usize, operable: bool) -> Self {
        let mut installation = stream(seed, 3100);
        let mut rng = stream(seed, 3101 + index as u64);
        let choice = rng.random_range(0..12);
        Self {
            identity: rng.random(),
            installation: seed,
            thickness_m: installation.random_range(0.006..0.014),
            treatment: if choice < 6 { 0 } else { choice - 5 },
            band_m: Vec2::new(
                installation.random_range(0.50..1.05),
                installation.random_range(1.25..1.90),
            ),
            feather_m: rng.random_range(0.01..0.16),
            pitch_m: rng.random_range(0.065..0.20),
            duty: rng.random_range(0.22..0.60),
            frost_roughness: rng.random_range(0.18..0.48),
            opening: if operable && rng.random_bool(0.35) {
                rng.random_range(0.15..1.)
            } else {
                0.
            },
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(0.004..=0.02).contains(&self.thickness_m)
            || self.treatment > 6
            || !self.band_m.is_finite()
            || self.band_m.x < 0.
            || self.band_m.y <= self.band_m.x
            || !(0.005..=0.3).contains(&self.feather_m)
            || !(0.04..=0.3).contains(&self.pitch_m)
            || !(0.1..=0.8).contains(&self.duty)
            || !(0.1..=0.6).contains(&self.frost_roughness)
            || !(0.0..=1.).contains(&self.opening)
        {
            return Err("invalid glazing construction/treatment".into());
        }
        Ok(())
    }
    pub fn frost(&self, p: Vec2) -> f32 {
        let smooth = |x: f32| {
            let x = x.clamp(0., 1.);
            x * x * (3. - 2. * x)
        };
        let band = smooth((p.y - self.band_m.x) / self.feather_m)
            * smooth((self.band_m.y - p.y) / self.feather_m);
        match self.treatment {
            1 => 1.,
            2 => band,
            3 => smooth((self.band_m.y - p.y) / (self.band_m.y - self.band_m.x)),
            4 => {
                let q = Vec2::new(
                    (p.x / self.pitch_m).rem_euclid(1.),
                    (p.y / self.pitch_m).rem_euclid(1.),
                ) - Vec2::splat(0.5);
                band * smooth((self.duty * 0.5 - q.length()) * self.pitch_m / 0.004)
            }
            5 => {
                band * smooth(
                    (self.duty - (p.y / self.pitch_m).rem_euclid(1.)) * self.pitch_m / 0.004,
                )
            }
            _ => 0.,
        }
    }
}

#[derive(Clone, Copy)]
enum MapFootprint {
    Opening,
    /// Only height varies across this installation, including at grazing angles.
    VerticalStrip,
}

fn roughness_texel(moment: f32) -> u8 {
    (moment.sqrt().sqrt() * 255.).round() as u8
}

fn roughness_image(
    program: &GlazingProgram,
    opening: &WindowOpening,
    clear: f32,
    footprint: MapFootprint,
) -> Image {
    const SIZE: u32 = 128;
    // Only dots vary across X. Evaluate each distinct row once. Exterior maps
    // keep their full width for the same GPU mip selection at grazing angles.
    let mut cols = if program.treatment == 4 { SIZE } else { 1 };
    let width = match footprint {
        MapFootprint::Opening => SIZE,
        MapFootprint::VerticalStrip => cols,
    };
    let mut moments: Vec<f32> = (0..cols * SIZE)
        .map(|i| {
            let p = Vec2::new((i % cols) as f32, (i / cols) as f32);
            [
                Vec2::new(0.25, 0.25),
                Vec2::new(0.75, 0.25),
                Vec2::new(0.25, 0.75),
                Vec2::new(0.75, 0.75),
            ]
            .map(|offset| {
                let xy = opening.min + (p + offset) / SIZE as f32 * (opening.max - opening.min);
                clear
                    .lerp(program.frost_roughness, program.frost(xy))
                    .powi(4)
            })
            .iter()
            .sum::<f32>()
                * 0.25
        })
        .collect();
    let texels = if width == 1 {
        2 * SIZE - 1
    } else {
        (4 * SIZE * SIZE - 1) / 3
    };
    let mut bytes = Vec::with_capacity(4 * texels as usize);
    let mut mip_width = width;
    let mut n = SIZE;
    loop {
        for &m in &moments {
            let texel = [255, roughness_texel(m), 0, 255];
            for _ in 0..mip_width / cols {
                bytes.extend_from_slice(&texel);
            }
        }
        if n == 1 {
            break;
        }
        let next_cols = (cols / 2).max(1);
        let mut next = Vec::with_capacity((n / 2 * next_cols) as usize);
        let dx = u32::from(cols > 1);
        for y in 0..n / 2 {
            for x in 0..next_cols {
                next.push(
                    [0, dx, cols, cols + dx]
                        .iter()
                        .map(|d| moments[(y * 2 * cols + x * 2 + d) as usize])
                        .sum::<f32>()
                        * 0.25,
                );
            }
        }
        moments = next;
        n /= 2;
        mip_width = (mip_width / 2).max(1);
        cols = next_cols;
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    );
    image.data = Some(bytes);
    image.texture_descriptor.mip_level_count = SIZE.ilog2() + 1;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

pub(super) fn build(
    scene: &IndoorManifest,
    quality: IndoorQuality,
    images: &mut impl super::super::preparation::AssetStore<Image>,
    materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
) -> BTreeMap<u64, Handle<StandardMaterial>> {
    use crate::scene::procedural_indoor::architecture::interior_glazing::{
        partition_height, shared_facade, InteriorGlazing,
    };
    let facades: Vec<_> = if let Some(envelope) = &scene.envelope {
        envelope
            .walls
            .iter()
            .filter_map(|w| w.facade.as_ref())
            .collect()
    } else {
        scene
            .exterior
            .iter()
            .flat_map(|e| e.facades.iter())
            .collect()
    };
    let mut result = BTreeMap::new();
    let adjoining = scene
        .adjoining_elevation
        .as_ref()
        .and_then(|p| p.windows.as_ref())
        .map(|w| &w.facade);
    // All entry-wall bays have the same vertical UV range and a height-only
    // treatment. Register just one material, and none for a solid wall.
    let shared = (scene.envelope.is_some()
        && InteriorGlazing::for_scene(scene).is_some_and(|p| p.vision_panels.is_some()))
    .then(|| shared_facade(scene, partition_height(scene)));
    for (opening, footprint) in facades
        .into_iter()
        .chain(adjoining)
        .flat_map(|f| &f.openings)
        .map(|o| (o, MapFootprint::Opening))
        .chain(
            shared
                .as_ref()
                .and_then(|f| f.openings.first())
                .map(|o| (o, MapFootprint::VerticalStrip)),
        )
    {
        let Some(program) = &opening.glazing else {
            continue;
        };
        let recipe = program.glass_recipe();
        let mut material = StandardMaterial::default();
        recipe.apply(&mut material, quality);
        match program.treatment {
            1 => {
                // Uniform frost is constant across every texel and mip. Match
                // the original UNORM8 material input without uploading a map.
                let uniform = recipe.roughness.lerp(program.frost_roughness, 1.);
                material.perceptual_roughness = f32::from(roughness_texel(uniform.powi(4))) / 255.;
            }
            2..=5 => {
                material.metallic_roughness_texture = Some(images.add(roughness_image(
                    program,
                    opening,
                    recipe.roughness,
                    footprint,
                )));
                material.perceptual_roughness = 1.;
            }
            _ => {}
        }
        result.insert(program.identity, materials.add(material));
    }
    result
}

#[cfg(test)]
mod tests;
