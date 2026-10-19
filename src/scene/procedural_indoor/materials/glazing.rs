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

fn roughness_image(program: &GlazingProgram, opening: &WindowOpening, clear: f32) -> Image {
    const SIZE: u32 = 128;
    let mut moments: Vec<f32> = (0..SIZE * SIZE)
        .map(|i| {
            let p = Vec2::new((i % SIZE) as f32, (i / SIZE) as f32);
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
    let mut bytes = Vec::new();
    let mut n = SIZE;
    loop {
        for &m in &moments {
            bytes.extend([255, (m.sqrt().sqrt() * 255.).round() as u8, 0, 255]);
        }
        if n == 1 {
            break;
        }
        let mut next = Vec::with_capacity((n * n / 4) as usize);
        for y in 0..n / 2 {
            for x in 0..n / 2 {
                next.push(
                    [0, 1, n, n + 1]
                        .iter()
                        .map(|d| moments[(y * 2 * n + x * 2 + d) as usize])
                        .sum::<f32>()
                        * 0.25,
                );
            }
        }
        moments = next;
        n /= 2;
    }
    let mut image = Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0; (SIZE * SIZE * 4) as usize],
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
    for opening in facades.into_iter().flat_map(|f| &f.openings) {
        let Some(program) = &opening.glazing else {
            continue;
        };
        let mut recipe = glass::GlassRecipe::sample(program.installation, Surface::Glass);
        recipe.thickness_m = program.thickness_m;
        let mut material = StandardMaterial::default();
        recipe.apply(&mut material, quality);
        if (1..=5).contains(&program.treatment) {
            material.metallic_roughness_texture =
                Some(images.add(roughness_image(program, opening, recipe.roughness)));
            material.perceptual_roughness = 1.;
        }
        result.insert(program.identity, materials.add(material));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn treatments_are_metric_bounded_and_replayable() {
        let mut treatments = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let p = GlazingProgram::sample(seed, 0, true);
            p.validate().unwrap();
            treatments.insert(p.treatment);
            assert_eq!(p, GlazingProgram::sample(seed, 0, true));
            assert_eq!(p.band_m, GlazingProgram::sample(seed, 3, true).band_m);
            assert_eq!(
                p.thickness_m,
                GlazingProgram::sample(seed, 1, true).thickness_m
            );
            for y in 0..100 {
                let f = p.frost(Vec2::new(0.13, y as f32 * 0.04));
                assert!((0.0..=1.).contains(&f));
            }
        }
        assert_eq!(treatments.len(), 7);
    }
    #[test]
    fn roughness_pyramid_keeps_energy_and_full_chain() {
        let mut p = GlazingProgram::sample(21, 0, false);
        p.treatment = 2;
        let opening = WindowOpening {
            min: Vec2::ZERO,
            max: Vec2::new(2., 3.),
            columns: 1,
            transoms: Vec::new(),
            transom: 0.,
            operable: false,
            glazing: Some(p.clone()),
        };
        let image = roughness_image(&p, &opening, 0.025);
        assert_eq!(image.texture_descriptor.mip_level_count, 8);
        let bytes = image.data.unwrap();
        assert_eq!(
            bytes.len(),
            (0..8).map(|i| 4 * (128usize >> i).pow(2)).sum::<usize>()
        );
        let mean = bytes[..128 * 128 * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| (b[1] as f32 / 255.).powi(4))
            .sum::<f32>()
            / (128. * 128.);
        let tail = (bytes[bytes.len() - 3] as f32 / 255.).powi(4);
        assert!((mean - tail).abs() < 0.001);
    }
}
