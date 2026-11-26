//! Object-local swatches for plated soup. Pigment and dielectric roughness
//! describe the stirred liquid; highlights come from the actual room lighting.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

pub(crate) const INTRODUCED: u32 = 265;
pub(crate) const SLOT: usize = 105;
pub(crate) const LABEL: &str = "food#finish105#uv_local";
pub(crate) const HERB_SLOT: usize = 107;
pub(crate) const HERB_LABEL: &str = "food#finish107";
const TILE: usize = 64;
const SIDE: usize = TILE * 2;
const MIPS: u32 = 5;
const GUTTER: f32 = 8.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Recipe {
    GoldenStock,
    Tomato,
    Squash,
    Pea,
}

impl Recipe {
    pub(crate) const ALL: [Self; 4] = [Self::GoldenStock, Self::Tomato, Self::Squash, Self::Pea];

    fn corner(self) -> Vec2 {
        Vec2::new((self as usize % 2) as f32, (self as usize / 2) as f32) * TILE as f32
    }

    pub(crate) fn uv(self, local: Vec2) -> [f32; 2] {
        ((self.corner() + GUTTER + local * (TILE as f32 - 2. * GUTTER)) / SIDE as f32).to_array()
    }

    // sRGB endpoints for the body of the soup, and its oil or cream.
    fn palette(self) -> ([u8; 3], [u8; 3], [u8; 3]) {
        match self {
            Self::GoldenStock => ([111, 67, 28], [196, 147, 67], [230, 184, 87]),
            Self::Tomato => ([129, 39, 22], [215, 90, 47], [250, 197, 137]),
            Self::Squash => ([164, 91, 33], [229, 173, 76], [249, 218, 165]),
            Self::Pea => ([66, 86, 42], [155, 161, 81], [225, 211, 152]),
        }
    }
}

struct Batch {
    seed: u64,
    low: Vec3,
    high: Vec3,
    film: Vec3,
    cream: bool,
    richness: f32,
    roughness: f32,
    // Oil beads coalesce into lenses of different sizes. Keep their pattern
    // independent of the broad mixing field and of garnish positions.
    drops: [(Vec2, f32); 9],
}

impl Batch {
    fn new(recipe: Recipe, seed: u64) -> Self {
        let mut r = stream(seed, 0x534f_5550_4d49_5845 ^ recipe as u64);
        let (low, high, film) = recipe.palette();
        let rgb = |c: [u8; 3]| Vec3::from_array(c.map(f32::from));
        Self {
            seed: r.random(),
            low: rgb(low) * r.random_range(0.83..1.08),
            high: rgb(high) * r.random_range(0.91..1.06),
            film: rgb(film),
            cream: recipe != Recipe::GoldenStock && r.random_bool(0.76),
            richness: r.random_range(0.32..0.91),
            roughness: if recipe == Recipe::GoldenStock {
                r.random_range(0.14..0.23)
            } else {
                r.random_range(0.23..0.38)
            },
            drops: std::array::from_fn(|_| {
                (
                    Vec2::new(r.random(), r.random()),
                    r.random_range(0.018..0.083),
                )
            }),
        }
    }

    fn texel(&self, p: Vec2) -> ([u8; 4], [u8; 4]) {
        let noise = |p: Vec2, x, y, salt| periodic_noise(p.x, p.y, x, y, self.seed ^ salt);
        let a = noise(p, 3, 3, 1);
        let b = noise(p, 3, 4, 2);
        let q = p + Vec2::new(b - 0.5, a - 0.5) * 0.29;
        let mixed = noise(q, 4, 3, 3);
        let body = (0.19 + 0.53 * mixed + 0.19 * a + 0.09 * noise(q, 17, 19, 4)).clamp(0., 1.);
        let film = if self.cream {
            // A folded ribbon thins and breaks as it mixes into the purée.
            let band = (1. - (mixed - 0.53).abs() / (0.05 + b * 0.12)).clamp(0., 1.);
            band * band * (0.28 + 0.72 * a)
        } else {
            self.drops
                .iter()
                .map(|&(center, radius)| {
                    (1. - p.distance_squared(center) / (radius * radius)).clamp(0., 1.)
                })
                .fold(0.0_f32, f32::max)
        } * self.richness;
        let color = self.low.lerp(self.high, body).lerp(self.film, film);
        let rough = self.roughness + (1. - body) * 0.055 - film * 0.13;
        let [r, g, b] = color.to_array().map(|v| v.round().clamp(0., 255.) as u8);
        ([r, g, b, 255], [255, (rough * 255.).round() as u8, 0, 255])
    }
}

fn maps(seed: u64) -> [Image; 2] {
    let mut albedo = vec![0; SIDE * SIDE * 4];
    let mut orm = vec![0; albedo.len()];
    for recipe in Recipe::ALL {
        let batch = Batch::new(recipe, seed);
        let corner = recipe.corner().as_uvec2();
        for y in 0..TILE {
            for x in 0..TILE {
                let p = (Vec2::new(x as f32, y as f32) + 0.5) / TILE as f32;
                let (color, data) = batch.texel(p);
                let i = ((corner.y as usize + y) * SIDE + corner.x as usize + x) * 4;
                albedo[i..i + 4].copy_from_slice(&color);
                orm[i..i + 4].copy_from_slice(&data);
            }
        }
    }
    [
        swatch_image(albedo, SIDE, MIPS, MapType::Color),
        swatch_image(orm, SIDE, MIPS, MapType::Data),
    ]
}

pub(super) fn material(seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let [color, orm] = maps(seed);
    StandardMaterial {
        base_color_texture: Some(images.add(color)),
        metallic_roughness_texture: Some(images.add(orm)),
        perceptual_roughness: 1.,
        metallic: 0.,
        ior: 1.333,
        reflectance: 0.356,
        ..default()
    }
}

pub(super) fn herb(seed: u64) -> StandardMaterial {
    let mut r = stream(seed, 0x4845_5242_4c45_4146);
    StandardMaterial {
        base_color: Color::srgb(
            r.random_range(0.16..0.28),
            r.random_range(0.30..0.43),
            r.random_range(0.09..0.18),
        ),
        perceptual_roughness: r.random_range(0.44..0.62),
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_has_separate_seeded_pigment_and_roughness_and_protected_mips() {
        let [color, roughness] = maps(76142);
        let [same_color, same_rough] = maps(76142);
        assert_eq!(color.data, same_color.data);
        assert_eq!(roughness.data, same_rough.data);
        let [other, other_rough] = maps(76176);
        assert_ne!(color.data, other.data);
        assert_ne!(roughness.data, other_rough.data);
        assert_eq!(
            color.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            roughness.texture_descriptor.format,
            TextureFormat::Rgba8Unorm
        );
        for image in [&color, &roughness] {
            assert_eq!(image.texture_descriptor.mip_level_count, MIPS);
            assert_eq!(
                image.data.as_ref().unwrap().len(),
                (0..MIPS).map(|m| 4 * (SIDE >> m).pow(2)).sum::<usize>()
            );
            let ImageSampler::Descriptor(s) = &image.sampler else {
                panic!("missing sampler")
            };
            assert_eq!(s.anisotropy_clamp, 1);
            assert_eq!(s.lod_max_clamp, (MIPS - 1) as f32);
        }
        for recipe in Recipe::ALL {
            for m in 0..MIPS {
                let edge = (SIDE >> m) as f32;
                let low = recipe.corner() / (1 << m) as f32;
                let high = low + (TILE >> m) as f32;
                assert!((Vec2::from_array(recipe.uv(Vec2::ZERO)) * edge - 0.5)
                    .cmpge(low)
                    .all());
                assert!((Vec2::from_array(recipe.uv(Vec2::ONE)) * edge + 0.5)
                    .cmple(high)
                    .all());
            }
            let c = color.data.as_ref().unwrap();
            let r = roughness.data.as_ref().unwrap();
            let corner = recipe.corner().as_uvec2();
            let mut extrema = [(255u8, 0u8); 2];
            for y in 9..TILE - 9 {
                for x in 9..TILE - 9 {
                    let i = 4 * ((corner.y as usize + y) * SIDE + corner.x as usize + x);
                    assert_eq!([r[i], r[i + 2], r[i + 3]], [255, 0, 255]);
                    assert!((12..=125).contains(&r[i + 1]));
                    for (range, v) in extrema.iter_mut().zip([c[i], r[i + 1]]) {
                        range.0 = range.0.min(v);
                        range.1 = range.1.max(v);
                    }
                }
            }
            assert!(
                extrema[0].1 - extrema[0].0 > 18 && extrema[1].1 - extrema[1].0 > 5,
                "{recipe:?} {extrema:?}"
            );
        }
    }
}
