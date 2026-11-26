//! Shared fruit peel in sRGB, tangent-space normal and linear roughness maps.
//! Each swatch has a protected gutter at every sampled mip, including on WebGPU.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::sync::OnceLock;

pub(crate) const SLOT: usize = 811;
pub(crate) const LABEL: &str = "food#finish811#uv_local";
const TILE: usize = 32;
const SIDE: usize = 4 * TILE;
const MIPS: u32 = 4;
const GUTTER: f32 = 4.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Skin {
    RedApple,
    BlushApple,
    GreenApple,
    GoldenApple,
    GreenPear,
    RussetPear,
    RedPear,
    YellowPear,
    Tangerine,
    Orange,
    Lemon,
    Lime,
    VioletPlum,
    RedPlum,
    BluePlum,
    Greengage,
}

impl Skin {
    const ALL: [Self; 16] = [
        Self::RedApple,
        Self::BlushApple,
        Self::GreenApple,
        Self::GoldenApple,
        Self::GreenPear,
        Self::RussetPear,
        Self::RedPear,
        Self::YellowPear,
        Self::Tangerine,
        Self::Orange,
        Self::Lemon,
        Self::Lime,
        Self::VioletPlum,
        Self::RedPlum,
        Self::BluePlum,
        Self::Greengage,
    ];
    pub(crate) fn chart(self) -> (Vec2, f32) {
        let i = self as usize;
        let corner = Vec2::new((i % 4 * TILE) as f32, (i / 4 * TILE) as f32) + GUTTER;
        (
            corner / SIDE as f32,
            (TILE as f32 - 2. * GUTTER) / SIDE as f32,
        )
    }
    // Two pigment endpoints per cultivar. Ripening, russet, bloom and freckles
    // are sampled across these colours, independently of fruit outline and yaw.
    fn pigment(self) -> ([u8; 3], [u8; 3]) {
        match self {
            Self::RedApple => ([112, 28, 26], [209, 62, 46]),
            Self::BlushApple => ([218, 178, 80], [188, 55, 45]),
            Self::GreenApple => ([110, 146, 49], [176, 194, 79]),
            Self::GoldenApple => ([150, 160, 56], [222, 183, 69]),
            Self::GreenPear => ([107, 124, 50], [180, 174, 73]),
            Self::RussetPear => ([139, 98, 56], [187, 149, 89]),
            Self::RedPear => ([110, 49, 36], [188, 83, 52]),
            Self::YellowPear => ([156, 152, 66], [211, 185, 81]),
            Self::Tangerine => ([200, 86, 22], [244, 143, 40]),
            Self::Orange => ([202, 108, 30], [238, 158, 46]),
            Self::Lemon => ([201, 169, 54], [242, 217, 81]),
            Self::Lime => ([79, 125, 47], [152, 174, 60]),
            Self::VioletPlum => ([75, 46, 71], [127, 97, 123]),
            Self::RedPlum => ([100, 40, 52], [166, 86, 96]),
            Self::BluePlum => ([47, 48, 79], [116, 111, 142]),
            Self::Greengage => ([102, 119, 48], [164, 165, 79]),
        }
    }
}

struct Texel {
    rgb: [u8; 3],
    height: f32,
    rough: u8,
}

fn field(skin: Skin, x: usize, y: usize, lot: usize) -> Texel {
    let core = TILE - 8;
    // The repeated longitude and clamped latitude extend into every gutter.
    let u = ((x + core - 4) % core) as f32 / (core - 1) as f32;
    let v = (y.saturating_sub(4).min(core - 1)) as f32 / (core - 1) as f32;
    let key = 0x5045_454c + (lot as u64) * 5231 + skin as u64 * 53;
    let latitude = (std::f32::consts::PI * v).sin().max(0.);
    let cloud = periodic_noise(u, v, 4, 3, key);
    let detail = periodic_noise(u, v, 17, 19, key + 1);
    let grain = periodic_noise(u, v, 11, 6, key + 2);
    let sun = 0.5 + 0.5 * (std::f32::consts::TAU * (u + cloud * 0.22 + lot as f32 * 0.19)).cos();
    let group = skin as usize / 4;
    let (blend, relief, rough) = match group {
        0 => (
            sun * 0.57 + cloud * 0.30 + grain * 0.13,
            (detail - 0.5) * 0.08,
            0.35 + cloud * 0.11,
        ),
        1 => (
            cloud * 0.62 + sun * 0.27 + grain * 0.11,
            (detail - 0.5) * 0.19,
            0.53 + grain * 0.17,
        ),
        2 => (
            cloud * 0.73 + sun * 0.27,
            -(detail - 0.48).max(0.) * 0.59,
            0.42 + detail * 0.17,
        ),
        _ => (
            cloud * 0.80 + grain * 0.2,
            (detail - 0.5) * 0.045,
            0.43 + cloud * 0.28,
        ),
    };
    let pigment = (blend - 0.5) * latitude + 0.5;
    let dots = if group <= 1 {
        ((detail - 0.69) * 3.8).clamp(0., 0.58) * latitude
    } else {
        0.
    };
    let scar = if group == 2 && !matches!(skin, Skin::Lemon | Skin::Lime) {
        (1. - v * 17.).max(0.) * 0.33
    } else {
        0.
    };
    let (base, ripe) = skin.pigment();
    let speckle_shadow = dots * if group == 1 { 0.63 } else { 0.31 };
    let speckle_light = dots * if group == 0 { 57. } else { 8. };
    let rgb = std::array::from_fn(|i| {
        let c = f32::from(base[i]) + (f32::from(ripe[i]) - f32::from(base[i])) * pigment;
        (c * (1. - speckle_shadow - scar) + speckle_light).clamp(0., 255.) as u8
    });
    Texel {
        rgb,
        height: relief * latitude,
        rough: (rough * 255.) as u8,
    }
}

fn stock(lot: usize) -> &'static [Image; 3] {
    static STOCK: [OnceLock<[Image; 3]>; 4] = [const { OnceLock::new() }; 4];
    STOCK[lot].get_or_init(|| {
        let mut color = vec![0; SIDE * SIDE * 4];
        let mut normal = color.clone();
        let mut orm = color.clone();
        for skin in Skin::ALL {
            let field: Vec<_> = (0..TILE * TILE)
                .map(|n| field(skin, n % TILE, n / TILE, lot))
                .collect();
            let i = skin as usize;
            for y in 0..TILE {
                for x in 0..TILE {
                    let t = &field[y * TILE + x];
                    let dx = (field[y * TILE + (x + 1).min(TILE - 1)].height
                        - field[y * TILE + x.saturating_sub(1)].height)
                        * 0.5;
                    let dy = (field[(y + 1).min(TILE - 1) * TILE + x].height
                        - field[y.saturating_sub(1) * TILE + x].height)
                        * 0.5;
                    let n = Vec3::new(-dx, dy, 1.).normalize();
                    let k = ((i / 4 * TILE + y) * SIDE + (i % 4 * TILE + x)) * 4;
                    color[k..k + 4].copy_from_slice(&[t.rgb[0], t.rgb[1], t.rgb[2], 255]);
                    let [r, g, b] = n.to_array().map(|v| ((v * 0.5 + 0.5) * 255.).round() as u8);
                    normal[k..k + 4].copy_from_slice(&[r, g, b, 255]);
                    orm[k..k + 4].copy_from_slice(&[255, t.rough, 0, 255]);
                }
            }
        }
        [
            swatch_image(color, SIDE, MIPS, MapType::Color),
            swatch_image(normal, SIDE, MIPS, MapType::Normal),
            swatch_image(orm, SIDE, MIPS, MapType::Data),
        ]
    })
}

pub(super) fn material(seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"PeelCoat"));
    let [color, normal, orm] = stock(r.random_range(0..4));
    let tint = r.random_range(0.92..1.);
    StandardMaterial {
        base_color: Color::srgb(
            tint,
            tint * r.random_range(0.975..1.),
            tint * r.random_range(0.94..1.),
        ),
        base_color_texture: Some(images.add(color.clone())),
        normal_map_texture: Some(images.add(normal.clone())),
        metallic_roughness_texture: Some(images.add(orm.clone())),
        metallic: 0.,
        perceptual_roughness: r.random_range(0.88..1.),
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peel_stock_uses_dielectric_linear_maps_and_never_filters_between_cultivars() {
        let [color, normal, orm] = stock(0);
        assert!(std::ptr::eq(stock(0), stock(0)));
        assert_ne!(color.data, stock(1)[0].data);
        assert_eq!(
            color.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        for image in [color, normal, orm] {
            assert_eq!(image.texture_descriptor.mip_level_count, MIPS);
            let bytes: usize = (0..MIPS).map(|m| 4 * (SIDE >> m).pow(2)).sum();
            assert_eq!(image.data.as_ref().unwrap().len(), bytes);
            let ImageSampler::Descriptor(sampler) = &image.sampler else {
                panic!("missing sampler");
            };
            assert_eq!(sampler.lod_max_clamp, (MIPS - 1) as f32);
            assert_eq!(sampler.anisotropy_clamp, 1);
        }
        for image in [normal, orm] {
            assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);
        }
        for skin in Skin::ALL {
            let (origin, span) = skin.chart();
            let tile = Vec2::new((skin as usize % 4) as f32, (skin as usize / 4) as f32);
            for mip in 0..MIPS {
                let side = (SIDE >> mip) as f32;
                let low = tile * (TILE >> mip) as f32;
                let high = low + (TILE >> mip) as f32;
                // Half-texel bilinear footprint, including fractional mip levels.
                assert!((origin * side - 0.5).cmpge(low).all());
                assert!(((origin + span) * side + 0.5).cmple(high).all());
            }
            let (mut low_roughness, mut high_roughness) = (u8::MAX, u8::MIN);
            let mut relief = 0;
            let r = orm.data.as_ref().unwrap();
            let n = normal.data.as_ref().unwrap();
            for y in 4..TILE - 4 {
                for x in 4..TILE - 4 {
                    let k =
                        ((skin as usize / 4 * TILE + y) * SIDE + skin as usize % 4 * TILE + x) * 4;
                    assert_eq!((r[k], r[k + 2], r[k + 3]), (255, 0, 255));
                    low_roughness = low_roughness.min(r[k + 1]);
                    high_roughness = high_roughness.max(r[k + 1]);
                    let v = Vec3::new(n[k] as f32, n[k + 1] as f32, n[k + 2] as f32) / 127.5 - 1.;
                    assert!((v.length() - 1.).abs() < 0.008 && v.z > 0.95);
                    relief += usize::from(n[k] != 128 || n[k + 1] != 128);
                }
            }
            assert!(high_roughness - low_roughness > 9, "{skin:?}");
            assert!(relief > 35, "{skin:?}");
        }
    }
}
