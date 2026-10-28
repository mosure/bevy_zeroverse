//! Seeded documents: layout and ink live in one filtered map, never raised lines.
use super::{raster::Canvas, *};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PaperProgram {
    /// Plain, ruled, squared, dotted, agenda, sketched notes.
    pub layout: u8,
    pub margin: [i32; 4],
    pub spacing: usize,
    pub vertical_spacing: usize,
    pub ruling_strength: f32,
    pub margin_line: bool,
    pub writing: f32,
    pub slant: f32,
    pub columns: usize,
    pub paper: [u8; 3],
    pub ink: [u8; 3],
    pub roughness: f32,
}
pub fn parameters(seed: u64) -> PaperProgram {
    let mut r = stream(seed, 864);
    let shade = r.random_range(231_u8..253);
    let paper = match r.random_range(0..8) {
        0 => [shade, shade, (shade as i32 - r.random_range(25..55)) as u8],
        1 => [(shade as i32 - 15) as u8, shade, (shade as i32 - 8) as u8],
        2 => [shade, (shade as i32 - 17) as u8, (shade as i32 - 10) as u8],
        _ => [
            shade,
            shade.saturating_sub(r.random_range(0..5)),
            shade.saturating_sub(r.random_range(2..12)),
        ],
    };
    PaperProgram {
        layout: r.random_range(0..6),
        margin: [
            r.random_range(10..42),
            r.random_range(10..40),
            r.random_range(10..35),
            r.random_range(10..35),
        ],
        spacing: r.random_range(8..23),
        vertical_spacing: r.random_range(8..25),
        ruling_strength: r.random_range(0.10..0.36),
        margin_line: r.random_bool(0.45),
        writing: if r.random_bool(0.25) {
            0.
        } else {
            r.random_range(0.12..0.94)
        },
        slant: r.random_range(-0.30..0.30),
        columns: if r.random_bool(0.3) { 2 } else { 1 },
        paper,
        ink: [[50, 63, 77], [40, 61, 116], [30, 79, 66], [83, 58, 48]][r.random_range(0..4)],
        roughness: r.random_range(0.82..0.98),
    }
}
pub(crate) fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut r = stream(seed, 0x5041504552575249);
    let mut c = Canvas::new(p.paper);
    let [left, top, right, bottom] = p.margin;
    let right = 256 - right;
    let bottom = 256 - bottom;
    let rule = std::array::from_fn(|i| {
        (p.paper[i] as f32 * (1. - p.ruling_strength) + p.ink[i] as f32 * p.ruling_strength).round()
            as u8
    });
    if matches!(p.layout, 1 | 2 | 4) {
        for y in (top..bottom).step_by(p.spacing) {
            c.rect(left, y, right - left, 1, rule);
        }
    }
    if p.layout == 2 || p.layout == 3 {
        for x in (left..right).step_by(p.vertical_spacing) {
            if p.layout == 2 {
                c.rect(x, top, 1, bottom - top, rule);
            } else {
                for y in (top..bottom).step_by(p.spacing) {
                    c.rect(x, y, 1, 1, rule);
                }
            }
        }
    }
    if p.margin_line && matches!(p.layout, 1 | 2) {
        c.rect(
            (left - 7).max(2),
            top / 2,
            1,
            bottom - top / 2,
            [205, 162, 158],
        );
    }
    c.clipped([left, top, right - left, bottom - top], |c| {
        let col_width = (right - left) / p.columns as i32;
        for column in 0..p.columns {
            let origin = left + column as i32 * col_width;
            for baseline in (top + p.spacing as i32 - 2..bottom).step_by(p.spacing) {
                if !r.random_bool(p.writing as f64) {
                    continue;
                }
                let mut x = origin + r.random_range(0..8);
                if p.layout == 4 {
                    let size = (p.spacing as i32 / 3).max(3);
                    for (a, b) in [
                        ([x, baseline - size], [x + size, baseline - size]),
                        ([x + size, baseline - size], [x + size, baseline]),
                        ([x + size, baseline], [x, baseline]),
                        ([x, baseline], [x, baseline - size]),
                    ] {
                        c.line(a.map(|v| v as f32), b.map(|v| v as f32), 1, rule);
                    }
                    x += size + 5;
                }
                let end = origin + col_width - r.random_range(8..(col_width / 3).max(9));
                while x + 8 < end {
                    let word = r.random_range(8..30).min(end - x);
                    let mut previous = [x as f32, baseline as f32];
                    for dx in (2..=word).step_by(2) {
                        let point = [
                            (x + dx) as f32 + p.slant * r.random_range(0.0..3.0),
                            baseline as f32 + r.random_range(-4.0..0.5),
                        ];
                        c.line(previous, point, 1, p.ink);
                        previous = point;
                    }
                    x += word + r.random_range(4..10);
                }
            }
        }
        if p.layout == 5 && p.writing > 0. {
            for _ in 0..r.random_range(1..=4) {
                let center = [
                    r.random_range((left + 18) as f32..(right - 18) as f32),
                    r.random_range((top + 18) as f32..(bottom - 18) as f32),
                ];
                let radius = [r.random_range(10.0..28.0), r.random_range(6.0..18.0)];
                c.ring(center, radius, p.ink);
                c.line(
                    [center[0] + radius[0], center[1]],
                    [
                        center[0] + radius[0] + r.random_range(9.0..35.0),
                        center[1] + r.random_range(-12.0..12.0),
                    ],
                    1,
                    p.ink,
                );
            }
        }
    });
    c.0
}
pub(super) fn prepare(seed: u64) -> Image {
    let mut texture = mip_image(pixels(seed), 256, MapType::Color);
    texture.sampler = ImageSampler::linear();
    texture
}
pub(super) fn apply_prepared(
    seed: u64,
    texture: Image,
    mat: &mut StandardMaterial,
    images: &mut impl AssetStore<Image>,
) {
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(images.add(texture));
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
    mat.perceptual_roughness = parameters(seed).roughness;
    mat.metallic = 0.;
}
pub(super) fn apply(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    apply_prepared(seed, prepare(seed), mat, images);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paper_content_is_independent_of_coordinated_furniture_finish() {
        let slots: std::collections::BTreeSet<_> = (0..128)
            .map(|seed| variants::object_slot(Surface::PrintedPaper, seed, Some(42)))
            .collect();
        assert_eq!(slots.len(), variants::COUNT);
        assert!(slots.into_iter().all(|slot| slot < variants::COUNT));
    }
    #[test]
    fn documents_vary_layout_spacing_content_and_keep_matte_filtered_paper() {
        use std::collections::BTreeSet;
        let mut layouts = BTreeSet::new();
        let mut spacing = BTreeSet::new();
        let mut colors = BTreeSet::new();
        let mut maps = BTreeSet::new();
        let mut blank = 0;
        for seed in 0..64 {
            let p = parameters(seed);
            assert_eq!(p, parameters(seed));
            layouts.insert(p.layout);
            spacing.insert((p.spacing, p.vertical_spacing));
            colors.insert(p.paper);
            blank += usize::from(p.writing == 0.);
            let bytes = pixels(seed);
            assert_eq!(bytes, pixels(seed));
            assert!(bytes.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
            maps.insert(bytes);
            let mut images = Assets::default();
            let mut material = StandardMaterial::default();
            apply(seed, &mut material, &mut images);
            let image = images
                .get(material.base_color_texture.as_ref().unwrap())
                .unwrap();
            assert_eq!(image.texture_descriptor.mip_level_count, 9);
            assert!((0.82..0.98).contains(&material.perceptual_roughness));
            assert_eq!(material.metallic, 0.);
        }
        assert_eq!(layouts.len(), 6);
        assert!(spacing.len() > 45 && colors.len() > 50 && maps.len() > 60 && blank > 8);
    }
}
