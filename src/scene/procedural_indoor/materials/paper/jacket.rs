//! One unfolded print chart per edition, shared by copies in the same room.
use super::*;

mod artwork;
mod lettering;

pub(crate) const INTRODUCED: u32 = 272;
pub(crate) const FIRST: usize = 2 * variants::COUNT;
const BOOK_EDITIONS: usize = 8;
pub(crate) const END: usize = FIRST + BOOK_EDITIONS + 3;
const BACK_END: i32 = 114;
const FRONT_START: i32 = 142;
pub(crate) const SPINE: [f32; 2] = [BACK_END as f32 / 256., FRONT_START as f32 / 256.];

pub(crate) fn slot(version: u32, edition: u64, notebook: bool) -> Option<usize> {
    if version < INTRODUCED {
        return None;
    }
    let mut r = stream(edition, u64::from_le_bytes(*b"BookWrap"));
    let (chance, slots) = if notebook {
        (0.39, FIRST + BOOK_EDITIONS..END)
    } else {
        (0.76, FIRST..FIRST + BOOK_EDITIONS)
    };
    r.random_bool(chance).then(|| r.random_range(slots))
}

pub(crate) struct Prepared {
    color: Image,
    varnish: Option<Image>,
    roughness: f32,
    coat: f32,
}

fn texture(data: Vec<u8>, side: u32, kind: MapType) -> Image {
    let mut image = match kind {
        MapType::Color => mipmaps::printed_mip_image(data, side),
        _ => mip_image(data, side, kind),
    };
    if let ImageSampler::Descriptor(s) = &mut image.sampler {
        s.address_mode_u = ImageAddressMode::ClampToEdge;
        s.address_mode_v = ImageAddressMode::ClampToEdge;
    }
    image
}

pub(crate) fn prepare(seed: u64, slot: usize) -> Prepared {
    debug_assert!((FIRST..END).contains(&slot));
    let (rgba, accent) = artwork::print(seed, slot >= FIRST + BOOK_EDITIONS);
    let mut stock = stream(seed, u64::from_le_bytes(*b"BookCoat"));
    let (roughness, coat) = match stock.random_range(0..6) {
        0 => (stock.random_range(0.25..0.40), 0.42),
        1 | 2 => (stock.random_range(0.44..0.64), 0.14),
        _ => (stock.random_range(0.72..0.94), 0.),
    };
    // Localized varnish changes the specular lobe over the printed motif only.
    // Filter pigment coverage across each 4x4 footprint before the mip chain.
    let varnish = stock.random_bool(0.29).then(|| {
        let ink = stock.random_range(0.19..0.32_f32).min(roughness * 0.78);
        let mut packed = Vec::with_capacity(64 * 64 * 4);
        for rows in rgba.as_chunks::<{ 256 * 4 * 4 }>().0 {
            for x in 0..64 {
                let coverage = rows
                    .as_chunks::<{ 256 * 4 }>()
                    .0
                    .iter()
                    .map(|row| {
                        row[x * 16..x * 16 + 16]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .filter(|p| p[..3] == accent)
                            .count()
                    })
                    .sum::<usize>() as f32
                    / 16.;
                let r = roughness + (ink - roughness) * coverage;
                packed.extend([255, (255. * r).round() as u8, 0, 255]);
            }
        }
        texture(packed, 64, MapType::Data)
    });
    Prepared {
        color: texture(rgba, 256, MapType::Color),
        varnish,
        roughness,
        coat,
    }
}

impl Prepared {
    pub(crate) fn apply(self, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
        mat.base_color = Color::WHITE;
        mat.base_color_texture = Some(images.add(self.color));
        mat.uv_transform = bevy::math::Affine2::IDENTITY;
        mat.metallic = 0.;
        mat.perceptual_roughness = if self.varnish.is_some() {
            1.
        } else {
            self.roughness
        };
        mat.metallic_roughness_texture = self.varnish.map(|image| images.add(image));
        mat.clearcoat = self.coat;
        mat.clearcoat_perceptual_roughness = self.roughness;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn jacket_editions_are_selective_replayable_and_use_distinct_notebook_stock() {
        let mut lots = [BTreeSet::new(), BTreeSet::new()];
        let mut printed = [0; 2];
        for seed in 0..512 {
            for notebook in [false, true] {
                assert!(slot(INTRODUCED - 1, seed, notebook).is_none());
                if let Some(s) = slot(INTRODUCED, seed, notebook) {
                    assert_eq!(Some(s), slot(INTRODUCED, seed, notebook));
                    assert_eq!(s >= FIRST + BOOK_EDITIONS, notebook);
                    lots[usize::from(notebook)].insert(s);
                    printed[usize::from(notebook)] += 1;
                }
            }
        }
        assert_eq!(
            lots.map(|s| s.len()),
            [BOOK_EDITIONS, END - FIRST - BOOK_EDITIONS]
        );
        assert!(
            (335..430).contains(&printed[0]) && (146..255).contains(&printed[1]),
            "{printed:?}"
        );
    }

    #[test]
    fn jacket_ink_and_spot_varnish_share_a_chart_and_filter_to_the_last_mip() {
        let mut varnished = 0;
        let mut colors = BTreeSet::new();
        for seed in 0..24 {
            let jacket = prepare(seed, FIRST);
            let (pixels, accent) = artwork::print(seed, false);
            let base = &jacket.color.data.as_ref().unwrap()[..256 * 256 * 4];
            assert_eq!(base, pixels);
            colors.insert((accent, pixels[..3].to_vec()));
            assert_eq!(
                jacket.color.texture_descriptor.format,
                TextureFormat::Rgba8UnormSrgb
            );
            assert_eq!(jacket.color.texture_descriptor.mip_level_count, 9);
            assert!((0.25..0.94).contains(&jacket.roughness));
            if let Some(map) = jacket.varnish {
                assert_eq!(map.texture_descriptor.format, TextureFormat::Rgba8Unorm);
                assert_eq!(map.texture_descriptor.mip_level_count, 7);
                let mat = (255. * jacket.roughness).round() as u8;
                let mut shiny = 0;
                for (i, p) in map.data.as_ref().unwrap()[..64 * 64 * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                {
                    let x = (i % 64) * 4;
                    let y = (i / 64) * 4;
                    let has_ink = (y..y + 4).any(|row| {
                        pixels[(256 * row + x) * 4..(256 * row + x + 4) * 4]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .any(|v| v[..3] == accent)
                    });
                    assert!(p[1] <= mat);
                    if !has_ink {
                        assert_eq!(p[1], mat);
                    }
                    if p[1] < mat {
                        shiny += 1;
                    }
                    assert_eq!([p[0], p[2], p[3]], [255, 0, 255]);
                }
                assert!(shiny > 0);
                varnished += 1;
            }
        }
        assert!(colors.len() >= 22 && (2..20).contains(&varnished));
    }
}
