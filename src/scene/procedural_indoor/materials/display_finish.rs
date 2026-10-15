//! Baked display-cover and marker-board finishes. No bump from printed pixels.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

#[derive(Clone, Debug, serde::Serialize)]
pub struct DisplayFinish {
    pub roughness: f32,
    pub clearcoat: f32,
    pub coat_roughness: f32,
    pub haze: f32,
    pub smudges: u32,
}

impl DisplayFinish {
    pub fn sample(seed: u64, board: bool) -> Self {
        let mut rng = stream(seed, 0x46494e4953484449);
        Self {
            roughness: rng.random_range(if board { 0.18..0.41 } else { 0.10..0.37 }),
            clearcoat: rng.random_range(0.02..0.42),
            coat_roughness: rng.random_range(0.08..0.32),
            haze: rng.random_range(0.0..if board { 0.12 } else { 0.06 }),
            smudges: if rng.random_bool(0.6) {
                0
            } else {
                rng.random_range(1..=5)
            },
        }
    }

    pub(super) fn apply(
        &self,
        seed: u64,
        board: bool,
        content: &[u8],
        mat: &mut StandardMaterial,
        images: &mut impl AssetStore<Image>,
    ) {
        let mut rng = stream(seed, 0x534d55444745);
        let marks: Vec<_> = (0..self.smudges)
            .map(|_| {
                [
                    rng.random_range(0.05..0.95),
                    rng.random_range(0.05..0.95),
                    rng.random_range(0.02..0.09),
                    rng.random_range(0.05..0.19),
                ]
            })
            .collect();
        mat.normal_map_texture = None;
        mat.occlusion_texture = None;
        mat.metallic = 0.;
        mat.perceptual_roughness = self.roughness;
        mat.clearcoat = self.clearcoat;
        mat.clearcoat_perceptual_roughness = self.coat_roughness;
        if !board && marks.is_empty() {
            // Uniform cover finishes need no texture allocation or upload.
            mat.metallic_roughness_texture = None;
            mat.perceptual_roughness *= 0.83 + self.haze * 0.25;
            return;
        }
        let mut data = Vec::with_capacity(256 * 256 * 4);
        for (i, pixel) in content.as_chunks::<4>().0.iter().enumerate() {
            let uv = [(i % 256) as f32 / 256., (i / 256) as f32 / 256.];
            let ink = if board {
                1. - f32::from(pixel[0].max(pixel[1]).max(pixel[2])) / 255.
            } else {
                0.
            };
            let mut smudge = 0.0_f32;
            for m in &marks {
                let d = ((uv[0] - m[0]) / m[2]).powi(2) + ((uv[1] - m[1]) / m[3]).powi(2);
                smudge = smudge.max((1. - d).clamp(0., 1.));
            }
            let value = (0.83 + self.haze * 0.25 + smudge * 0.15 - ink * 0.13).clamp(0.65, 1.);
            data.extend([255, (value * 255.).round() as u8, 0, 255]);
        }
        // Relative roughness map; independent of luminance and never metallic.
        mat.metallic_roughness_texture = Some(images.add(filter::flat_roughness(data, 256)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cover_finishes_have_linear_dielectric_mips_and_no_text_normals() {
        for seed in 0..16 {
            for board in [false, true] {
                let mut images = Assets::<Image>::default();
                let mut mat = StandardMaterial::default();
                let content = if board {
                    boards::pixels(seed)
                } else {
                    screens::pixels(seed)
                };
                let p = DisplayFinish::sample(seed, board);
                p.apply(seed, board, &content, &mut mat, &mut images);
                assert_eq!(images.len(), usize::from(board || p.smudges > 0));
                assert_eq!(mat.metallic, 0.);
                assert!(mat.normal_map_texture.is_none());
                let Some(handle) = mat.metallic_roughness_texture.as_ref() else {
                    continue;
                };
                let image = images.get(handle).unwrap();
                assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);
                assert_eq!(image.texture_descriptor.mip_level_count, 9);
                let data = image.data.as_ref().unwrap();
                assert_eq!(
                    data.len(),
                    (0..9).map(|m| (256_usize >> m).pow(2) * 4).sum::<usize>()
                );
                for pixel in data.as_chunks::<4>().0.iter() {
                    assert_eq!([pixel[0], pixel[2], pixel[3]], [255, 0, 255]);
                    assert!((160..=255).contains(&pixel[1]));
                }
            }
        }
    }
}
