//! Frozen scalar finish oracle, including every roughness mip and PBR scalar.
use super::*;
impl DisplayFinish {
    fn reference_apply(
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
#[test]
fn prepared_finishes_replay_scalar_maps_and_materials() {
    let content: Vec<u8> = (0..256 * 256)
        .flat_map(|i| [i as u8, (i / 256) as u8, (i * 73) as u8, 255])
        .collect();
    for seed in [0, 1, 7, 202, 600, 601, u64::MAX] {
        for board in [false, true] {
            for smudges in [0, 1, 5] {
                let mut finish = DisplayFinish::sample(seed, board);
                finish.smudges = smudges;
                let mut a = StandardMaterial {
                    base_color: Color::srgb(0.2, 0.4, 0.6),
                    metallic: 0.7,
                    diffuse_transmission: 0.16,
                    ..default()
                };
                let mut b = a.clone();
                let mut ia = Assets::<Image>::default();
                let mut ib = Assets::<Image>::default();
                finish.reference_apply(seed, board, &content, &mut a, &mut ia);
                finish.apply(seed, board, &content, &mut b, &mut ib);
                super::super::variants::content::assert_materials_equal(&a, &ia, &b, &ib);
            }
        }
    }
}
