//! Pigmented decorative paper under an embossed resin wear layer. The printed
//! decor and the press plate have independent fields and physical scales.
use super::*;

pub(super) struct Sheet {
    seed: u64,
    pigment: Vec3,
    mottling: f32,
    flecks: f32,
    period: Vec2,
    emboss_cells: [u32; 2],
    relief: f32,
    roughness: f32,
    gloss: f32,
}

impl Sheet {
    pub(super) fn sample(seed: u64, slot: usize) -> Self {
        let mut rng = stream(seed, 0x4c41_4d49_4e41_5445 ^ slot as u64);
        let lightness = rng.random_range(0.18..0.88);
        let (hue, chroma) = if rng.random_bool(0.16) {
            (rng.random_range(0.0..360.), rng.random_range(0.14..0.48))
        } else {
            (rng.random_range(16.0..77.), rng.random_range(0.006..0.14))
        };
        let color = Color::hsl(hue, chroma, lightness).to_srgba();
        let gloss = rng.random::<f32>();
        Self {
            seed: rng.random(),
            pigment: Vec3::new(color.red, color.green, color.blue),
            mottling: rng.random_range(0.003..0.032),
            flecks: rng.random_range(0.0f32..1.).powi(2) * rng.random_range(-0.14..0.12),
            period: Vec2::new(rng.random_range(0.075..0.18), rng.random_range(0.07..0.16)),
            emboss_cells: [rng.random_range(11..31), rng.random_range(17..31)],
            relief: (1. - gloss * 0.78) * rng.random_range(0.000035..0.00017),
            roughness: 0.58 - 0.30 * gloss,
            gloss,
        }
    }

    fn texel(&self, p: Vec2) -> (Vec3, f32, f32) {
        let decor = periodic_noise(p.x, p.y, 7, 11, self.seed);
        let press = periodic_noise(
            p.x,
            p.y,
            self.emboss_cells[0],
            self.emboss_cells[1],
            self.seed ^ 1,
        );
        let grain = periodic_noise(p.x, p.y, 29, 31, self.seed ^ 2);
        let fleck = ((grain - 0.65) * 4.).clamp(0., 1.);
        let height = 0.74 * press + 0.26 * grain;
        (
            self.pigment + Vec3::splat((decor - 0.5) * self.mottling + self.flecks * fleck),
            height * self.relief,
            self.roughness + (height - 0.5) * 0.14,
        )
    }

    pub(super) fn material(&self, images: &mut impl AssetStore<Image>) -> StandardMaterial {
        sheet_material::<64>(
            images,
            self.period,
            self.gloss * 0.18,
            self.roughness,
            |p| self.texel(p),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printed_decor_and_resin_relief_replay_with_filtered_dielectric_maps() {
        let mut colored = 0;
        let mut darkest = 1.0_f32;
        let mut lightest = 0.0_f32;
        for seed in 0..256 {
            let s = Sheet::sample(seed, FIRST);
            let color = s.pigment;
            colored += usize::from(color.max_element() - color.min_element() > 0.16);
            darkest = darkest.min(color.max_element());
            lightest = lightest.max(color.min_element());
            let p = Vec2::new(0.38, 0.64);
            let (c, h, r) = s.texel(p);
            assert_eq!((c, h, r), Sheet::sample(seed, FIRST).texel(p));
            let (tile, height, rough) = s.texel(p + Vec2::new(1., -2.));
            assert!(
                c.distance(tile) < 1e-5 && (h - height).abs() < 1e-8 && (r - rough).abs() < 1e-5
            );
            assert!((0.0..0.00018).contains(&h) && (0.20..0.68).contains(&r));
        }
        assert!((12..75).contains(&colored) && darkest < 0.24 && lightest > 0.82);
        let mut images = Assets::<Image>::default();
        let material = Sheet::sample(4, FIRST).material(&mut images);
        assert_eq!(images.len(), 3);
        assert_eq!(material.metallic, 0.);
        for (index, handle) in [
            material.base_color_texture,
            material.normal_map_texture,
            material.metallic_roughness_texture,
        ]
        .into_iter()
        .enumerate()
        {
            let im = images.get(handle.unwrap().id()).unwrap();
            assert_eq!(im.texture_descriptor.mip_level_count, 7);
            assert_eq!(im.texture_descriptor.format.is_srgb(), index == 0);
            let data = im.data.as_ref().unwrap();
            if index == 1 {
                assert!(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|p| p[0].abs_diff(128) > 1 || p[1].abs_diff(128) > 1));
            } else if index == 2 {
                assert!(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| [p[0], p[2], p[3]] == [255, 0, 255]));
            }
        }
    }
}
