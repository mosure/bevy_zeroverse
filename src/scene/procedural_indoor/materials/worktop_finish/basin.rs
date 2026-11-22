//! Pressed stainless steel, vitreous ceramic and mineral-composite bowls.
use super::*;

pub(super) fn material(
    surface: Surface,
    seed: u64,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut r = stream(seed, 0x4241_5349_4e53_544b ^ surface as u64);
    let (value, roughness, variation, texture_scale) = match surface {
        Surface::Metal => (
            r.random_range(0.72..0.85),
            r.random_range(0.21..0.39),
            0.17,
            Vec2::new(2.1, 5.7),
        ),
        Surface::Ceramic => (
            r.random_range(0.77..0.95),
            r.random_range(0.16..0.29),
            0.035,
            Vec2::splat(2.8),
        ),
        _ => (
            r.random_range(0.20..0.73),
            r.random_range(0.37..0.58),
            0.14,
            Vec2::splat(r.random_range(3.0..5.6)),
        ),
    };
    let warmth = r.random_range(-0.026..0.038);
    let tint = Vec3::new(value + warmth, value, value - warmth);
    let metal = surface == Surface::Metal;
    let composite = surface == Surface::Concrete;
    let mut pigment = Vec::new();
    let mut orm = Vec::with_capacity(64 * 64 * 4);
    let key = r.random::<u64>();
    for y in 0..64 {
        for x in 0..64 {
            let u = (x as f32 + 0.5) / 64.;
            let v = (y as f32 + 0.5) / 64.;
            let grain = periodic_noise(u, v, if metal { 4 } else { 25 }, 31, key);
            let grit = periodic_noise(u, v, 29, 27, key ^ 1);
            let finish =
                (roughness + ((grain - 0.5) * 0.76 + (grit - 0.5) * 0.24) * variation) * 255.;
            orm.extend([255, finish.round() as u8, if metal { 255 } else { 0 }, 255]);
            if composite {
                let v = tint + Vec3::splat((grit - 0.5) * 0.050 + (grain - 0.62).max(0.) * 0.12);
                pigment.extend(v.to_array().map(|c| (c.clamp(0., 1.) * 255.).round() as u8));
                pigment.push(255);
            }
        }
    }
    StandardMaterial {
        base_color: if composite {
            Color::WHITE
        } else {
            Color::srgb(tint.x, tint.y, tint.z)
        },
        base_color_texture: composite.then(|| images.add(mip_image(pigment, 64, MapType::Color))),
        metallic_roughness_texture: Some(images.add(mip_image(orm, 64, MapType::Data))),
        uv_transform: bevy::math::Affine2::from_scale(texture_scale),
        metallic: if metal { 1. } else { 0. },
        perceptual_roughness: 1.,
        clearcoat: if surface == Surface::Ceramic {
            0.24
        } else {
            0.
        },
        clearcoat_perceptual_roughness: roughness,
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bowl_material_preserves_substrate_and_only_composite_needs_pigment_map() {
        for s in [Surface::Metal, Surface::Concrete, Surface::Ceramic] {
            let mut images = Assets::<Image>::default();
            let mat = material(s, 64, &mut images);
            let metal = s == Surface::Metal;
            assert_eq!(mat.metallic > 0., metal);
            assert_eq!(mat.clearcoat > 0., s == Surface::Ceramic);
            assert_eq!(mat.base_color_texture.is_some(), s == Surface::Concrete);
            let orm = images
                .get(mat.metallic_roughness_texture.unwrap().id())
                .unwrap();
            assert_eq!(orm.texture_descriptor.mip_level_count, 7);
            assert!(orm
                .data
                .as_ref()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| v[2] == if metal { 255 } else { 0 }));
        }
    }
}
