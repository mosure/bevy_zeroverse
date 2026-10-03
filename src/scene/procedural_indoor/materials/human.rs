//! Shared, seeded human microstructure; actor tint and UVs vary independently.
use super::*;
use crate::scene::procedural_indoor::preparation::AssetStore;

/// Shared, neutral microstructure. Human pigmentation and wardrobe colours are
/// applied separately, never multiplied by a furniture upholstery palette.
pub(super) fn maps(
    seed: u64,
    mut cloth: StandardMaterial,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
) -> [Handle<StandardMaterial>; 3] {
    use rand::Rng;
    let mut rng = crate::scene::procedural_indoor::layout::stream(seed, 0x48554d414e4d4150);
    // Preserve the skin/hair RNG stream; wardrobe no longer uses sinusoidal maps.
    let _ = rng.random_range(18..41);
    let _ = rng.random_range(14..37);
    let _ = rng.random_range(0.0..1.0_f32);
    let strand = rng.random_range(32..81) as u32;
    cloth.base_color = Color::WHITE;
    // Anny's atlas is normalized. A two-metre UV reference is modulated per actor.
    cloth.uv_transform *= bevy::math::Affine2::from_scale(Vec2::splat(2.));
    let cloth = materials.add(cloth);
    [0, 1, 2].map(|kind| {
        if kind == 0 {
            return cloth.clone();
        }
        let mut albedo = Vec::with_capacity(256 * 256 * 4);
        let mut normal = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let u = x as f32 / 256.0;
                let v = y as f32 / 256.0;
                let phase = std::f32::consts::TAU;
                let (shade, nx, ny) = match kind {
                    1 => {
                        // Irregular pores, not a regular embossed grid. Derive
                        // relief from the same field used by pigmentation.
                        let pore = |u, v| periodic_noise(u, v, 47, 53, seed.wrapping_add(919));
                        (
                            0.985 + 0.012 * (pore(u, v) - 0.5),
                            (pore(u + 1.0 / 256.0, v) - pore(u - 1.0 / 256.0, v)) * 0.14,
                            (pore(u, v + 1.0 / 256.0) - pore(u, v - 1.0 / 256.0)) * 0.14,
                        )
                    }
                    _ => (
                        0.86 + 0.24
                            * (periodic_noise(u, v, strand, 3, seed.wrapping_add(813)) - 0.5),
                        0.11 * (u * phase * strand as f32 + 0.12 * (v * phase).sin()).sin(),
                        0.005 * (v * phase * 2.0).cos(),
                    ),
                };
                let n = Vec3::new(nx, ny, 1.0).normalize();
                let c = (shade * 255.0) as u8;
                albedo.extend([c, c, c, 255]);
                normal.extend([
                    (n.x * 127.0 + 128.0) as u8,
                    (n.y * 127.0 + 128.0) as u8,
                    (n.z * 127.0 + 128.0) as u8,
                    255,
                ]);
            }
        }
        materials.add(StandardMaterial {
            base_color_texture: Some(images.add(mip_image(albedo, 256, MapType::Color))),
            normal_map_texture: Some(images.add(mip_image(normal, 256, MapType::Normal))),
            // Body UV islands cover several metres; the fibre/pores remain fine.
            uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(if kind == 2 {
                25.0
            } else {
                24.0
            })),
            perceptual_roughness: if kind == 2 { 0.60 } else { 0.85 },
            // The indoor shading integration initializes this tangent frame
            // with and without a normal prepass. V follows the groom fibres.
            anisotropy_strength: if kind == 2 { 0.35 } else { 0.0 },
            anisotropy_rotation: std::f32::consts::FRAC_PI_2,
            ..default()
        })
    })
}
