//! Shared, seeded human microstructure; actor tint and UVs vary independently.
use super::*;
use crate::scene::procedural_indoor::preparation::AssetStore;

/// Shared, neutral microstructure. Human pigmentation and wardrobe colours are
/// applied separately, never multiplied by a furniture upholstery palette.
pub(super) fn maps(
    seed: u64,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
) -> [Handle<StandardMaterial>; 3] {
    use rand::Rng;
    let mut rng = crate::scene::procedural_indoor::layout::stream(seed, 0x48554d414e4d4150);
    let warp = rng.random_range(18..41) as f32;
    let weft = rng.random_range(14..37) as f32;
    let twill = rng.random_range(0.0..1.0_f32);
    let strand = rng.random_range(32..81) as u32;
    [0, 1, 2].map(|kind| {
        let mut albedo = Vec::with_capacity(256 * 256 * 4);
        let mut normal = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let u = x as f32 / 256.0;
                let v = y as f32 / 256.0;
                let phase = std::f32::consts::TAU;
                let (shade, nx, ny) = match kind {
                    0 => (
                        0.96 + 0.025
                            * ((1.0 - twill) * (u * phase * warp).sin() * (v * phase * weft).sin()
                                + twill * ((u * warp + v * weft) * phase).sin()),
                        0.075 * (u * phase * warp).cos(),
                        0.075 * (v * phase * weft).cos(),
                    ),
                    1 => (
                        0.985 + 0.01 * periodic_noise(u, v, 32, 32, seed.wrapping_add(919)),
                        0.025 * (u * phase * 40.0).sin(),
                        0.025 * (v * phase * 40.0).sin(),
                    ),
                    _ => (
                        0.85 + 0.14 * periodic_noise(u, v, strand, 2, seed.wrapping_add(813)),
                        0.18 * (u * phase * strand as f32).sin(),
                        0.01 * (v * phase * 2.0).cos(),
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
                10.0
            } else {
                24.0
            })),
            perceptual_roughness: if kind == 2 { 0.60 } else { 0.85 },
            // Bevy 0.19's LOAD_PREPASS_NORMALS path skips anisotropy_T/B
            // initialization. Enabling anisotropy with the normal prepass then
            // yields non-finite specular radiance (white hair after tonemapping).
            // Use an aggregate isotropic fibre lobe until that path supplies a
            // valid tangent frame; keep the directional microstructure/geometry.
            anisotropy_strength: 0.0,
            ..default()
        })
    })
}
