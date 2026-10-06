//! Shared, seeded human microstructure; actor tint and UVs vary independently.
use super::*;
use crate::scene::procedural_indoor::preparation::AssetStore;

/// One neutral loop-knit atlas shared by the room's jersey/knit garments.
pub(super) fn knit(
    seed: u64,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
) -> Handle<StandardMaterial> {
    let mut recipe =
        program::sample(seed.wrapping_add(0x4b4e4954))[Surface::Fabric as usize].clone();
    recipe.layers = None;
    let textile = recipe.textile.as_mut().unwrap();
    textile.knit = true;
    textile.yarn_tint = [[1.; 3]; 2];
    textile.lustre *= 0.35;
    recipe.period_m = 0.0015 * textile.yarns[1] as f32;
    recipe.relief_m = 0.00010 + textile.crimp * 0.0002;
    recipe.roughness = 0.83;
    let maps = recipe.maps(0).map(|map| images.add(map));
    let mut material = StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(maps[0].clone()),
        normal_map_texture: Some(maps[1].clone()),
        metallic_roughness_texture: Some(maps[2].clone()),
        occlusion_texture: Some(maps[2].clone()),
        uv_transform: bevy::math::Affine2::from_scale(recipe.period_uv().recip() * 2.),
        ..default()
    };
    recipe.apply_pbr(&mut material);
    materials.add(material)
}

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
        // Finite differences previously re-evaluated this same field five or
        // six times per texel. A one-pixel halo retains the exact x/256 and
        // y/256 inputs, including -1 and 256; wrapping indices instead would
        // subtly change rounding inside the warped hair field at atlas seams.
        let field = |u, v| {
            if kind == 1 {
                periodic_noise(u, v, 47, 53, seed.wrapping_add(919))
            } else {
                let warp = periodic_noise(u, v, 7, 5, seed.wrapping_add(729));
                0.65 * periodic_noise(
                    u + (warp - 0.5) * 0.035,
                    v,
                    strand,
                    5,
                    seed.wrapping_add(813),
                ) + 0.35 * periodic_noise(u, v, strand * 2 + 1, 11, seed.wrapping_add(1183))
            }
        };
        let mut halo = Vec::with_capacity(258 * 258);
        for y in -1..=256 {
            for x in -1..=256 {
                halo.push(field(x as f32 / 256.0, y as f32 / 256.0));
            }
        }
        let mut albedo = Vec::with_capacity(256 * 256 * 4);
        let mut normal = Vec::with_capacity(256 * 256 * 4);
        let mut packed = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let u = x as f32 / 256.0;
                let v = y as f32 / 256.0;
                let center = (y + 1) * 258 + x + 1;
                let f = halo[center];
                let dx = halo[center + 1] - halo[center - 1];
                let dy = halo[center + 258] - halo[center - 258];
                let (shade, nx, ny, roughness) = match kind {
                    1 => {
                        // Irregular pores, not a regular embossed grid. Derive
                        // relief from the same field used by pigmentation.
                        (
                            0.985 + 0.012 * (f - 0.5),
                            dx * 0.14,
                            dy * 0.14,
                            0.90 + f * 0.10,
                        )
                    }
                    _ => {
                        let tone = periodic_noise(u, v, 5, 9, seed.wrapping_add(1927));
                        (
                            0.94 + 0.18 * (f - 0.5) + 0.12 * (tone - 0.5),
                            dx * 0.11,
                            dy * 0.025,
                            0.76 + f * 0.20,
                        )
                    }
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
                packed.extend([255, (roughness * 255.0) as u8, 0, 255]);
            }
        }
        materials.add(StandardMaterial {
            base_color_texture: Some(images.add(mip_image(albedo, 256, MapType::Color))),
            normal_map_texture: Some(images.add(mip_image(normal, 256, MapType::Normal))),
            metallic_roughness_texture: Some(images.add(mip_image(packed, 256, MapType::Data))),
            // Body UV islands cover several metres; the fibre/pores remain fine.
            uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(if kind == 2 {
                25.0
            } else {
                24.0
            })),
            perceptual_roughness: if kind == 2 { 0.60 } else { 0.85 },
            // The indoor shading integration initializes this tangent frame
            // with and without a normal prepass. V follows the groom fibres.
            anisotropy_strength: if kind == 2 { 0.55 } else { 0.0 },
            anisotropy_rotation: std::f32::consts::FRAC_PI_2,
            ..default()
        })
    })
}

#[cfg(test)]
mod replay_tests;
