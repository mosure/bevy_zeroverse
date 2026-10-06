//! Original pre-cache atlas implementation, retained independently for exact replay.
use super::*;

fn reference_maps(
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
        let mut packed = Vec::with_capacity(256 * 256 * 4);
        for y in 0..256 {
            for x in 0..256 {
                let u = x as f32 / 256.0;
                let v = y as f32 / 256.0;
                let (shade, nx, ny, roughness) = match kind {
                    1 => {
                        // Irregular pores, not a regular embossed grid. Derive
                        // relief from the same field used by pigmentation.
                        let pore = |u, v| periodic_noise(u, v, 47, 53, seed.wrapping_add(919));
                        (
                            0.985 + 0.012 * (pore(u, v) - 0.5),
                            (pore(u + 1.0 / 256.0, v) - pore(u - 1.0 / 256.0, v)) * 0.14,
                            (pore(u, v + 1.0 / 256.0) - pore(u, v - 1.0 / 256.0)) * 0.14,
                            0.90 + pore(u, v) * 0.10,
                        )
                    }
                    _ => {
                        let fibre = |u, v| {
                            let warp = periodic_noise(u, v, 7, 5, seed.wrapping_add(729));
                            0.65 * periodic_noise(
                                u + (warp - 0.5) * 0.035,
                                v,
                                strand,
                                5,
                                seed.wrapping_add(813),
                            ) + 0.35
                                * periodic_noise(u, v, strand * 2 + 1, 11, seed.wrapping_add(1183))
                        };
                        let f = fibre(u, v);
                        let tone = periodic_noise(u, v, 5, 9, seed.wrapping_add(1927));
                        (
                            0.94 + 0.18 * (f - 0.5) + 0.12 * (tone - 0.5),
                            (fibre(u + 1.0 / 256.0, v) - fibre(u - 1.0 / 256.0, v)) * 0.11,
                            (fibre(u, v + 1.0 / 256.0) - fibre(u, v - 1.0 / 256.0)) * 0.025,
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

fn cloth_template(seed: u64) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.2, 0.4, 0.6),
        perceptual_roughness: 0.65 + (seed % 97) as f32 / 970.,
        metallic: 0.13,
        anisotropy_strength: 0.18,
        uv_transform: bevy::math::Affine2::from_scale_angle_translation(
            Vec2::new(1.6, 0.7),
            0.13,
            Vec2::new(0.17, 0.71),
        ),
        ..default()
    }
}

#[test]
fn cached_human_fields_replay_every_atlas_mip_and_pbr_template() {
    for seed in [0, 1, 7, 202, 1_013_005, 42_430_575, 43_084_584, u64::MAX] {
        let (mut before_images, mut before_materials) = (Assets::default(), Assets::default());
        let before = reference_maps(
            seed,
            cloth_template(seed),
            &mut before_images,
            &mut before_materials,
        );
        let (mut after_images, mut after_materials) = (Assets::default(), Assets::default());
        let after = maps(
            seed,
            cloth_template(seed),
            &mut after_images,
            &mut after_materials,
        );
        assert_eq!(before_images.len(), 6);
        assert_eq!(after_images.len(), 6);
        for (before, after) in before.iter().zip(&after) {
            let mut a = before_materials.get(before).unwrap().clone();
            let mut b = after_materials.get(after).unwrap().clone();
            for (ah, bh) in [
                (a.base_color_texture.take(), b.base_color_texture.take()),
                (a.normal_map_texture.take(), b.normal_map_texture.take()),
                (
                    a.metallic_roughness_texture.take(),
                    b.metallic_roughness_texture.take(),
                ),
            ] {
                assert_eq!(ah.is_some(), bh.is_some());
                if let (Some(ah), Some(bh)) = (ah, bh) {
                    let a = before_images.get(&ah).unwrap();
                    let b = after_images.get(&bh).unwrap();
                    assert_eq!(a.texture_descriptor, b.texture_descriptor);
                    assert_eq!(a.data, b.data, "seed {seed}");
                    assert_eq!(format!("{:?}", a.sampler), format!("{:?}", b.sampler));
                    assert_eq!(a.asset_usage, b.asset_usage);
                }
            }
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "seed {seed}");
        }
    }
}

#[test]
#[ignore = "counterbalanced CPU microbenchmark; run explicitly with --ignored --nocapture"]
fn cached_human_fields_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    // Warm transfer tables and the allocator before recurring-room timing.
    let (mut images, mut materials) = (Assets::default(), Assets::default());
    black_box(maps(0, cloth_template(0), &mut images, &mut materials));
    black_box(reference_maps(
        0,
        cloth_template(0),
        &mut images,
        &mut materials,
    ));
    drop((images, materials));
    let mut reference_seconds = Vec::new();
    let mut cached_seconds = Vec::new();
    for seed in 0..12 {
        let reference = || {
            let (mut images, mut materials) = (Assets::default(), Assets::default());
            let start = Instant::now();
            black_box(reference_maps(
                black_box(seed),
                cloth_template(seed),
                &mut images,
                &mut materials,
            ));
            start.elapsed().as_secs_f64()
        };
        let cached = || {
            let (mut images, mut materials) = (Assets::default(), Assets::default());
            let start = Instant::now();
            black_box(maps(
                black_box(seed),
                cloth_template(seed),
                &mut images,
                &mut materials,
            ));
            start.elapsed().as_secs_f64()
        };
        let (a, b) = if seed % 2 == 0 {
            (reference(), cached())
        } else {
            let b = cached();
            (reference(), b)
        };
        reference_seconds.push(a);
        cached_seconds.push(b);
    }
    reference_seconds.sort_by(f64::total_cmp);
    cached_seconds.sort_by(f64::total_cmp);
    let median = |x: &[f64]| (x[5] + x[6]) * 0.5;
    let reference = median(&reference_seconds);
    let cached = median(&cached_seconds);
    eprintln!(
        "{}",
        serde_json::json!({
            "atlas_size": 256,
            "halo_samples": 258 * 258,
            "peak_field_cache_bytes": 258 * 258 * 4,
            "rooms": 12,
            "reference_median_seconds": reference,
            "cached_median_seconds": cached,
            "kernel_speedup": reference / cached,
            "reference_seconds": reference_seconds,
            "cached_seconds": cached_seconds,
            "scope": "CPU skin/hair templates and all mips only; counterbalanced recurring-room work; no GPU or end-to-end capture claim"
        })
    );
}
