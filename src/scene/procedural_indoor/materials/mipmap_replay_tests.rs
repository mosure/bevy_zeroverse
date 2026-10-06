//! Allocation-only changes replay the original mip reducer and image metadata.
use super::*;

fn original_image(base: Vec<u8>, size: u32, kind: MapType) -> Image {
    // Input texels are bytes: evaluate exactly the same transfer function once
    // per possible input instead of millions of powf calls per room. Filtering
    // and the floating-point accumulation order remain unchanged.
    let linear = srgb8_table();
    let mut bytes = base.clone();
    let mut prev = base;
    let mut n = size as usize;
    while n > 1 {
        let next_n = n / 2;
        let mut next = Vec::with_capacity(next_n * next_n * 4);
        for y in 0..next_n {
            for x in 0..next_n {
                let mut sum = Vec3::ZERO;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = ((y * 2 + dy) * n + x * 2 + dx) * 4;
                    sum += match kind {
                        MapType::Color => Vec3::new(
                            linear[prev[i] as usize],
                            linear[prev[i + 1] as usize],
                            linear[prev[i + 2] as usize],
                        ),
                        MapType::Normal => {
                            Vec3::new(prev[i] as f32, prev[i + 1] as f32, prev[i + 2] as f32)
                                / 255.0
                                * 2.0
                                - Vec3::ONE
                        }
                        MapType::Data => {
                            Vec3::new(prev[i] as f32, prev[i + 1] as f32, prev[i + 2] as f32)
                                / 255.0
                        }
                    };
                }
                let value = match kind {
                    MapType::Color => (sum * 0.25).map(linear_to_srgb),
                    MapType::Normal => sum.normalize_or_zero() * 0.5 + Vec3::splat(0.5),
                    MapType::Data => sum * 0.25,
                };
                next.extend([
                    (value.x * 255.0) as u8,
                    (value.y * 255.0) as u8,
                    (value.z * 255.0) as u8,
                    255,
                ]);
            }
        }
        bytes.extend_from_slice(&next);
        prev = next;
        n = next_n;
    }
    original_chain_image(bytes, size, kind)
}

fn original_chain_image(bytes: Vec<u8>, size: u32, kind: MapType) -> Image {
    // Image::new validates base-level size, so append mip data afterwards.
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0; (size * size * 4) as usize],
        if matches!(kind, MapType::Color) {
            TextureFormat::Rgba8UnormSrgb
        } else {
            TextureFormat::Rgba8Unorm
        },
        RenderAssetUsages::default(),
    );
    image.data = Some(bytes);
    image.texture_descriptor.mip_level_count = size.ilog2() + 1;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        anisotropy_clamp: 4,
        ..default()
    });
    image
}

#[test]
fn contiguous_mips_preserve_all_bytes_and_image_metadata() {
    for size in [1_u32, 2, 8, 64, 256, 512] {
        for seed in [0_u64, 202, 1_013_005] {
            let mut base = Vec::with_capacity((size * size * 4) as usize);
            for y in 0..size {
                for x in 0..size {
                    let value = hash(x, y, seed).to_bits();
                    base.extend(value.to_le_bytes());
                }
            }
            for kind in [MapType::Color, MapType::Normal, MapType::Data] {
                let a = original_image(base.clone(), size, kind);
                let b = mip_image(base.clone(), size, kind);
                assert_eq!(a.data, b.data, "size {size} seed {seed}");
                assert_eq!(a.texture_descriptor, b.texture_descriptor);
                assert_eq!(a.texture_view_descriptor, b.texture_view_descriptor);
                assert_eq!(a.asset_usage, b.asset_usage);
                assert_eq!(a.data_order, b.data_order);
                assert_eq!(a.sampler, b.sampler);
                assert_eq!(a.copy_on_resize, b.copy_on_resize);
            }
        }
    }
}

#[test]
#[ignore = "counterbalanced color mip kernel timing; run explicitly"]
fn cached_color_mip_cpu_benchmark() {
    use std::{hint::black_box, time::Instant};
    for size in [256, 512] {
        let mut sources = Vec::new();
        for seed in [200, 202, 207] {
            let recipes = program::sample(seed);
            for surface in [
                Surface::Paint,
                Surface::Wood,
                Surface::Concrete,
                Surface::Fabric,
                Surface::Floor,
                Surface::Ceramic,
                Surface::Ceiling,
            ] {
                let recipe = &recipes[surface as usize];
                if recipe.map_size(0) == size {
                    let (color, _, _) =
                        texture_maps(surface, 0, seed, recipe.roughness, Some(recipe));
                    assert_eq!(color.len(), (size * size * 4) as usize);
                    sources.push(color);
                }
            }
        }
        for source in &sources {
            let a = mip_image_with_transfer(source.clone(), size, MapType::Color, |v| {
                (linear_to_srgb(v) * 255.) as u8
            });
            let b = mip_image(source.clone(), size, MapType::Color);
            assert_eq!(a.data, b.data);
        }
        let (mut reference, mut current) = (Vec::new(), Vec::new());
        for round in 0..8 {
            for cached in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                for source in &sources {
                    let image = if cached {
                        mip_image(source.clone(), size, MapType::Color)
                    } else {
                        mip_image_with_transfer(source.clone(), size, MapType::Color, |v| {
                            (linear_to_srgb(v) * 255.) as u8
                        })
                    };
                    black_box(image);
                }
                if cached { &mut current } else { &mut reference }
                    .push(start.elapsed().as_secs_f64());
            }
        }
        program::replay_helpers::report_benchmark(
            "exact mip transfer cache, including allocation",
            size,
            reference,
            current,
        );
    }
}
