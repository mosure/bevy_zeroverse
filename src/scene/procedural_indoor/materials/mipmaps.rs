//! Color-space-aware image construction and mip transfer primitives.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum MapType {
    Color,
    Normal,
    Data,
}

pub(super) fn mip_image(bytes: Vec<u8>, size: u32, kind: MapType) -> Image {
    let mut transfer = matches!(kind, MapType::Color).then(mip_transfer::ColorTransfer::new);
    mip_image_with_transfer(bytes, size, kind, |value| {
        transfer.as_mut().unwrap().encode(value)
    })
}

pub(super) fn mip_image_with_transfer(
    mut bytes: Vec<u8>,
    size: u32,
    kind: MapType,
    mut color_transfer: impl FnMut(f32) -> u8,
) -> Image {
    // Input texels are bytes: evaluate exactly the same transfer function once
    // per possible input instead of millions of powf calls per room. Filtering
    // and the floating-point accumulation order remain unchanged.
    let linear = srgb8_table();
    // Keep all levels in their final allocation. Reading a completed level and
    // appending its successor avoids cloning the base and allocating each mip.
    let mut n = size as usize;
    let mut mip_bytes = 0;
    let mut level = n;
    while level > 1 {
        level /= 2;
        mip_bytes += level * level * 4;
    }
    bytes.reserve(mip_bytes);
    let mut previous = 0;
    while n > 1 {
        let next_n = n / 2;
        let next_offset = bytes.len();
        for y in 0..next_n {
            for x in 0..next_n {
                let mut sum = Vec3::ZERO;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = previous + ((y * 2 + dy) * n + x * 2 + dx) * 4;
                    sum += match kind {
                        MapType::Color => Vec3::new(
                            linear[bytes[i] as usize],
                            linear[bytes[i + 1] as usize],
                            linear[bytes[i + 2] as usize],
                        ),
                        MapType::Normal => {
                            Vec3::new(bytes[i] as f32, bytes[i + 1] as f32, bytes[i + 2] as f32)
                                / 255.0
                                * 2.0
                                - Vec3::ONE
                        }
                        MapType::Data => {
                            Vec3::new(bytes[i] as f32, bytes[i + 1] as f32, bytes[i + 2] as f32)
                                / 255.0
                        }
                    };
                }
                if matches!(kind, MapType::Color) {
                    let value = sum * 0.25;
                    bytes.extend([
                        color_transfer(value.x),
                        color_transfer(value.y),
                        color_transfer(value.z),
                        255,
                    ]);
                    continue;
                }
                let value = match kind {
                    MapType::Color => unreachable!("color transfer is present"),
                    MapType::Normal => sum.normalize_or_zero() * 0.5 + Vec3::splat(0.5),
                    MapType::Data => sum * 0.25,
                };
                bytes.extend([
                    (value.x * 255.0) as u8,
                    (value.y * 255.0) as u8,
                    (value.z * 255.0) as u8,
                    255,
                ]);
            }
        }
        previous = next_offset;
        n = next_n;
    }
    mip_chain_image(bytes, size, kind)
}

pub(super) fn mip_chain_image(bytes: Vec<u8>, size: u32, kind: MapType) -> Image {
    // The complete chain is already owned; avoid allocating and zeroing a base
    // image only to replace its data immediately.
    let mut image = Image::new_uninit(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
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

pub(super) fn srgb8_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear((Vec3::splat(i as f32) / 255.0).x)))
}

pub(super) fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
pub(super) fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
