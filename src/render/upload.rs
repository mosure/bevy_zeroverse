//! Native uploads without libc's large REP MOVSB copy into uncached Vulkan BAR
//! mappings. Uses published Bevy/WGPU APIs; shaders and texture bytes are unchanged.
use bevy::{
    image::ImageSampler,
    prelude::*,
    render::{
        render_asset::{
            prepare_assets, ExtractedAssets, RenderAssetBytesPerFrameLimiter, RenderAssets,
        },
        render_resource::*,
        renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
};

/// Fixed-size copies stay below libc's ERMS threshold. The opaque destination
/// prevents LLVM coalescing the loop back into one large memcpy. Safe slices
/// preserve all alignment/tail cases; no CPU features, unsafe stores or env vars.
pub(super) fn copy_upload(destination: wgpu::WriteOnly<'_, [u8]>, source: &[u8]) {
    assert_eq!(destination.len(), source.len());
    let (dst, mut dst_tail) = destination.into_chunks::<256>();
    let (src, src_tail) = source.as_chunks::<256>();
    for (dst, src) in dst.into_iter().zip(src) {
        std::hint::black_box(dst).write(*src);
    }
    dst_tail.copy_from_slice(src_tail);
}

pub(super) fn install(app: &mut App) {
    let Some(adapter) = app.world().get_resource::<RenderAdapterInfo>() else {
        return;
    };
    if adapter.vendor != 0x10de || adapter.backend != wgpu::Backend::Vulkan {
        return;
    }
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(
            Render,
            upload_images
                .in_set(RenderSystems::PrepareAssets)
                .before(prepare_assets::<GpuImage>),
        );
    }
}

#[derive(Debug)]
struct CopyRegion {
    source: usize,
    destination: u64,
    row_bytes: u32,
    row_pitch: u32,
    size: Extent3d,
    layer: u32,
    mip: u32,
}

/// Only complete initial, uncompressed color images take this path. Bevy keeps
/// ownership of removals, updates, resizing, unsupported formats and byte budgets.
fn plan(image: &Image) -> Option<(Vec<CopyRegion>, u64)> {
    let desc = &image.texture_descriptor;
    let data = image.data.as_ref()?;
    if desc.dimension != TextureDimension::D2
        || desc.sample_count != 1
        || image.copy_on_resize
        || !matches!(
            desc.format,
            TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba8UnormSrgb
                | TextureFormat::Rgba16Float
                | TextureFormat::Rgba32Float
        )
    {
        return None;
    }
    let block = desc.format.block_copy_size(None)?;
    let (outer, inner) = match image.data_order {
        wgpu::util::TextureDataOrder::LayerMajor => {
            (desc.size.depth_or_array_layers, desc.mip_level_count)
        }
        wgpu::util::TextureDataOrder::MipMajor => {
            (desc.mip_level_count, desc.size.depth_or_array_layers)
        }
    };
    let mut regions = Vec::new();
    let (mut source, mut destination) = (0usize, 0u64);
    for a in 0..outer {
        for b in 0..inner {
            let (layer, mip) = match image.data_order {
                wgpu::util::TextureDataOrder::LayerMajor => (a, b),
                wgpu::util::TextureDataOrder::MipMajor => (b, a),
            };
            let mut size = desc.mip_level_size(mip)?;
            size.depth_or_array_layers = 1;
            let row_bytes = size.width.checked_mul(block)?;
            let row_pitch = row_bytes.checked_add(255)? / 256 * 256;
            regions.push(CopyRegion {
                source,
                destination,
                row_bytes,
                row_pitch,
                size,
                layer,
                mip,
            });
            source = source.checked_add((row_bytes as usize).checked_mul(size.height as usize)?)?;
            destination = destination.checked_add(u64::from(row_pitch) * u64::from(size.height))?;
        }
    }
    (source == data.len() && destination > 0).then_some((regions, destination))
}

fn pack(mut destination: wgpu::WriteOnly<'_, [u8]>, source: &[u8], regions: &[CopyRegion]) {
    for region in regions {
        for row in 0..region.size.height as usize {
            let src = region.source + row * region.row_bytes as usize;
            let dst = region.destination as usize + row * region.row_pitch as usize;
            copy_upload(
                destination.slice(dst..dst + region.row_bytes as usize),
                &source[src..src + region.row_bytes as usize],
            );
            destination
                .slice(dst + region.row_bytes as usize..dst + region.row_pitch as usize)
                .fill(0);
        }
    }
}

fn upload_images(
    mut extracted: ResMut<ExtractedAssets<GpuImage>>,
    mut assets: ResMut<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    sampler: Res<DefaultImageSampler>,
    budget: Res<RenderAssetBytesPerFrameLimiter>,
) {
    // Explicit caller budgets remain Bevy's responsibility, including deferred
    // uploads. The default capture pipeline has no per-frame byte limit.
    if budget.max_bytes.is_some() {
        return;
    }
    let mut encoder = None;
    let extracted = &mut *extracted;
    let removed = &extracted.removed;
    extracted.extracted.retain(|(id, image)| {
        if assets.get(*id).is_some() || removed.contains(id) {
            return true;
        }
        let Some((regions, size)) = plan(image) else {
            return true;
        };
        if size > device.limits().max_buffer_size {
            return true;
        }
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("zeroverse_image_upload"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        {
            // mapped_at_creation would zero the entire BAR mapping with libc
            // memset before our copy, recreating the same stall. Queue staging
            // needs no implicit zero-fill; pack writes every row and padding.
            let mut mapped = queue
                .write_buffer_with(&staging, 0, std::num::NonZeroU64::new(size).unwrap())
                .expect("nonempty aligned image upload");
            pack(mapped.slice(..), image.data.as_ref().unwrap(), &regions);
        }
        let mut desc = image.texture_descriptor.clone();
        desc.usage |= TextureUsages::COPY_DST;
        let texture = device.create_texture(&desc);
        let encoder = encoder.get_or_insert_with(|| {
            device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("zeroverse_image_uploads"),
            })
        });
        for region in regions {
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: &staging,
                    layout: TexelCopyBufferLayout {
                        offset: region.destination,
                        bytes_per_row: Some(region.row_pitch),
                        rows_per_image: Some(region.size.height),
                    },
                },
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: region.mip,
                    origin: Origin3d {
                        x: 0,
                        y: 0,
                        z: region.layer,
                    },
                    aspect: TextureAspect::All,
                },
                region.size,
            );
        }
        let texture_view =
            texture.create_view(&image.texture_view_descriptor.clone().unwrap_or_default());
        let sampler = match &image.sampler {
            ImageSampler::Default => (**sampler).clone(),
            ImageSampler::Descriptor(desc) => device.create_sampler(&desc.as_wgpu()),
        };
        assets.insert(
            *id,
            GpuImage {
                texture,
                texture_view,
                sampler,
                texture_descriptor: image.texture_descriptor.clone(),
                texture_view_descriptor: image.texture_view_descriptor.clone(),
                had_data: true,
            },
        );
        false
    });
    if let Some(encoder) = encoder {
        // Submit before Bevy encodes any material draws. WGPU retains staging
        // allocations through completion; no map wait, device poll or unsafe HAL.
        queue.submit([encoder.finish()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upload_copy_preserves_unaligned_ranges_and_tails() {
        for size in [0, 1, 255, 256, 257, 4095, 4096, 65537] {
            let source: Vec<_> = (0..size).map(|i| (i * 17) as u8).collect();
            let mut target = vec![91; size + 6];
            copy_upload((&mut target[3..3 + size]).into(), &source);
            assert_eq!(&target[3..3 + size], &source);
            assert_eq!(&target[..3], &[91; 3]);
            assert_eq!(&target[3 + size..], &[91; 3]);
        }
    }
    #[test]
    fn layered_mips_keep_exact_source_order_and_padded_rows() {
        for order in [
            wgpu::util::TextureDataOrder::LayerMajor,
            wgpu::util::TextureDataOrder::MipMajor,
        ] {
            let mut image = Image::default();
            image.texture_descriptor.format = TextureFormat::Rgba8Unorm;
            image.texture_descriptor.size = Extent3d {
                width: 5,
                height: 3,
                depth_or_array_layers: 2,
            };
            image.texture_descriptor.mip_level_count = 3;
            image.data_order = order;
            let source: Vec<_> = (0..(5 * 3 + 2 + 1) * 4 * 2).map(|i| i as u8).collect();
            image.data = Some(source.clone());
            let (regions, size) = plan(&image).unwrap();
            assert_eq!(regions.len(), 6);
            assert_eq!(
                regions.iter().map(|r| (r.layer, r.mip)).collect::<Vec<_>>(),
                match order {
                    wgpu::util::TextureDataOrder::LayerMajor =>
                        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                    wgpu::util::TextureDataOrder::MipMajor =>
                        vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
                }
            );
            let mut packed = vec![191; size as usize];
            pack(packed.as_mut_slice().into(), &source, &regions);
            let mut recovered = Vec::new();
            for r in regions {
                assert_eq!(r.destination % 256, 0);
                for row in 0..r.size.height as usize {
                    let start = r.destination as usize + row * r.row_pitch as usize;
                    recovered.extend_from_slice(&packed[start..start + r.row_bytes as usize]);
                    assert!(
                        packed[start + r.row_bytes as usize..start + r.row_pitch as usize]
                            .iter()
                            .all(|byte| *byte == 0)
                    );
                }
            }
            assert_eq!(recovered, source);
            image.data.as_mut().unwrap().pop();
            assert!(plan(&image).is_none());
        }
    }
}
