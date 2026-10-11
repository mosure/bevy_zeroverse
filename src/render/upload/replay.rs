//! Compare the staging uploader with WGPU's independent uploader and CPU source.
use super::*;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a native GPU"]
fn gpu_texture_uploads_match_source_and_wgpu_for_all_layers_and_mips() {
    let instance = wgpu::Instance::default();
    let adapter = futures_lite::future::block_on(instance.request_adapter(&default())).unwrap();
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&default())).unwrap();
    for format in [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgba32Float,
    ] {
        for order in [
            wgpu::util::TextureDataOrder::LayerMajor,
            wgpu::util::TextureDataOrder::MipMajor,
        ] {
            let mut image = Image::default();
            image.texture_descriptor.size = Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 6,
            };
            image.texture_descriptor.format = format;
            image.texture_descriptor.mip_level_count = 6;
            image.texture_descriptor.usage =
                TextureUsages::COPY_SRC | TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
            image.data_order = order;
            let bytes = format.block_copy_size(None).unwrap() as usize;
            let length = (0..6).map(|m| (32usize >> m).pow(2) * 6 * bytes).sum();
            // Raw nonuniform bits also catch swizzles, offset and half-float conversions.
            image.data = Some(
                (0..length)
                    .map(|i| ((i * 131 + i / 23) % 251) as u8)
                    .collect(),
            );
            let source = image.data.as_ref().unwrap();
            let (regions, size) = plan(&image).unwrap();
            let regular =
                device.create_texture_with_data(&queue, &image.texture_descriptor, order, source);
            let staged = device.create_texture(&image.texture_descriptor);
            let staging = device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            {
                let mut mapped = queue
                    .write_buffer_with(&staging, 0, std::num::NonZeroU64::new(size).unwrap())
                    .unwrap();
                pack(mapped.slice(..), source, &regions);
            }
            let mut encoder = device.create_command_encoder(&default());
            for r in &regions {
                encoder.copy_buffer_to_texture(
                    TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: layout(r),
                    },
                    TexelCopyTextureInfo {
                        texture: &staged,
                        mip_level: r.mip,
                        origin: Origin3d {
                            x: 0,
                            y: 0,
                            z: r.layer,
                        },
                        aspect: TextureAspect::All,
                    },
                    r.size,
                );
            }
            queue.submit([encoder.finish()]);
            for (name, texture) in [("wgpu", &regular), ("staged", &staged)] {
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: None,
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&default());
                for r in &regions {
                    encoder.copy_texture_to_buffer(
                        TexelCopyTextureInfo {
                            texture,
                            mip_level: r.mip,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: r.layer,
                            },
                            aspect: TextureAspect::All,
                        },
                        TexelCopyBufferInfo {
                            buffer: &buffer,
                            layout: layout(r),
                        },
                        r.size,
                    );
                }
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer.slice(..).map_async(MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let mapped = buffer.slice(..).get_mapped_range().unwrap();
                for r in &regions {
                    for row in 0..r.size.height as usize {
                        let src = r.source + row * r.row_bytes as usize;
                        let dst = r.destination as usize + row * r.row_pitch as usize;
                        assert_eq!(
                            &mapped[dst..dst + r.row_bytes as usize],
                            &source[src..src + r.row_bytes as usize],
                            "{name}: {format:?} {order:?}, layer {}, mip {}, row {row}",
                            r.layer,
                            r.mip
                        );
                    }
                }
                drop(mapped);
                buffer.unmap();
            }
        }
    }
}

fn layout(r: &CopyRegion) -> TexelCopyBufferLayout {
    TexelCopyBufferLayout {
        offset: r.destination,
        bytes_per_row: Some(r.row_pitch),
        rows_per_image: Some(r.size.height),
    }
}
