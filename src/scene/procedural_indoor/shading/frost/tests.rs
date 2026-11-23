use super::*;
use bevy::render::render_resource::Texture;
use half::f16;
use wgpu as gpu;

#[test]
fn reserve_angular_filtering_for_rough_transmissive_materials() {
    for (transmission, roughness, expected) in [
        (0., 0.6, false),
        (1., 0.045, false),
        (0.8, 0.18, true),
        (0.9, 1., true),
    ] {
        let material = StandardMaterial {
            specular_transmission: transmission,
            perceptual_roughness: roughness,
            ..default()
        };
        assert_eq!(needs_filter(&material), expected);
    }
}

// Independent integration of piecewise-constant source texels in destination
// coordinates. This deliberately doesn't use the integer shader construction.
fn cpu_mip(source: &[[f32; 4]], width: u32, height: u32) -> Vec<[f32; 4]> {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let mut out = Vec::new();
    for j in 0..h {
        for i in 0..w {
            let bounds = |p: u32, n: u32, d: u32| {
                (
                    p as f64 * n as f64 / d as f64,
                    (p + 1) as f64 * n as f64 / d as f64,
                )
            };
            let (left, right) = bounds(i, width, w);
            let (top, bottom) = bounds(j, height, h);
            let mut color = [0.; 4];
            for y in top.floor() as u32..bottom.ceil() as u32 {
                for x in left.floor() as u32..right.ceil() as u32 {
                    let area = ((x as f64 + 1.).min(right) - (x as f64).max(left))
                        * ((y as f64 + 1.).min(bottom) - (y as f64).max(top));
                    let p = source[(y * width + x) as usize];
                    for k in 0..3 {
                        color[k] += p[k] * (area / ((right - left) * (bottom - top))) as f32;
                    }
                    color[3] = if p[3].min(color[3]) < 0. {
                        -1.
                    } else {
                        p[3].max(color[3])
                    };
                }
            }
            out.push(color);
        }
    }
    out
}

#[test]
#[ignore = "requires a wgpu adapter; runs on software Vulkan"]
fn gpu_area_mips_preserve_hdr_edges_coverage_and_nearest_depth() {
    let instance = gpu::Instance::new(gpu::InstanceDescriptor {
        backends: gpu::Backends::from_env().unwrap_or(gpu::Backends::PRIMARY),
        ..gpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&default())).unwrap();
    let filters = Filters::new(&device);
    for (width, height) in [(9, 7), (1, 9), (16, 8)] {
        let size = gpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let input: Texture = device
            .create_texture(&gpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: gpu::TextureDimension::D2,
                format: gpu::TextureFormat::Rgba16Float,
                usage: gpu::TextureUsages::TEXTURE_BINDING | gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
            .into();
        let depth: Texture = device
            .create_texture(&gpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: gpu::TextureDimension::D2,
                format: gpu::TextureFormat::Depth16Unorm,
                usage: gpu::TextureUsages::TEXTURE_BINDING | gpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
            .into();
        let input_view = input.create_view(&default());
        let depth_view = depth.create_view(&default());
        let mut values: Vec<[f32; 4]> = (0..width * height)
            .map(|i| {
                [
                    0.375,
                    if i % width == width - 1 { 4. } else { 0. },
                    if i == width * height - 1 { 32. } else { 0. },
                    if i == 0 { 0.5 } else { 1. },
                ]
            })
            .collect();
        // A poisoned HDR texel must invalidate its ancestors for refinement
        // without leaking nonfinite radiance into any neighbor or mip level.
        values[2][0] = f32::NAN;
        values[4][1] = f32::INFINITY;
        values[6][2] = f32::NEG_INFINITY;
        let data: Vec<u8> = values
            .iter()
            .flatten()
            .flat_map(|&x| f16::from_f32(x).to_le_bytes())
            .collect();
        queue.write_texture(
            input.as_image_copy(),
            &data,
            gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: None,
            },
            size,
        );
        let pyramid = Pyramid::new(&device, &filters, width, height);
        let mut cache = None;
        let depths: Vec<u16> = (0..width * height)
            .map(|i| if i % 5 == 3 { 53_713 } else { 9_173 })
            .collect();
        queue.write_texture(
            gpu::TexelCopyTextureInfo {
                aspect: gpu::TextureAspect::DepthOnly,
                ..depth.as_image_copy()
            },
            bytemuck::cast_slice(&depths),
            gpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 2),
                rows_per_image: None,
            },
            size,
        );

        for with_depth in [true, false] {
            let mut encoder = device.create_command_encoder(&default());
            pyramid.encode(
                &device,
                &filters,
                &mut encoder,
                &input_view,
                with_depth.then_some(&depth_view),
                &mut cache,
            );
            let mut expected = values.clone();
            for (p, &z) in expected.iter_mut().zip(&depths) {
                if p.iter().any(|v| !v.is_finite()) {
                    *p = [0., 0., 0., -1.];
                    continue;
                }
                p[3] = if p[3] < 1. {
                    -1.
                } else if with_depth {
                    f32::from(z) / f32::from(u16::MAX)
                } else {
                    0.
                };
            }
            let mut checks = Vec::new();
            let (mut w, mut h) = (width, height);
            for mip in 1..pyramid.texture.mip_level_count() {
                expected = cpu_mip(&expected, w, h);
                w = (w / 2).max(1);
                h = (h / 2).max(1);
                let read = device.create_buffer(&gpu::BufferDescriptor {
                    label: None,
                    size: 256 * u64::from(h),
                    usage: gpu::BufferUsages::MAP_READ | gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    gpu::TexelCopyTextureInfo {
                        mip_level: mip,
                        ..pyramid.texture.as_image_copy()
                    },
                    gpu::TexelCopyBufferInfo {
                        buffer: &read,
                        layout: gpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: None,
                        },
                    },
                    gpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                );
                checks.push((read, w, expected.clone()));
            }
            queue.submit([encoder.finish()]);
            for (read, w, expected) in checks {
                let (tx, rx) = std::sync::mpsc::channel();
                read.slice(..)
                    .map_async(gpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(gpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = read.slice(..).get_mapped_range().unwrap();
                for (i, p) in expected.iter().enumerate() {
                    let start = i / w as usize * 256 + i % w as usize * 8;
                    for (k, &value) in p.iter().enumerate() {
                        let n = start + 2 * k;
                        let got = f16::from_le_bytes([bytes[n], bytes[n + 1]]).to_f32();
                        assert!(
                            (got - value).abs() <= 0.002 * value.abs().max(0.01),
                            "{width}x{height} mip width={w}, pixel {i}, channel {k}: {got} vs {value}"
                        );
                    }
                }
            }
        }
    }
}
