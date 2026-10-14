use super::*;

#[test]
fn reflection_program_is_bounded_and_covers_every_cube_face_and_mip() {
    let p = program();
    assert_eq!(std::mem::size_of::<Tap>(), 48);
    assert_eq!(std::mem::size_of::<Pixel>(), 16);
    assert!(p.taps.len() * 48 < 32 * 1024 * 1024);
    assert_eq!(p.copies.len(), 6 * (1 + 7));
    let mut covered = std::collections::BTreeSet::new();
    for pixel in &p.pixels {
        assert!((u64::from(pixel.offset) + 1) * 8 <= p.bytes);
        assert!(covered.insert(pixel.offset));
        if pixel.kind != 0 {
            assert!(pixel.taps as usize + 64 <= p.taps.len());
        }
    }
    for copy in &p.copies {
        assert_eq!(copy.offset % 256, 0);
        assert_eq!(copy.stride % 256, 0);
        for y in 0..copy.size {
            for x in 0..copy.size {
                assert!(covered.contains(&((copy.offset / 8) as u32 + y * copy.stride / 8 + x)));
            }
        }
    }
}

#[test]
#[ignore = "requires a native GPU; reflection-map and mip parity"]
fn gpu_reflection_maps_match_cpu_quadrature() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &layout_entries(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("IBL oracle"),
        source: wgpu::ShaderSource::Wgsl(include_str!("convolve.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("convolve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let workspace = Workspace::new(&device);
    for case in 0..3 {
        let pixels: Vec<_> = (0..6 * SIZE * SIZE)
            .map(|i| match case {
                0 => Vec3::new(0.3, 15., 2300.),
                1 => Vec3::new(
                    (i % 73) as f32 * 0.03,
                    (i % 113) as f32 * 5.,
                    (i % 19) as f32,
                ),
                _ => {
                    if i % 1307 < 3 {
                        Vec3::splat(64000.)
                    } else {
                        Vec3::splat(0.01)
                    }
                }
            })
            .collect();
        let expected = super::super::stencil::convolve(&pixels);
        let source: Vec<_> = RadianceMips::new(&pixels, SIZE)
            .0
            .into_iter()
            .flat_map(|(_, p)| p.into_iter().map(|p| [p.x, p.y, p.z, 0.]))
            .collect();
        let textures = expected.each_ref().map(|image| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: image.texture_descriptor.size,
                mip_level_count: image.texture_descriptor.mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: program().bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        workspace.encode(
            &device,
            &mut encoder,
            &pipeline,
            &layout,
            &source,
            [&textures[0], &textures[1]],
        );
        // Copy the textures back, rather than merely checking intermediate scratch.
        for c in &program().copies {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &textures[c.target],
                    mip_level: c.mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: c.face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &read,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: c.offset,
                        bytes_per_row: Some(c.stride),
                        rows_per_image: Some(c.size),
                    },
                },
                wgpu::Extent3d {
                    width: c.size,
                    height: c.size,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = read.slice(..).get_mapped_range().unwrap();
        let mut offsets = [0usize; 2];
        let mut error_sum = 0.;
        let mut count = 0;
        for c in &program().copies {
            let cpu = expected[c.target].data.as_ref().unwrap();
            for y in 0..c.size {
                for x in 0..c.size * 4 {
                    let a = c.offset as usize + y as usize * c.stride as usize + x as usize * 2;
                    let b = offsets[c.target] + (y * c.size * 4 + x) as usize * 2;
                    let a = half::f16::from_le_bytes(mapped[a..a + 2].try_into().unwrap()).to_f32();
                    let b = half::f16::from_le_bytes(cpu[b..b + 2].try_into().unwrap()).to_f32();
                    let error = (a - b).abs() / b.abs().max(1.);
                    assert!(
                        a.is_finite() && error <= 0.002,
                        "case {case} face {} mip {}: {a} vs {b}",
                        c.face,
                        c.mip
                    );
                    error_sum += f64::from(error);
                    count += 1;
                }
            }
            offsets[c.target] += (c.size * c.size * 8) as usize;
        }
        assert!(
            error_sum / (count as f64) < 0.00003,
            "mean relative error {}",
            error_sum / (count as f64)
        );
    }
}
