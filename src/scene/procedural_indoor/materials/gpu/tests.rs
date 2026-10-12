use super::super::{program, Surface};
use super::*;
use wgpu::util::DeviceExt;

#[test]
fn recipe_abi_and_bounded_mip_layout() {
    assert_eq!(std::mem::size_of::<recipe::Recipe>(), 15 * 16);
    assert_eq!(kernel::SHADER, include_str!("synthesis.wgsl"));
    for size in [256, 512] {
        let levels = kernel::levels(size);
        assert_eq!(levels.len(), size.ilog2() as usize + 1);
        for l in levels {
            assert_eq!(l.offset % 256, 0);
            assert_eq!(l.stride % 256, 0);
            assert!(l.stride >= l.size * 4);
        }
    }
    for floor in 0..3 {
        for r in program::sample_with_floor(202, floor).unwrap() {
            if recipe::Recipe::new(&r, floor).is_some() {
                assert!(matches!(
                    r.surface,
                    Surface::Fabric
                        | Surface::FabricAlt
                        | Surface::Leather
                        | Surface::Metal
                        | Surface::Chrome
                        | Surface::Plastic
                        | Surface::Rubber
                        | Surface::Paper
                        | Surface::Floor
                ));
                assert!(r.surface != Surface::Floor || floor == 1);
            }
        }
    }
}

#[test]
fn lighting_samples_preserve_full_resolution_cpu_point_sampling() {
    for floor in 0..3 {
        for mut r in program::sample_with_floor(u64::MAX - 7, floor).unwrap() {
            if r.surface == Surface::FabricAlt {
                r.textile.as_mut().unwrap().knit = true;
            }
            let Maps::Gpu(plan) = Maps::prepare(&r, floor, true) else {
                continue;
            };
            let cpu = r.maps(floor);
            let data = cpu[0].data.as_ref().unwrap();
            let n = cpu[0].width() as usize;
            for y in 0..32 {
                for x in 0..32 {
                    let i = ((y * n / 32) * n + x * n / 32) * 4;
                    assert_eq!(
                        plan.albedo[y * 32 + x],
                        data[i..i + 3],
                        "{:?} {floor} {x} {y}",
                        r.surface
                    );
                }
            }
        }
    }
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("hardware GPU required");
    eprintln!(
        "material GPU qualification adapter: {:?}",
        adapter.get_info()
    );
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits::default(),
        required_features: wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ..Default::default()
    }))
    .unwrap()
}

#[test]
#[ignore = "requires a native GPU; compares all base maps and every normal/GGX mip against the CPU oracle"]
fn gpu_maps_match_cpu_material_programs_and_mips() {
    let (device, queue) = device();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &kernel::layout_entries(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("material qualification"),
        source: wgpu::ShaderSource::Wgsl(
            std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/scene/procedural_indoor/materials/gpu/synthesis.wgsl"
            ))
            .unwrap()
            .into(),
        ),
    });
    let pipelines = kernel::ENTRIES.map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let scratch = kernel::Workspace::new(&device);
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("material bake timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut report = Vec::new();
    for seed in [0, 1, 202, 400, u64::MAX - 7] {
        for floor in [0, 1, 2] {
            for mut r in program::sample_with_floor(seed, floor).unwrap() {
                if floor != 1 && r.surface == Surface::Floor {
                    continue;
                }
                // A single floor-independent pass plus explicit knitted fabric and floor carpet.
                if floor != 0 && !matches!(r.surface, Surface::Floor | Surface::FabricAlt) {
                    continue;
                }
                if floor == 2 {
                    r.textile.as_mut().unwrap().knit = true;
                }
                if seed == u64::MAX - 7 {
                    r.seed = seed;
                } // exercise u64 salt carry in WGSL
                if seed == 1 && floor == 0 {
                    if r.surface == Surface::FabricAlt {
                        r.textile = None;
                    }
                    if r.surface == Surface::Leather {
                        r.leather = None;
                    }
                }
                let Some(params) = recipe::Recipe::new(&r, floor) else {
                    continue;
                };
                let cpu_started = std::time::Instant::now();
                let expected = r.maps(floor);
                let cpu_synthesis_seconds = cpu_started.elapsed().as_secs_f64();
                let n = r.map_size(floor);
                let textures = expected.each_ref().map(|image| {
                    let mut d = image.texture_descriptor.clone();
                    d.usage |= wgpu::TextureUsages::COPY_SRC;
                    device.create_texture(&d)
                });
                let bind = scratch.bindings(&device, &layout, &params);
                let bytes = kernel::plane_bytes(n) * 3;
                let read = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &vec![0; bytes as usize + 16],
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.write_timestamp(&queries, 0);
                scratch.encode(
                    &mut encoder,
                    [
                        &pipelines[params.meta[0] as usize],
                        &pipelines[4],
                        &pipelines[5],
                    ],
                    &bind,
                    n,
                    textures.each_ref(),
                );
                encoder.write_timestamp(&queries, 1);
                encoder.resolve_query_set(&queries, 0..2, &query_buffer, 0);
                encoder.copy_buffer_to_buffer(&query_buffer, 0, &read, bytes, 16);
                for (map, texture) in textures.iter().enumerate() {
                    for (mip, l) in kernel::levels(n).into_iter().enumerate() {
                        encoder.copy_texture_to_buffer(
                            wgpu::TexelCopyTextureInfo {
                                texture,
                                mip_level: mip as u32,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyBufferInfo {
                                buffer: &read,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: map as u64 * kernel::plane_bytes(n) + l.offset,
                                    bytes_per_row: Some(l.stride),
                                    rows_per_image: None,
                                },
                            },
                            wgpu::Extent3d {
                                width: l.size,
                                height: l.size,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                read.slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let data = read.slice(..).get_mapped_range().unwrap();
                let begin = u64::from_le_bytes(
                    data[bytes as usize..bytes as usize + 8].try_into().unwrap(),
                );
                let end = u64::from_le_bytes(
                    data[bytes as usize + 8..bytes as usize + 16]
                        .try_into()
                        .unwrap(),
                );
                let gpu_bake_seconds =
                    (end - begin) as f64 * f64::from(queue.get_timestamp_period()) * 1e-9;
                let mut summaries = Vec::new();
                for (map, image) in expected.iter().enumerate() {
                    let oracle = image.data.as_ref().unwrap();
                    let mut cursor = 0;
                    for (mip, l) in kernel::levels(n).into_iter().enumerate() {
                        let mut squared = 0u64;
                        let mut max = 0u8;
                        let mut changed = 0u64;
                        let mut decoded = Vec::new();
                        for y in 0..l.size {
                            let start = (map as u64 * kernel::plane_bytes(n)
                                + l.offset
                                + u64::from(y * l.stride))
                                as usize;
                            for x in 0..l.size as usize * 4 {
                                let a = data[start + x];
                                let b = oracle[cursor];
                                cursor += 1;
                                let diff = a.abs_diff(b);
                                max = max.max(diff);
                                squared += u64::from(diff).pow(2);
                                changed += u64::from(diff != 0);
                                decoded.push(a);
                            }
                        }
                        let rmse = (squared as f64 / (l.size * l.size * 4) as f64).sqrt();
                        summaries.push(serde_json::json!({"map":map,"mip":mip,"rmse_bytes":rmse,"max_bytes":max,"changed_bytes":changed}));
                        // Bounds in encoded 8-bit units. The CPU/GPU floating-point execution need not be bit-identical.
                        assert!(max<=3 && rmse<=0.65,"seed {seed} {:?} floor {floor} map {map} mip {mip}: max {max}, rmse {rmse}",r.surface);
                        if seed == 202 && floor == 0 && mip == 0 {
                            let path = format!(
                                "out/gpu_materials/seed202_{:?}_map{map}_gpu.png",
                                r.surface
                            );
                            std::fs::create_dir_all("out/gpu_materials").unwrap();
                            image::save_buffer(
                                path,
                                &decoded,
                                l.size,
                                l.size,
                                image::ColorType::Rgba8,
                            )
                            .unwrap();
                            image::save_buffer(
                                format!(
                                    "out/gpu_materials/seed202_{:?}_map{map}_cpu.png",
                                    r.surface
                                ),
                                &oracle[..(n * n * 4) as usize],
                                n,
                                n,
                                image::ColorType::Rgba8,
                            )
                            .unwrap();
                        }
                    }
                }
                report.push(serde_json::json!({"seed":seed,"floor":floor,"surface":r.surface,"knit":r.textile.as_ref().is_some_and(|t|t.knit),"cpu_synthesis_seconds":cpu_synthesis_seconds,"gpu_bake_seconds":gpu_bake_seconds,"maps":summaries}));
            }
        }
    }
    std::fs::create_dir_all("out/gpu_materials").unwrap();
    std::fs::write(
        "out/gpu_materials/map_parity.json",
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    eprintln!(
        "qualified {} recipes, all three maps and complete mip chains",
        report.len()
    );
}
