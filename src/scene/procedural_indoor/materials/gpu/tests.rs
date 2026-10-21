use super::super::{program, Surface};
use super::*;
use wgpu::util::DeviceExt;

#[path = "minification_tests.rs"]
mod minification_tests;
#[path = "web_fixtures.rs"]
mod web_fixtures;

#[test]
fn chrome_bakes_retain_only_the_maps_its_material_samples() {
    let recipes = program::sample_with_floor(3, 0).unwrap();
    for surface in [Surface::Chrome, Surface::Fabric] {
        let mut images = Assets::<Image>::default();
        let mut requests = Requests::default();
        let handles =
            Maps::prepare(&recipes[surface as usize], 0, true).insert(&mut images, &mut requests);
        assert_eq!(requests.0.len(), 1);
        let normal_output = surface != Surface::Chrome;
        assert_eq!(requests.0[0].normal_output, normal_output);
        assert_eq!(images.len(), if normal_output { 3 } else { 2 });
        assert!(images.get(&handles[0]).is_some());
        assert!(images.get(&handles[2]).is_some());
        assert_eq!(images.get(&handles[1]).is_some(), normal_output);
        assert_eq!(requests.albedo(handles[0].id()).unwrap().len(), 1024);
    }
}
#[test]
fn material_shader_parses_and_validates_without_a_gpu() {
    let module = wgpu::naga::front::wgsl::parse_str(kernel::SHADER)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(kernel::SHADER)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
fn recipe_abi_and_bounded_mip_layout() {
    assert_eq!(std::mem::size_of::<recipe::Recipe>(), 42 * 16);
    assert_eq!(std::mem::size_of::<kernel::Dispatch>(), 3 * 16);
    assert!(kernel::SHADER.contains("fn paint_map"));
    for size in [256, 512] {
        let levels = kernel::levels(size);
        assert_eq!(levels.len(), size.ilog2() as usize + 1);
        let dispatches = kernel::dispatches(size);
        let mut occupied = vec![false; kernel::plane_bytes(size) as usize / 4];
        let mut moments = 0;
        for (i, l) in levels.iter().enumerate() {
            assert_eq!(l.offset % 256, 0);
            assert_eq!(l.stride % 256, 0);
            assert!(l.stride >= l.size * 4);
            let d = dispatches[i];
            assert_eq!(
                d.destination,
                [l.size, l.offset as u32 / 4, l.stride / 4, moments]
            );
            assert_eq!(d.plane[0] as u64 * 4, kernel::plane_bytes(size));
            assert_eq!(d.source, dispatches[i.saturating_sub(1)].destination);
            for y in 0..l.size {
                for x in 0..l.size {
                    let at = (l.offset / 4) as usize + (y * l.stride / 4 + x) as usize;
                    assert!(!std::mem::replace(&mut occupied[at], true), "mip alias");
                }
            }
            moments += l.size * l.size;
        }
        assert_eq!(moments, (size * size * 4 - 1) / 3);
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
                        | Surface::Paint
                        | Surface::Accent
                        | Surface::Ceiling
                ));
                assert!(r.surface != Surface::Floor || floor == 1);
            }
        }
    }
}

#[test]
fn expanded_material_programs_have_explicit_gpu_parameters_and_safe_fallback() {
    let mut coverings = 0;
    for seed in 0..64 {
        let recipes = program::sample_with_floor(seed, 1).unwrap();
        for surface in [
            Surface::Paint,
            Surface::Accent,
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Rubber,
        ] {
            let r = &recipes[surface as usize];
            let params = recipe::Recipe::new(r, 1).unwrap();
            if let Some(w) = r.coating.as_ref().and_then(|c| c.wall.as_ref()) {
                coverings += 1;
                assert_eq!(params.wall_weights, w.weights);
                assert_eq!(
                    params.wall_counts,
                    [
                        w.fiber_cells[0],
                        w.fiber_cells[1],
                        w.bands,
                        w.print_style + 1
                    ]
                );
                assert_eq!(&params.wall_tint[..3], &w.secondary_tint);
                assert_eq!(
                    params.wall0,
                    [w.warp, w.pigment, w.relief_m, w.roughness_variation]
                );
            }
            if let Some(m) = &r.manufactured {
                assert_eq!(params.manufactured_counts, [m.cells[0], m.cells[1], 1, 0]);
                assert_eq!(&params.manufactured[..2], &[m.structure, m.peening]);
            }
        }
        // A caller may author a hard-floor installation on carpet. GPU support
        // must be declined rather than silently baking it as a rectangular grid.
        let mut r = recipes[Surface::Floor as usize].clone();
        let hard = program::sample_with_floor(seed, 2).unwrap();
        r.flooring = hard[Surface::Floor as usize].flooring.clone();
        use super::super::flooring::FloorLayout;
        if !matches!(
            r.flooring.as_ref().unwrap().layout,
            FloorLayout::Straight | FloorLayout::BrokenBond
        ) {
            assert!(recipe::Recipe::new(&r, 1).is_none());
        }
    }
    assert!(coverings > 50);
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

#[test]
#[ignore = "CPU diagnostic for 32x32 transport proxies, including evaluator preparation"]
fn transport_proxy_cpu_benchmark() {
    use std::{collections::BTreeMap, time::Instant};
    let mut times: BTreeMap<String, (usize, [f64; 2])> = BTreeMap::new();
    for seed in 0..8 {
        for floor in 0..3 {
            for r in program::sample_with_floor(seed, floor).unwrap() {
                if recipe::Recipe::new(&r, floor).is_none() {
                    continue;
                }
                let mut outputs = [Vec::new(), Vec::new()];
                for mode in if seed % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let started = Instant::now();
                    let prepared = (mode == 1).then(|| r.prepare_texels(floor)).flatten();
                    let n = r.map_size(floor);
                    for y in 0..32 {
                        for x in 0..32 {
                            let t = r.texel_prepared(
                                (x * n / 32) as f32 / n as f32,
                                (y * n / 32) as f32 / n as f32,
                                floor,
                                prepared.as_ref(),
                            );
                            outputs[mode].push(t.color.map(|c| (c.clamp(0., 1.) * 255.) as u8));
                        }
                    }
                    let row = times.entry(format!("{:?}", r.surface)).or_default();
                    row.0 += usize::from(mode == 0);
                    row.1[mode] += started.elapsed().as_secs_f64();
                }
                assert_eq!(outputs[0], outputs[1]);
            }
        }
    }
    println!("{}", serde_json::to_string(&times).unwrap());
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
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
        source: wgpu::ShaderSource::Wgsl(kernel::SHADER.into()),
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
            let mut recipes = program::sample_with_floor(seed, floor).unwrap();
            if seed == 0 {
                // Every expanded process pole, not only whichever blended pole
                // a short seed list happened to select. Reuse the same strict
                // CPU oracle for all mip levels and all three material maps.
                let conditioned: Vec<_> = recipes
                    .iter()
                    .filter(|r| {
                        recipe::Recipe::new(r, floor).is_some()
                            && (floor == 0 && r.surface != Surface::Floor
                                || floor == 1 && r.surface == Surface::Floor)
                    })
                    .flat_map(|r| {
                        (0..4).map(move |family| {
                            let mut r = r.clone();
                            let mut weights = [0.; 4];
                            weights[family] = 1.;
                            super::super::construction::apply(
                                &mut r,
                                super::super::construction::ConstructionSample {
                                    weights,
                                    scale: if family % 2 == 0 { 0.5 } else { 2. },
                                    variation: 0.6,
                                },
                                floor,
                            )
                            .unwrap();
                            r
                        })
                    })
                    .collect();
                recipes.extend(conditioned);
            }
            if floor == 0 {
                if seed == 0 {
                    let parent = recipes[Surface::Accent as usize].clone();
                    recipes.extend((0..5).map(|style| {
                        let mut r = parent.clone();
                        let mut w = (0..64)
                            .find_map(|s| {
                                super::super::wall::WallFinish::sample(s, Surface::Accent)
                            })
                            .unwrap();
                        w.print_style = style;
                        w.weights = [0., 0., 0., 1.];
                        r.coating.as_mut().unwrap().wall = Some(w);
                        r
                    }));
                }
                recipes.push(super::super::human::knit_recipe(seed));
                for surface in [Surface::Metal, Surface::Chrome, Surface::Plastic] {
                    let parent = recipes[surface as usize].clone();
                    recipes.extend((1..3).map(|slot| parent.variant(slot)));
                }
                for surface in [Surface::Paint, Surface::Floor] {
                    let mut r = recipes[surface as usize].clone();
                    r.detail_scale = Some(0.);
                    r.relief_m = 0.;
                    recipes.push(r);
                }
                let parent = recipes[Surface::FabricAlt as usize].clone();
                recipes.extend((0..6).map(|pattern| {
                    let mut r = super::super::rugs::recipe(&parent, pattern % 4);
                    let design = r.carpet.as_mut().unwrap().rug.as_mut().unwrap();
                    if pattern < 4 {
                        design.weights = [0.; 4];
                        design.weights[pattern] = 1.;
                    } else if pattern == 4 {
                        design.quiet = 0.98;
                        design.border_width = 0.;
                        design.center = 0.;
                    } else {
                        design.patchwork = 1.;
                    }
                    r
                }));
            }
            for mut r in recipes {
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
                        &pipelines[kernel::NORMALS],
                        &pipelines[kernel::MIPS],
                    ],
                    &bind,
                    n,
                    textures.each_ref().map(Some),
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
                report.push(serde_json::json!({"seed":seed,"floor":floor,"surface":r.surface,"construction":r.construction,"rug":r.carpet.as_ref().and_then(|c|c.rug.as_ref()),"knit":r.textile.as_ref().is_some_and(|t|t.knit),"cpu_synthesis_seconds":cpu_synthesis_seconds,"gpu_bake_seconds":gpu_bake_seconds,"maps":summaries}));
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
