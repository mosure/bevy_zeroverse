use bevy::{
    asset::RenderAssetUsages,
    pbr::StandardMaterial,
    prelude::{Mesh3d, MeshMaterial3d, *},
    render::render_resource::PrimitiveTopology,
    render::renderer::{RenderDevice, RenderQueue, WgpuWrapper},
    MinimalPlugins,
};
use bevy_zeroverse::{
    app::{BevyZeroverseConfig, OvoxelMode},
    ovoxel::{OvoxelExport, OvoxelPlugin, OvoxelVolume},
    ovoxel_mesh::{ovoxel_to_mesh, write_mesh_as_glb},
    scene::RegenerateSceneEvent,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tempfile::TempDir;
const MAX_FRAMES_PER_ITER: usize = 50_000;

fn cube_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
    );
    mesh.insert_indices(bevy_mesh::Indices::U32(vec![
        0, 1, 2, 0, 2, 3, // back
        4, 6, 5, 4, 7, 6, // front
        0, 4, 5, 0, 5, 1, // bottom
        3, 2, 6, 3, 6, 7, // top
        1, 5, 6, 1, 6, 2, // right
        0, 3, 7, 0, 7, 4, // left
    ]));
    mesh
}

fn dense_volume(resolution: u32) -> OvoxelVolume {
    // Fill a centered cube occupying half the resolution to keep volume moderate.
    let half = resolution / 2;
    let start = half / 2;
    let end = start + half;
    let mut coords = Vec::new();
    let mut dual_vertices = Vec::new();
    let mut intersected = Vec::new();
    let mut base_color = Vec::new();
    let mut semantics = Vec::new();
    for x in start..end {
        for y in start..end {
            for z in start..end {
                coords.push([x, y, z]);
                dual_vertices.push([128, 128, 128]);
                intersected.push(0);
                base_color.push([200, 120, 80, 255]);
                semantics.push(1);
            }
        }
    }

    OvoxelVolume {
        coords,
        dual_vertices,
        intersected,
        base_color,
        semantics,
        semantic_labels: vec!["unlabeled".into(), "block".into()],
        resolution,
        aabb: [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    }
}

fn bench_modes() -> Vec<(String, OvoxelMode)> {
    let env = std::env::var("OVOXEL_BENCH_MODES")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_lowercase();
    let mut modes = Vec::new();
    for token in env.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match token {
            "cpu" => modes.push(("cpu".to_string(), OvoxelMode::CpuAsync)),
            "gpu" => modes.push(("gpu".to_string(), OvoxelMode::GpuCompute)),
            "both" => {
                modes.push(("cpu".to_string(), OvoxelMode::CpuAsync));
                modes.push(("gpu".to_string(), OvoxelMode::GpuCompute));
            }
            _ => {}
        }
    }
    if modes.is_empty() {
        modes.push(("cpu".to_string(), OvoxelMode::CpuAsync));
    }
    modes
}

fn wait_for_volume(app: &mut App, root: Entity, label: &str, res: u32) -> OvoxelVolume {
    for frame in 0..MAX_FRAMES_PER_ITER {
        app.update();
        if let Some(v) = app.world().entity(root).get::<OvoxelVolume>() {
            return v.clone();
        }
        if frame % 128 == 0 {
            std::thread::yield_now();
        }
    }
    panic!("volume should be populated after updates (mode {label}, res {res})");
}

fn gpu_device_and_queue() -> Option<(RenderDevice, RenderQueue)> {
    let instance = wgpu::Instance::default();
    let adapter =
        futures_lite::future::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
    let info = adapter.get_info();
    if matches!(info.device_type, wgpu::DeviceType::Cpu) {
        println!("Skipping GPU benches: adapter is {:?}", info.device_type);
        return None;
    }
    let device_desc = wgpu::DeviceDescriptor {
        label: Some("ovoxel_bench_device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::default(),
    };
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&device_desc)).ok()?;
    Some((
        RenderDevice::from(device),
        RenderQueue(WgpuWrapper::new(queue).into()),
    ))
}

fn ovoxel_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ovoxel_creation");
    group.sample_size(10);
    let gpu_ctx = gpu_device_and_queue();

    for &(ref label, mode) in bench_modes().iter() {
        if matches!(mode, OvoxelMode::GpuCompute) && gpu_ctx.is_none() {
            println!("Skipping GPU benches: no compatible adapter/device available on this host");
            continue;
        }
        for &res in &[64u32, 96u32, 128u32, 256u32, 512u32] {
            let bench_name = format!("{label}/voxelize_res_{res}");
            group.bench_function(bench_name, |b| {
                b.iter_batched(
                    || {
                        let mut app = App::new();
                        app.add_plugins(MinimalPlugins);
                        app.add_message::<RegenerateSceneEvent>();
                        app.insert_resource(BevyZeroverseConfig {
                            ovoxel_mode: mode,
                            ..Default::default()
                        });
                        app.add_plugins(OvoxelPlugin);
                        if let Some((device, queue)) = gpu_ctx.clone() {
                            app.insert_resource(device);
                            app.insert_resource(queue);
                        }
                        app.insert_resource(Assets::<Mesh>::default());
                        app.insert_resource(Assets::<StandardMaterial>::default());

                        let mesh_handle = {
                            let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
                            meshes.add(cube_mesh())
                        };
                        let mat_handle = {
                            let mut materials =
                                app.world_mut().resource_mut::<Assets<StandardMaterial>>();
                            materials.add(StandardMaterial {
                                base_color: Color::srgb(0.6, 0.3, 0.2),
                                ..Default::default()
                            })
                        };

                        let root = app
                            .world_mut()
                            .spawn((
                                OvoxelExport {
                                    resolution: res,
                                    aabb: Some(([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])),
                                },
                                Transform::IDENTITY,
                                GlobalTransform::IDENTITY,
                            ))
                            .id();
                        let child = app
                            .world_mut()
                            .spawn((
                                Mesh3d(mesh_handle),
                                MeshMaterial3d(mat_handle),
                                Transform::IDENTITY,
                                GlobalTransform::IDENTITY,
                            ))
                            .id();
                        app.world_mut().entity_mut(root).add_child(child);
                        (app, root)
                    },
                    |(mut app, root)| {
                        let volume = wait_for_volume(&mut app, root, label, res);
                        std::hint::black_box(volume.coords.len());
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

fn ovoxel_conversion_benchmark(c: &mut Criterion) {
    let volume = dense_volume(96);
    let tmp = TempDir::new().unwrap();
    let glb_path = tmp.path().join("ovoxel_bench.glb");

    let mut group = c.benchmark_group("ovoxel_conversion");
    group.sample_size(20);

    group.bench_function("mesh_from_volume", |b| {
        b.iter(|| {
            let mesh = ovoxel_to_mesh(&volume);
            std::hint::black_box(mesh);
        });
    });

    group.bench_function("glb_from_volume", |b| {
        b.iter(|| {
            let mesh = ovoxel_to_mesh(&volume);
            write_mesh_as_glb(&mesh, &glb_path).expect("glb write");
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    ovoxel_creation_benchmark,
    ovoxel_conversion_benchmark
);
criterion_main!(benches);
