use bevy::{
    asset::RenderAssetUsages,
    pbr::StandardMaterial,
    prelude::{Mesh3d, MeshMaterial3d, *},
    render::render_resource::PrimitiveTopology,
    MinimalPlugins,
};
use bevy_zeroverse::{
    ovoxel::{OvoxelExport, OvoxelPlugin, OvoxelVolume},
    ovoxel_mesh::{ovoxel_to_mesh, write_mesh_as_glb},
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use tempfile::TempDir;

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

fn ovoxel_creation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ovoxel_creation");
    group.sample_size(10);

    for &res in &[64u32, 96u32, 128u32] {
        group.bench_function(format!("voxelize_res_{res}"), |b| {
            b.iter_batched(
                || {
                    let mut app = App::new();
                    app.add_plugins(MinimalPlugins);
                    app.add_plugins(OvoxelPlugin);
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
                    app.update();
                    let volume = app
                        .world()
                        .entity(root)
                        .get::<OvoxelVolume>()
                        .expect("volume should be populated");
                    std::hint::black_box(volume.coords.len());
                },
                BatchSize::SmallInput,
            );
        });
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
