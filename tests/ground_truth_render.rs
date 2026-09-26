//! Independent f64 ray/triangle reference for the native float32 MRT renderer.
#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]

use bevy::{
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    math::{DMat4, DVec3},
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureFormat, TextureUsages},
        renderer::RenderDevice,
    },
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless::{create_app, setup_globals},
    io::image_copy::{CapturedImages, ImageCopier},
    render::{
        ground_truth::{semantic_id, GroundTruthCamera, GroundTruthDiagnostics},
        semantic::SemanticLabel,
    },
    scene::ZeroverseSceneType,
};
use std::time::{Duration, Instant};

struct ReferenceMesh {
    entity: Entity,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<usize>,
    transform: Transform,
    label: SemanticLabel,
}

fn spawn_mesh(
    app: &mut App,
    mesh: Mesh,
    transform: Transform,
    label: SemanticLabel,
    glass: bool,
) -> ReferenceMesh {
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        panic!()
    };
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        panic!()
    };
    let reference = (
        positions.clone(),
        normals.clone(),
        mesh.indices().unwrap().iter().collect(),
    );
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            cull_mode: None,
            double_sided: true,
            base_color: Color::srgba(0.4, 0.5, 0.6, if glass { 0.08 } else { 1.0 }),
            alpha_mode: if glass {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            unlit: true,
            ..default()
        });
    let entity = app
        .world_mut()
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            transform,
            label.clone(),
        ))
        .id();
    ReferenceMesh {
        entity,
        positions: reference.0,
        normals: reference.1,
        indices: reference.2,
        transform,
        label,
    }
}

fn capture(app: &mut App, copier: &ImageCopier, id: u64) -> CapturedImages {
    copier.request(id);
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            copier.failure().is_none(),
            "readback failed: {:?}",
            copier.failure()
        );
        if let Some(packet) = copier.take(id) {
            return packet;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "ground-truth request stalled"
        );
    }
}

fn dmatrix(matrix: Mat4) -> DMat4 {
    DMat4::from_cols_array(&matrix.to_cols_array().map(|v| v as f64))
}

/// CPU reference uses f64 mesh intersections, independent of GPU-produced positions/depth.
fn check_reference(
    packet: &CapturedImages,
    meshes: &[ReferenceMesh],
    camera: Transform,
    fov: f32,
    size: UVec2,
) {
    assert_eq!(packet.planes.len(), 3);
    let world_depth: Vec<f32> = packet.planes[1]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_ne_bytes(*p))
        .collect();
    let normal_semantic: Vec<f32> = packet.planes[2]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| f32::from_ne_bytes(*p))
        .collect();
    assert_eq!(world_depth.len(), size.x as usize * size.y as usize * 4);
    let world_from_view = dmatrix(camera.to_matrix());
    let view_from_world = world_from_view.inverse();
    let origin = world_from_view.transform_point3(DVec3::ZERO);
    let tangent = (fov as f64 * 0.5).tan();
    let mut checked = 0;
    let mut seen = std::collections::BTreeSet::new();
    let mut max_position = 0.0_f64;
    let mut max_depth = 0.0_f64;
    let mut max_reprojection = 0.0_f64;
    let mut max_normal = 0.0_f64;
    let mut max_flat_normal = 0.0_f64;
    let mut precise_depth_values = 0;
    for y in (7..size.y - 7).step_by(11) {
        for x in (7..size.x - 7).step_by(11) {
            let ray_view = DVec3::new(
                (2.0 * (x as f64 + 0.5) / size.x as f64 - 1.0) * tangent * size.x as f64
                    / size.y as f64,
                (1.0 - 2.0 * (y as f64 + 0.5) / size.y as f64) * tangent,
                -1.0,
            );
            let ray = world_from_view.transform_vector3(ray_view).normalize();
            let mut nearest: Option<(f64, DVec3, u32, f64)> = None;
            for object in meshes {
                let model = dmatrix(object.transform.to_matrix());
                let normal_model = model.inverse().transpose();
                for triangle in object.indices.as_chunks::<3>().0 {
                    let positions = triangle.map(|i| {
                        model.transform_point3(Vec3::from_array(object.positions[i]).as_dvec3())
                    });
                    let [a, b, c] = positions;
                    let edge1 = b - a;
                    let edge2 = c - a;
                    let p = ray.cross(edge2);
                    let determinant = edge1.dot(p);
                    if determinant.abs() < 1e-12 {
                        continue;
                    }
                    let offset = origin - a;
                    let u = offset.dot(p) / determinant;
                    let q = offset.cross(edge1);
                    let v = ray.dot(q) / determinant;
                    let distance = edge2.dot(q) / determinant;
                    if u < 0.0 || v < 0.0 || u + v > 1.0 || distance <= 0.0 {
                        continue;
                    }
                    if nearest.as_ref().is_some_and(|hit| distance >= hit.0) {
                        continue;
                    }
                    let normals = triangle.map(|i| {
                        normal_model
                            .transform_vector3(Vec3::from_array(object.normals[i]).as_dvec3())
                            .normalize()
                    });
                    let normal =
                        (normals[0] * (1.0 - u - v) + normals[1] * u + normals[2] * v).normalize();
                    nearest = Some((
                        distance,
                        normal,
                        semantic_id(&object.label),
                        u.min(v).min(1.0 - u - v),
                    ));
                }
            }
            let offset = (y * size.x + x) as usize * 4;
            let wd = &world_depth[offset..offset + 4];
            let ns = &normal_semantic[offset..offset + 4];
            let Some((distance, normal, semantic, barycentric_margin)) = nearest else {
                assert_eq!(wd[3], 0.0, "unexpected foreground at {x},{y}");
                continue;
            };
            if barycentric_margin < 1e-4 {
                continue;
            }
            let point = origin + distance * ray;
            let depth = -view_from_world.transform_point3(point).z;
            let actual = DVec3::new(wd[0] as f64, wd[1] as f64, wd[2] as f64);
            max_position = max_position.max(actual.distance(point));
            max_depth = max_depth.max((wd[3] as f64 - depth).abs());
            precise_depth_values += usize::from(half::f16::from_f32(wd[3]).to_f32() != wd[3]);
            assert_eq!(ns[3], semantic as f32, "semantic mismatch at {x},{y}");
            let expected_normal = view_from_world.transform_vector3(normal).normalize();
            let actual_normal =
                DVec3::new(ns[0] as f64, ns[1] as f64, ns[2] as f64) * 2.0 - DVec3::ONE;
            let normal_error = actual_normal.distance(expected_normal);
            max_normal = max_normal.max(normal_error);
            if semantic != semantic_id(&SemanticLabel::OtherProp) {
                max_flat_normal = max_flat_normal.max(normal_error);
            }
            let p = view_from_world.transform_point3(actual);
            let pixel = DVec3::new(
                (p.x / (-p.z * tangent) * size.y as f64 / size.x as f64 + 1.0)
                    * size.x as f64
                    * 0.5,
                (1.0 - p.y / (-p.z * tangent)) * size.y as f64 * 0.5,
                0.0,
            );
            max_reprojection =
                max_reprojection.max((pixel.x - x as f64 - 0.5).hypot(pixel.y - y as f64 - 0.5));
            checked += 1;
            seen.insert(semantic);
        }
    }
    assert!(
        checked >= 80,
        "too few independent surface samples: {checked}"
    );
    assert!(
        seen.len() >= 3,
        "insufficient semantic surface coverage: {seen:?}"
    );
    println!(
        "request {}: {checked} f64 ray-triangle references, worldmax={max_position:.9}m depthmax={max_depth:.9}m reprojectionmax={max_reprojection:.6}px normalmax={max_normal:.8} flatnormalmax={max_flat_normal:.8} beyond_f16_depth={precise_depth_values}",
        packet.request_id
    );
    // Rasterizers discretize subpixel vertex positions when setting up triangles.
    // These independent ideal-pixel-center intersections therefore allow 0.005px
    // of raster error, including its depth/smooth-normal variation on the sphere.
    // The 100um scene-space bound is still well below f16 spacing at these depths;
    // flat normals have no interpolation gradient and use a stricter f32 bound.
    assert!(max_position < 1e-4, "world position error {max_position}m");
    assert!(max_depth < 1e-4, "linear depth error {max_depth}m");
    assert!(
        max_reprojection < 0.005,
        "reprojection error {max_reprojection}px"
    );
    assert!(max_normal < 3e-4, "geometric normal error {max_normal}");
    assert!(
        max_flat_normal < 2e-6,
        "flat normal error {max_flat_normal}"
    );
    assert!(
        precise_depth_values * 10 > checked * 9,
        "depth appears quantized to float16"
    );
}

#[test]
#[ignore = "requires a native GPU; checks analytic accuracy and cache invalidation"]
fn float32_mrt_matches_independent_ray_geometry_and_tracks_changes() {
    let empty_assets = tempfile::tempdir().unwrap();
    setup_globals(Some(empty_assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        initialize_scene: false,
        headless: true,
        editor: false,
        gizmos: false,
        image_copiers: true,
        num_cameras: 0,
        keybinds: false,
        press_esc_close: false,
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    // Subprocess-only diagnostic for renderer thread teardown. Removing this
    // sub-app before plugin cleanup uses Bevy's supported synchronous path.
    let synchronous = std::env::var_os("ZERO_VERSE_TEST_SYNCHRONOUS_RENDER").is_some();
    if synchronous {
        app.remove_sub_app(bevy::render::pipelined_rendering::RenderExtractApp);
    }
    app.finish();
    app.cleanup();
    assert_eq!(
        app.get_sub_app(bevy::render::RenderApp).is_some(),
        synchronous
    );
    eprintln!("teardown control: synchronous_render={synchronous}");
    let mut plane = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    plane.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-1.5, -1.2, 0.0],
            [1.5, -1.2, 0.0],
            [1.5, 1.2, 0.0],
            [-1.5, 1.2, 0.0],
        ],
    );
    plane.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 4]);
    plane.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    );
    plane.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let mut objects = vec![
        spawn_mesh(
            &mut app,
            plane,
            Transform::from_xyz(0.0, 1.2, -0.2)
                .with_rotation(Quat::from_rotation_y(0.31) * Quat::from_rotation_z(-0.09))
                .with_scale(Vec3::new(1.7, 0.9, 1.0)),
            SemanticLabel::Wall,
            false,
        ),
        spawn_mesh(
            &mut app,
            Mesh::from(Cuboid::new(1.0, 1.0, 1.0)),
            Transform::from_xyz(0.55, 0.75, 0.9)
                .with_rotation(Quat::from_rotation_y(-0.48))
                .with_scale(Vec3::new(0.7, 1.1, 0.55)),
            SemanticLabel::Chair,
            false,
        ),
        spawn_mesh(
            &mut app,
            Mesh::from(Cuboid::new(0.50, 0.8, 0.04)),
            Transform::from_xyz(-0.68, 1.25, 0.75),
            SemanticLabel::Window,
            true,
        ),
        spawn_mesh(
            &mut app,
            Sphere::new(0.32).mesh().uv(20, 12),
            Transform::from_xyz(-0.50, 0.45, 1.1).with_scale(Vec3::new(1.2, 1.0, 0.8)),
            SemanticLabel::OtherProp,
            false,
        ),
    ];
    let hidden = spawn_mesh(
        &mut app,
        Mesh::from(Cuboid::new(7.0, 5.0, 0.1)),
        Transform::from_xyz(0.0, 1.0, 2.5),
        SemanticLabel::OtherFurniture,
        false,
    );
    app.world_mut()
        .entity_mut(hidden.entity)
        .insert(Visibility::Hidden);
    let size = UVec2::new(321, 239);
    let fov = 61.0_f32.to_radians();
    let camera_transform =
        Transform::from_xyz(0.12, 1.55, 4.65).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    let (rgb, ground_truth) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut rgb = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
        rgb.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let rgb = images.add(rgb);
        let gt = GroundTruthCamera::new(&mut images, size);
        (rgb, gt)
    };
    let copier = ImageCopier::for_targets(
        vec![
            rgb.clone(),
            ground_truth.world_depth.clone(),
            ground_truth.normal_semantic.clone(),
        ],
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        app.world().resource::<RenderDevice>(),
    );
    let camera_entity = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(ImageRenderTarget::from(rgb)),
            Projection::Perspective(PerspectiveProjection {
                fov,
                near: 0.1,
                far: 50.0,
                ..default()
            }),
            camera_transform,
            Msaa::Off,
            ground_truth,
            copier.clone(),
        ))
        .id();
    for _ in 0..20 {
        app.update();
    }
    assert_eq!(
        app.world()
            .resource::<GroundTruthDiagnostics>()
            .snapshot()
            .rendered_views,
        0,
        "idle frames must not rasterize ground truth"
    );
    check_reference(
        &capture(&mut app, &copier, 1),
        &objects,
        camera_transform,
        fov,
        size,
    );
    let before = app.world().resource::<GroundTruthDiagnostics>().snapshot();
    for _ in 0..8 {
        app.update();
    }
    let idle = app.world().resource::<GroundTruthDiagnostics>().snapshot();
    assert_eq!(idle.rendered_views, before.rendered_views);
    assert_eq!(idle.geometry_uploads, before.geometry_uploads);
    objects[1].transform.translation.x += 0.27;
    objects[1].transform.rotate_y(0.2);
    app.world_mut()
        .entity_mut(objects[1].entity)
        .insert(objects[1].transform);
    for _ in 0..3 {
        app.update();
    }
    check_reference(
        &capture(&mut app, &copier, 2),
        &objects,
        camera_transform,
        fov,
        size,
    );
    let moved = app.world().resource::<GroundTruthDiagnostics>().snapshot();
    assert_eq!(
        moved.geometry_uploads, before.geometry_uploads,
        "rigid motion must not rebuild vertices"
    );
    assert!(moved.instance_updates > before.instance_updates);
    let removed = objects.remove(3);
    app.world_mut().despawn(removed.entity);
    for _ in 0..3 {
        app.update();
    }
    check_reference(
        &capture(&mut app, &copier, 3),
        &objects,
        camera_transform,
        fov,
        size,
    );
    let final_stats = app.world().resource::<GroundTruthDiagnostics>().snapshot();
    assert!(
        final_stats.vertices < moved.vertices,
        "despawned geometry remains resident"
    );
    assert!(final_stats.geometry_uploads > moved.geometry_uploads);

    let meshes: Vec<_> = app
        .world_mut()
        .query_filtered::<Entity, With<Mesh3d>>()
        .iter(app.world())
        .collect();
    for entity in meshes {
        app.world_mut().despawn(entity);
    }
    for _ in 0..3 {
        app.update();
    }
    let empty = capture(&mut app, &copier, 4);
    assert_eq!(empty.request_id, 4);
    assert_eq!(empty.planes.len(), 3);
    for plane in &empty.planes[1..] {
        assert_eq!(plane.len(), size.x as usize * size.y as usize * 16);
        assert!(
            plane.iter().all(|byte| *byte == 0),
            "empty geometry must clear both annotation targets"
        );
    }
    assert_eq!(
        app.world()
            .get::<GroundTruthCamera>(camera_entity)
            .unwrap()
            .rendered_frame(),
        Some(4),
    );
    let empty_stats = app.world().resource::<GroundTruthDiagnostics>().snapshot();
    assert_eq!(empty_stats.geometry_bytes, 0);
    assert_eq!(empty_stats.vertices, 0);
    assert_eq!(empty_stats.triangles, 0);
    assert_eq!(empty_stats.mesh_instances, 0);
    assert_eq!(empty_stats.geometry_uploads, final_stats.geometry_uploads);
    assert_eq!(empty_stats.rendered_views, final_stats.rendered_views + 1);
    println!(
        "request 4: empty scene completed; both annotation planes zero; geometry residency zero"
    );
}
