#![cfg(not(target_arch = "wasm32"))]
#![recursion_limit = "256"]
//! Independent ray/quad reference; production GPU projection/occlusion code is
//! deliberately not called by the oracle.
use bevy::{
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::{render_resource::*, renderer::RenderDevice},
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{CaptureCameraIndex, ZeroverseCamera},
    headless::{create_app, setup_globals},
    io::image_copy::{CapturedImages, ImageCopier},
    render::{
        co_visibility::{validate_plane, CoVisibilityDiagnostics, CoVisibilityLegend},
        ground_truth::GroundTruthCamera,
    },
};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Plane {
    transform: Transform,
    half: Vec2,
    entity: Entity,
}
struct Camera {
    index: usize,
    transform: Transform,
    size: UVec2,
    entity: Entity,
    copier: ImageCopier,
    fov: f32,
    optical_center: Vec2,
}

fn enable_manual_capture(app: &mut App) {
    // These fixtures own camera creation and copier request stamps. Disable
    // the production sampler's room-readiness/activity gate after creating the
    // windowless app, just as the analytic fisheye fixture does.
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .headless = false;
    app.world_mut()
        .resource_mut::<bevy_zeroverse::sample::SamplerState>()
        .enabled = false;
}

fn spawn_plane(app: &mut App, transform: Transform, half: Vec2) -> Plane {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-half.x, -half.y, 0.0],
            [half.x, -half.y, 0.0],
            [half.x, half.y, 0.0],
            [-half.x, half.y, 0.0],
        ],
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 4]);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    );
    mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            unlit: true,
            cull_mode: None,
            ..default()
        });
    let entity = app
        .world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material), transform))
        .id();
    Plane {
        transform,
        half,
        entity,
    }
}

fn spawn_camera(
    app: &mut App,
    index: usize,
    transform: Transform,
    size: UVec2,
    flow: bool,
) -> Camera {
    spawn_calibrated_camera(app, index, transform, size, flow, 1.0, Vec2::ZERO)
}

fn spawn_calibrated_camera(
    app: &mut App,
    index: usize,
    transform: Transform,
    size: UVec2,
    flow: bool,
    fov: f32,
    offset: Vec2,
) -> Camera {
    let (rgb, gt, targets) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut rgb = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
        rgb.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let rgb = images.add(rgb);
        let mut gt = GroundTruthCamera::new(&mut images, size);
        let mut targets = vec![
            rgb.clone(),
            gt.world_depth.clone(),
            gt.normal_semantic.clone(),
        ];
        if flow {
            targets.push(gt.enable_flow(&mut images));
        }
        targets.push(gt.enable_co_visibility(&mut images, size));
        (rgb, gt, targets)
    };
    let copier = ImageCopier::for_targets(
        targets,
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        app.world().resource::<RenderDevice>(),
    );
    let entity = app
        .world_mut()
        .spawn((
            bevy::prelude::Camera::default(),
            Camera3d::default(),
            RenderTarget::Image(ImageRenderTarget::from(rgb)),
            bevy_zeroverse::camera::pinhole::projection(
                PerspectiveProjection {
                    fov,
                    near: 0.1,
                    far: 50.0,
                    ..default()
                },
                offset,
            ),
            transform,
            Msaa::Off,
            gt,
            copier.clone(),
            CaptureCameraIndex(index),
            ZeroverseCamera {
                override_transform: Some(transform),
                principal_point_offset: offset,
                ..default()
            },
        ))
        .id();
    Camera {
        index,
        transform,
        size,
        entity,
        copier,
        fov,
        optical_center: (Vec2::splat(0.5) + offset) * size.as_vec2(),
    }
}

fn capture(app: &mut App, cameras: &[Camera], stamp: u64) -> Vec<CapturedImages> {
    for c in cameras {
        c.copier.request(stamp);
    }
    let start = Instant::now();
    loop {
        app.update();
        for c in cameras {
            assert!(c.copier.failure().is_none(), "{:?}", c.copier.failure());
        }
        if cameras.iter().all(|c| c.copier.ready(stamp)) {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "co-visibility capture stalled: {:?}, {:?}",
            app.world().resource::<CoVisibilityDiagnostics>().snapshot(),
            cameras
                .iter()
                .map(|c| {
                    let gt = app.world().get::<GroundTruthCamera>(c.entity).unwrap();
                    (
                        gt.frame_id,
                        gt.rendered_frame(),
                        gt.co_visibility.as_ref().unwrap().rendered_frame(),
                        gt.failure(),
                    )
                })
                .collect::<Vec<_>>()
        );
    }
    cameras
        .iter()
        .map(|c| c.copier.take(stamp).unwrap())
        .collect()
}
fn pixels(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|p| bytemuck::pod_read_unaligned(p))
        .collect()
}

// Intersect the ideal pixel-center ray in f64 with every finite analytic plane.
fn hit(camera: &Camera, pixel: Vec2, planes: &[Plane]) -> Option<(usize, f64)> {
    let world = camera.transform.to_matrix().as_dmat4();
    let origin = world.transform_point3(bevy::math::DVec3::ZERO);
    let f = camera.size.y as f64 / (2.0 * (camera.fov as f64 * 0.5).tan());
    let direction = world.transform_vector3(bevy::math::DVec3::new(
        (pixel.x as f64 - camera.optical_center.x as f64) / f,
        -(pixel.y as f64 - camera.optical_center.y as f64) / f,
        -1.0,
    ));
    let mut nearest = None;
    for (i, p) in planes.iter().enumerate() {
        let inverse = p.transform.to_matrix().as_dmat4().inverse();
        let o = inverse.transform_point3(origin);
        let d = inverse.transform_vector3(direction);
        if d.z.abs() < 1e-9 {
            continue;
        }
        let t = -o.z / d.z;
        let q = o + d * t;
        if t < 0.1 || t > 50.0 || q.x.abs() > p.half.x as f64 || q.y.abs() > p.half.y as f64 {
            continue;
        }
        if nearest.is_none_or(|(_, depth)| t < depth) {
            nearest = Some((i, t));
        }
    }
    nearest
}

fn verify(
    cameras: &[Camera],
    packets: &[CapturedImages],
    planes: &[Plane],
) -> (usize, usize, usize) {
    let mut checked = 0;
    let mut shared = 0;
    let mut occluded = 0;
    for (source, (camera, packet)) in cameras.iter().zip(packets).enumerate() {
        let wd = pixels(&packet.planes[1]);
        let annotations = packet.planes.last().unwrap();
        validate_plane(annotations, wd.len(), cameras.len(), source).unwrap();
        for (p, annotation) in pixels(annotations).into_iter().enumerate() {
            if p_is_background(p, &wd) {
                assert_eq!(annotation, [0.0; 4]);
                continue;
            }
            let source_pixel = Vec2::new(
                (p % camera.size.x as usize) as f32 + 0.5,
                (p / camera.size.x as usize) as f32 + 0.5,
            );
            let Some((surface, _)) = hit(camera, source_pixel, planes) else {
                continue;
            };
            let point = Vec3::new(wd[p][0], wd[p][1], wd[p][2]);
            let mut expected = 0u16;
            for (target, other) in cameras.iter().enumerate() {
                if target == source {
                    continue;
                }
                let q = other
                    .transform
                    .to_matrix()
                    .inverse()
                    .transform_point3(point);
                if -q.z < 0.1 || -q.z > 50.0 {
                    continue;
                }
                let focal = other.size.y as f32 / (2.0 * (other.fov * 0.5).tan());
                let pixel = Vec2::new(
                    q.x / -q.z * focal + other.optical_center.x,
                    -q.y / -q.z * focal + other.optical_center.y,
                );
                if pixel.x < 0.0
                    || pixel.y < 0.0
                    || pixel.x >= other.size.x as f32
                    || pixel.y >= other.size.y as f32
                {
                    continue;
                }
                if hit(other, pixel.floor() + Vec2::splat(0.5), planes)
                    .is_some_and(|(id, _)| id == surface)
                {
                    expected |= 1 << target;
                    shared += 1;
                } else {
                    occluded += 1;
                }
            }
            assert_eq!(
                annotation[0] as u16, expected,
                "source {source}, pixel {p}, point {point:?}"
            );
            checked += 1;
        }
    }
    (checked, shared, occluded)
}
fn p_is_background(p: usize, wd: &[[f32; 4]]) -> bool {
    wd[p][3] == 0.0
}

#[test]
#[ignore = "native GPU: RGB silhouettes and occlusion for wide/shifted/portrait pinhole views"]
fn wide_and_shifted_cameras_match_rgb_and_independent_surface_visibility() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: true,
            num_cameras: 0,
            ssaa_factor: 1,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    enable_manual_capture(&mut app);
    let planes = vec![
        spawn_plane(
            &mut app,
            Transform::from_xyz(0., 0., -5.),
            Vec2::new(8., 6.),
        ),
        spawn_plane(
            &mut app,
            Transform::from_xyz(0., 0., -2.8),
            Vec2::new(0.35, 0.65),
        ),
        spawn_plane(
            &mut app,
            Transform::from_xyz(1.1, 0., -4.).with_rotation(Quat::from_rotation_y(0.45)),
            Vec2::new(0.8, 1.5),
        ),
    ];
    for (channel, plane) in planes.iter().enumerate() {
        let material = app
            .world()
            .get::<MeshMaterial3d<StandardMaterial>>(plane.entity)
            .unwrap()
            .0
            .clone();
        let mut rgb = [0.; 3];
        rgb[channel] = 0.25;
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .get_mut(&material)
            .unwrap()
            .base_color = Color::linear_rgb(rgb[0], rgb[1], rgb[2]);
    }
    let cases = [
        (145_f32, UVec2::new(129, 97), Vec2::ZERO, -0.5),
        (70., UVec2::new(97, 97), Vec2::ZERO, 0.9),
        (130., UVec2::new(129, 73), Vec2::new(0.17, -0.12), 0.3),
        (100., UVec2::new(73, 129), Vec2::new(-0.15, 0.08), -0.9),
    ];
    let cameras: Vec<_> = cases
        .into_iter()
        .enumerate()
        .map(|(i, (degrees, size, offset, x))| {
            spawn_calibrated_camera(
                &mut app,
                i,
                Transform::from_xyz(x, 0., 0.),
                size,
                false,
                degrees.to_radians(),
                offset,
            )
        })
        .collect();
    for _ in 0..12 {
        app.update();
    }
    let packets = capture(&mut app, &cameras, 1);
    let metrics = verify(&cameras, &packets, &planes);
    let mut rgb_checked = 0;
    for (camera, packet) in cameras.iter().zip(&packets) {
        let rgb = pixels(&packet.planes[0]);
        for (i, color) in rgb.iter().enumerate() {
            let pixel = Vec2::new(
                (i % camera.size.x as usize) as f32 + 0.5,
                (i / camera.size.x as usize) as f32 + 0.5,
            );
            let Some((surface, _)) = hit(camera, pixel, &planes) else {
                continue;
            };
            // RGB may filter silhouettes; require an interior footprint without
            // excluding these boundary pixels from the visibility oracle above.
            if [Vec2::X, -Vec2::X, Vec2::Y, -Vec2::Y]
                .into_iter()
                .any(|offset| {
                    hit(camera, pixel + offset, &planes).is_none_or(|(other, _)| other != surface)
                })
            {
                continue;
            }
            let observed = (0..3)
                .max_by(|&a, &b| color[a].total_cmp(&color[b]))
                .unwrap();
            assert!(
                color[surface] > 0.005 && observed == surface,
                "RGB/geometry mismatch: view {}, pixel {pixel}, surface {surface}, RGB {color:?}",
                camera.index
            );
            rgb_checked += 1;
        }
    }
    assert!(
        metrics.0 > 10_000 && metrics.1 > 5_000 && metrics.2 > 100 && rgb_checked > 10_000,
        "{metrics:?}, RGB {rgb_checked}"
    );
    println!("wide/shifted pinhole: {metrics:?}; {rgb_checked} RGB surface checks");
}

#[test]
#[ignore = "native GPU: independent occlusion, camera order, motion, flow and 16-camera packing"]
fn same_time_visibility_matches_ray_geometry() {
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            initialize_scene: false,
            headless: true,
            // An editor marker is inserted explicitly below. The interactive
            // editor's grid-visibility controller must not disable this fixture.
            editor: false,
            gizmos: false,
            keybinds: false,
            press_esc_close: false,
            image_copiers: true,
            num_cameras: 0,
            ..default()
        }),
        false,
    );
    app.finish();
    app.cleanup();
    enable_manual_capture(&mut app);
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(
        app.world()
            .resource::<CoVisibilityDiagnostics>()
            .snapshot()
            .pipeline_initializations,
        0
    );
    let mut planes = vec![
        spawn_plane(
            &mut app,
            Transform::from_xyz(0.0, 0.0, -5.0),
            Vec2::new(4.0, 3.0),
        ),
        spawn_plane(
            &mut app,
            Transform::from_xyz(1.1, 0.0, -4.0).with_rotation(Quat::from_rotation_y(0.45)),
            Vec2::new(0.8, 1.5),
        ),
        spawn_plane(
            &mut app,
            Transform::from_xyz(0.0, 0.0, -2.8),
            Vec2::new(0.35, 0.65),
        ),
    ];
    let size = UVec2::new(97, 73);
    let mut cameras = vec![
        spawn_camera(
            &mut app,
            10,
            Transform::from_xyz(-0.5, 0.0, 0.0),
            size,
            true,
        ),
        spawn_camera(&mut app, 2, Transform::from_xyz(0.9, 0.1, 0.0), size, true),
        spawn_camera(
            &mut app,
            7,
            Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
            size,
            true,
        ),
    ];
    cameras.sort_by_key(|c| c.index);
    // An editor-tagged view at the same position must not become another bit.
    app.world_mut().spawn((
        bevy_zeroverse::camera::EditorCameraMarker::default(),
        bevy_zeroverse::camera::ProcessedEditorCameraMarker,
    ));
    for _ in 0..10 {
        app.update();
    }
    let first = capture(&mut app, &cameras, 1);
    let metrics = verify(&cameras, &first, &planes);
    assert!(
        metrics.0 > 10000 && metrics.1 > 5000 && metrics.2 > 100,
        "{metrics:?}"
    );
    assert_eq!(
        app.world().resource::<CoVisibilityLegend>().camera_indices,
        vec![2, 7, 10]
    );
    let stats = app.world().resource::<CoVisibilityDiagnostics>().snapshot();
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(
        stats.dispatches,
        app.world()
            .resource::<CoVisibilityDiagnostics>()
            .snapshot()
            .dispatches,
        "idle frames must not dispatch"
    );
    planes[2].transform.translation.x += 0.7;
    app.world_mut()
        .entity_mut(planes[2].entity)
        .insert(planes[2].transform);
    let second = capture(&mut app, &cameras, 2);
    let moved = verify(&cameras, &second, &planes);
    assert_ne!(first[0].planes[4], second[0].planes[4]);
    assert_eq!(
        stats.atlas_allocations,
        app.world()
            .resource::<CoVisibilityDiagnostics>()
            .snapshot()
            .atlas_allocations
    );
    assert!(
        pixels(&second[0].planes[3]).iter().any(|p| p[2] == 1.0),
        "flow attachment must coexist with co-visibility"
    );
    // The membership oracle must use the same behind-glass hits as all other
    // geometric planes. Treat the moving occluder as transmissive glass.
    let glass = app
        .world()
        .get::<MeshMaterial3d<StandardMaterial>>(planes[2].entity)
        .unwrap()
        .0
        .clone();
    app.world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&glass)
        .unwrap()
        .specular_transmission = 0.8;
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .annotation_glass = bevy_zeroverse::render::glass::AnnotationGlass::Through;
    let through = capture(&mut app, &cameras, 3);
    let behind: Vec<_> = planes
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 2)
        .map(|(_, p)| p.clone())
        .collect();
    verify(&cameras, &through, &behind);
    assert_ne!(
        second[0].planes[4], through[0].planes[4],
        "glass must affect occlusion membership"
    );
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .annotation_glass = bevy_zeroverse::render::glass::AnnotationGlass::Surface;
    for c in cameras.drain(..) {
        app.world_mut().despawn(c.entity);
    }
    for i in (0..16).rev() {
        cameras.push(spawn_camera(
            &mut app,
            i,
            Transform::default(),
            UVec2::new(63 + (i as u32 % 2) * 4, 47),
            false,
        ));
    }
    cameras.sort_by_key(|c| c.index);
    let packed = capture(&mut app, &cameras, 4);
    for (i, (c, packet)) in cameras.iter().zip(&packed).enumerate() {
        let data = pixels(packet.planes.last().unwrap());
        let center = data[(c.size.y / 2 * c.size.x + c.size.x / 2) as usize];
        assert_eq!(center[0] as u16, u16::MAX ^ (1 << i));
        assert_eq!(center[1], 15.0);
    }
    for c in cameras {
        app.world_mut().despawn(c.entity);
    }
    let single = vec![spawn_camera(&mut app, 4, Transform::default(), size, false)];
    let packet = capture(&mut app, &single, 5);
    let plane = packet[0].planes.last().unwrap();
    validate_plane(plane, (size.x * size.y) as usize, 1, 0).unwrap();
    assert!(pixels(plane).iter().all(|p| p[0] == 0.0 && p[1] == 0.0));
    assert!(pixels(plane).iter().any(|p| p[2] == 1.0));
    app.world_mut().despawn(single[0].entity);
    for _ in 0..5 {
        app.update();
    }
    assert_eq!(
        app.world()
            .resource::<CoVisibilityDiagnostics>()
            .snapshot()
            .atlas_bytes,
        0
    );
    println!("co-visibility: first={metrics:?}, moved={moved:?}; 16-camera heterogeneous-resolution packing, idle reuse and teardown passed");
}
