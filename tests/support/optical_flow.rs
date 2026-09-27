//! Analytic capture-time flow reference, including mesh deformation and visibility.
#![cfg(not(target_arch = "wasm32"))]
use bevy::{
    asset::RenderAssetUsages,
    camera::{ImageRenderTarget, RenderTarget},
    mesh::{
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
        Indices, PrimitiveTopology,
    },
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
        ground_truth::{semantic_id, GroundTruthCamera},
        semantic::SemanticLabel,
    },
};
use std::time::{Duration, Instant};

fn quad(width: f32, height: f32) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![
            [-width, -height, 0.0],
            [width, -height, 0.0],
            [width, height, 0.0],
            [-width, height, 0.0],
        ],
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 4]);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_UV_0,
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    );
    mesh.insert_indices(Indices::U32(vec![0, 1, 2, 0, 2, 3]));
    mesh
}

fn spawn(
    app: &mut App,
    mesh: Mesh,
    transform: Transform,
    label: SemanticLabel,
) -> (Entity, Handle<Mesh>) {
    let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            unlit: true,
            cull_mode: None,
            ..default()
        });
    let e = app
        .world_mut()
        .spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material),
            transform,
            label,
        ))
        .id();
    (e, mesh)
}

fn capture(app: &mut App, copier: &ImageCopier, id: u64) -> CapturedImages {
    // Production headless capture gates cameras off during asynchronous
    // readback. Temporal history must survive the missing ExtractedView.
    let set_active = |app: &mut App, active| {
        let mut cameras = app
            .world_mut()
            .query_filtered::<&mut Camera, With<GroundTruthCamera>>();
        for mut camera in cameras.iter_mut(app.world_mut()) {
            camera.is_active = active;
        }
    };
    set_active(app, false);
    for _ in 0..3 {
        app.update();
    }
    set_active(app, true);
    // Unrelated renderer frames must not become flow source frames.
    for _ in 0..7 {
        app.update();
    }
    copier.request(id);
    let start = Instant::now();
    loop {
        app.update();
        assert!(copier.failure().is_none(), "{:?}", copier.failure());
        if let Some(packet) = copier.take(id) {
            return packet;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "flow capture stalled"
        );
    }
}

fn floats(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|bytes| bytemuck::pod_read_unaligned(bytes))
        .collect()
}

/// Ground-truth source position is independently projected into the destination.
/// Raster coverage is checked away from boundaries, not matched by nearest pixels.
fn verify(
    source: &CapturedImages,
    pair: &CapturedImages,
    size: UVec2,
    target_camera: Transform,
    fov: f32,
    map: impl Fn(Vec3, u32) -> Vec3,
) -> (usize, f32) {
    let positions = floats(&source.planes[1]);
    let labels = floats(&source.planes[2]);
    let flow = floats(&pair.planes[3]);
    let view = target_camera.to_matrix().inverse();
    let focal = size.y as f64 / (2.0 * (fov as f64 * 0.5).tan());
    let mut checked = 0;
    let mut maximum = 0.0_f32;
    for y in 3..size.y - 3 {
        for x in 3..size.x - 3 {
            let i = (y * size.x + x) as usize;
            if positions[i][3] == 0.0 {
                continue;
            }
            assert_eq!(flow[i][2], 1.0, "lost source correspondence at {x},{y}");
            let p = map(
                Vec3::from_array(positions[i][..3].try_into().unwrap()),
                labels[i][3] as u32,
            );
            let q = view.transform_point3(p);
            let expected = Vec2::new(
                (q.x as f64 / -q.z as f64 * focal + size.x as f64 * 0.5 - x as f64 - 0.5) as f32,
                (-q.y as f64 / -q.z as f64 * focal + size.y as f64 * 0.5 - y as f64 - 0.5) as f32,
            );
            let actual = Vec2::new(flow[i][0], flow[i][1]);
            maximum = maximum.max(actual.distance(expected));
            checked += 1;
        }
    }
    assert!(checked > 1000, "too few correspondences: {checked}");
    assert!(maximum < 0.005, "projection error {maximum}px");
    println!(
        "flow pair {}: checked {checked} pixels, max analytic endpoint error {maximum:.7}px",
        pair.request_id
    );
    (checked, maximum)
}

pub fn run_validation() {
    let module =
        wgpu::naga::front::wgsl::parse_str(include_str!("../../src/render/ground_truth/flow.wgsl"))
            .unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let assets = tempfile::tempdir().unwrap();
    setup_globals(Some(assets.path().to_string_lossy().into_owned()));
    let config = BevyZeroverseConfig {
        initialize_scene: false,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        press_esc_close: false,
        image_copiers: true,
        num_cameras: 0,
        ..default()
    };
    let mut app = create_app(None, Some(config), false);
    app.finish();
    app.cleanup();
    let size = UVec2::new(97, 73);
    let fov = 1.0_f32;
    let (foreground, mesh) = spawn(
        &mut app,
        quad(0.8, 0.7),
        Transform::from_xyz(0.0, 0.0, -3.0),
        SemanticLabel::Person,
    );
    spawn(
        &mut app,
        quad(20.0, 20.0),
        Transform::from_xyz(0.0, 0.0, -6.0),
        SemanticLabel::Wall,
    );
    let (rgb, gt, flow) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut rgb = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
        rgb.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        let rgb = images.add(rgb);
        let mut gt = GroundTruthCamera::new(&mut images, size);
        let flow = gt.enable_flow(&mut images);
        gt.flow_sequence = 1;
        (rgb, gt, flow)
    };
    let copier = ImageCopier::for_targets(
        vec![
            rgb.clone(),
            gt.world_depth.clone(),
            gt.normal_semantic.clone(),
            flow,
        ],
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        app.world().resource::<RenderDevice>(),
    );
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(ImageRenderTarget::from(rgb)),
            Transform::IDENTITY,
            Projection::Perspective(PerspectiveProjection {
                fov,
                near: 0.1,
                far: 50.0,
                ..default()
            }),
            Msaa::Off,
            gt,
            copier.clone(),
        ))
        .id();
    let first = capture(&mut app, &copier, 1);
    assert!(first.planes[3].iter().all(|b| *b == 0));
    let still = capture(&mut app, &copier, 2);
    verify(&first, &still, size, Transform::IDENTITY, fov, |p, _| p);
    let stationary = floats(&still.planes[3]);
    assert!(stationary
        .iter()
        .filter(|p| p[2] == 1.0)
        .all(|p| p[0].abs() < 0.001 && p[1].abs() < 0.001 && p[3] == 1.0));
    let delta = Vec3::new(0.2, 0.1, 0.0);
    app.world_mut()
        .entity_mut(foreground)
        .insert(Transform::from_translation(
            Vec3::new(0.0, 0.0, -3.0) + delta,
        ));
    let rigid = capture(&mut app, &copier, 3);
    verify(&still, &rigid, size, Transform::IDENTITY, fov, |p, id| {
        if id == semantic_id(&SemanticLabel::Person) {
            p + delta
        } else {
            p
        }
    });
    let center = (size.y / 2 * size.x + size.x / 2) as usize;
    let values = floats(&rigid.planes[3]);
    assert!(values[center][0] > 4.0 && values[center][1] < -2.0);
    let moved_camera =
        Transform::from_xyz(0.12, 0.05, 0.0).with_rotation(Quat::from_rotation_y(0.06));
    let fov2 = 1.13;
    app.world_mut().entity_mut(camera).insert((
        moved_camera,
        Projection::Perspective(PerspectiveProjection {
            fov: fov2,
            near: 0.1,
            far: 50.0,
            ..default()
        }),
    ));
    let camera_motion = capture(&mut app, &copier, 4);
    verify(&rigid, &camera_motion, size, moved_camera, fov2, |p, _| p);
    if let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&mesh)
        .unwrap()
        .attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions {
            p[0] += 0.3 * p[1];
        }
    }
    let deformed = capture(&mut app, &copier, 5);
    verify(
        &camera_motion,
        &deformed,
        size,
        moved_camera,
        fov2,
        |mut p, id| {
            if id == semantic_id(&SemanticLabel::Person) {
                p.x += 0.3 * (p.y - delta.y);
            }
            p
        },
    );
    // Newly occluding geometry must not destroy the valid, but hidden, vector.
    let (blocker, _) = spawn(
        &mut app,
        quad(1.1, 1.1),
        Transform::from_xyz(0.0, 0.0, -1.5),
        SemanticLabel::Window,
    );
    let occluded = capture(&mut app, &copier, 6);
    let labels = floats(&deformed.planes[2]);
    let values = floats(&occluded.planes[3]);
    let hidden = values
        .iter()
        .zip(&labels)
        .filter(|(p, l)| {
            l[3] == semantic_id(&SemanticLabel::Person) as f32 && p[2] == 1.0 && p[3] == 0.0
        })
        .count();
    assert!(hidden > 300, "only {hidden} occluded person pixels");
    app.world_mut().despawn(blocker);
    app.world_mut()
        .get_mut::<GroundTruthCamera>(camera)
        .unwrap()
        .flow_sequence = 2;
    let reset = capture(&mut app, &copier, 7);
    assert!(reset.planes[3].iter().all(|b| *b == 0));
    // Topology changes have no established correspondence, even with equal counts.
    app.world_mut()
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&mesh)
        .unwrap()
        .insert_indices(Indices::U32(vec![0, 1, 3, 1, 2, 3]));
    let topology = capture(&mut app, &copier, 8);
    for (p, l) in floats(&topology.planes[3])
        .iter()
        .zip(floats(&reset.planes[2]))
    {
        if l[3] == semantic_id(&SemanticLabel::Person) as f32 {
            assert_eq!(*p, [0.0; 4]);
        }
    }
    app.world_mut().despawn(foreground);
    let mut skin = quad(0.8, 0.7);
    skin.insert_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        bevy::mesh::VertexAttributeValues::Uint16x4(vec![[0, 1, 0, 0]; 4]),
    );
    skin.insert_attribute(
        Mesh::ATTRIBUTE_JOINT_WEIGHT,
        vec![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
    );
    let joints = [
        app.world_mut()
            .spawn(Transform::from_xyz(0.0, 0.0, -3.0))
            .id(),
        app.world_mut()
            .spawn(Transform::from_xyz(0.0, 0.0, -3.0))
            .id(),
    ];
    let bind = app
        .world_mut()
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .add(vec![Mat4::IDENTITY; 2]);
    let (person, _) = spawn(&mut app, skin, Transform::IDENTITY, SemanticLabel::Person);
    app.world_mut().entity_mut(person).insert(SkinnedMesh {
        inverse_bindposes: bind,
        joints: joints.to_vec(),
    });
    let skin_source = capture(&mut app, &copier, 9);
    app.world_mut()
        .entity_mut(joints[1])
        .insert(Transform::from_xyz(0.25, 0.0, -3.0));
    let skinned = capture(&mut app, &copier, 10);
    verify(
        &skin_source,
        &skinned,
        size,
        moved_camera,
        fov2,
        |mut p, id| {
            if id == semantic_id(&SemanticLabel::Person) {
                p.x += 0.25 * (p.y + 0.7) / 1.4;
            }
            p
        },
    );
    app.world_mut().despawn(person);
    let removed = capture(&mut app, &copier, 11);
    for (p, l) in floats(&removed.planes[3])
        .iter()
        .zip(floats(&skinned.planes[2]))
    {
        if l[3] == semantic_id(&SemanticLabel::Person) as f32 {
            assert_eq!(*p, [0.0; 4]);
        }
    }
    let (boundary_subject, _) = spawn(
        &mut app,
        quad(0.8, 0.7),
        Transform::from_xyz(0.0, 0.0, -3.0),
        SemanticLabel::Person,
    );
    let boundary_source = capture(&mut app, &copier, 12);
    app.world_mut()
        .entity_mut(boundary_subject)
        .insert(Transform::from_xyz(10.0, 0.0, -3.0));
    let outside = capture(&mut app, &copier, 13);
    let mut outside_pixels = 0;
    for (p, l) in floats(&outside.planes[3])
        .iter()
        .zip(floats(&boundary_source.planes[2]))
    {
        if l[3] == semantic_id(&SemanticLabel::Person) as f32 {
            assert_eq!(p[2..], [1.0, 0.0]);
            assert!(p[0] > size.x as f32, "offscreen flow was clamped");
            outside_pixels += 1;
        }
    }
    assert!(outside_pixels > 300);
    app.world_mut()
        .entity_mut(boundary_subject)
        .insert(Transform::from_xyz(0.0, 0.0, -3.0));
    let behind_source = capture(&mut app, &copier, 14);
    app.world_mut()
        .entity_mut(boundary_subject)
        .insert(Transform::from_xyz(0.0, 0.0, 3.0));
    let behind = capture(&mut app, &copier, 15);
    for (p, l) in floats(&behind.planes[3])
        .iter()
        .zip(floats(&behind_source.planes[2]))
    {
        if l[3] == semantic_id(&SemanticLabel::Person) as f32 {
            assert_eq!(*p, [0.0; 4]);
        }
    }
    println!("occlusion mask: {hidden} hidden person pixels; {outside_pixels} valid offscreen pixels; behind-camera targets, topology changes, despawn, camera pauses and sequence reset passed");
}
