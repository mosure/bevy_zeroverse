//! Independent analytic box intersections and f64 lens projection validate the
//! nonlinear GPU ray pass, mixed-lens visibility and capture-time camera flow.
use bevy::{
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureFormat, TextureUsages},
        renderer::RenderDevice,
    },
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    calibration::{CameraCalibration, Fisheye, FisheyeFit},
    camera::{CaptureCameraIndex, ZeroverseCamera},
    headless::{create_app, setup_globals},
    io::image_copy::{CapturedImages, ImageCopier},
    render::{ground_truth::GroundTruthCamera, semantic::SemanticLabel},
};
use std::time::{Duration, Instant};
fn pixels(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|p| bytemuck::pod_read_unaligned(p))
        .collect()
}
fn point(
    camera: Transform,
    cal: &CameraCalibration,
    x: usize,
    y: usize,
    shift: f32,
) -> Option<Vec3> {
    let ray = camera.rotation * Vec3::from_array(cal.ray([x as f32 + 0.5, y as f32 + 0.5])?);
    let origin = camera.translation;
    let mut distance = f32::INFINITY;
    for i in 0..3 {
        if ray[i].abs() > 1e-7 {
            let t = (ray[i].signum() * 5. - origin[i]) / ray[i];
            if t > 0. {
                distance = distance.min(t);
            }
        }
    }
    // Independent analytic occluder: axis-aligned 0.9m cube at (0.4,0,-2).
    let lo = Vec3::new(-0.05 + shift, -0.45, -2.45);
    let hi = lo + Vec3::splat(0.9);
    let mut near = 0_f32;
    let mut far = f32::INFINITY;
    for i in 0..3 {
        if ray[i].abs() < 1e-7 {
            if origin[i] < lo[i] || origin[i] > hi[i] {
                far = -1.;
            }
        } else {
            let a = (lo[i] - origin[i]) / ray[i];
            let b = (hi[i] - origin[i]) / ray[i];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
        }
    }
    if far >= near && near > 0. {
        distance = distance.min(near);
    }
    Some(origin + ray * distance)
}
pub fn run() {
    let narrow = std::env::args().any(|a| a == "--narrow");
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
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .headless = false;
    app.world_mut()
        .resource_mut::<bevy_zeroverse::sample::SamplerState>()
        .enabled = false;
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::linear_rgb(0., 0., 0.25),
            unlit: true,
            cull_mode: None,
            ..default()
        });
    let mut moving_box = None;
    for (size, tf, label) in [
        (Vec3::splat(10.), Transform::IDENTITY, SemanticLabel::Wall),
        (
            Vec3::splat(0.9),
            Transform::from_xyz(0.4, 0., -2.),
            SemanticLabel::Table,
        ),
    ] {
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::from_size(size));
        let is_box = label == SemanticLabel::Table;
        let material = if is_box {
            app.world_mut()
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial {
                    base_color: Color::linear_rgb(0.25, 0., 0.),
                    unlit: true,
                    cull_mode: None,
                    ..default()
                })
        } else {
            material.clone()
        };
        let entity = app
            .world_mut()
            .spawn((Mesh3d(mesh), MeshMaterial3d(material.clone()), tf, label))
            .id();
        if is_box {
            moving_box = Some(entity);
        }
    }
    let size = [97, 73];
    let dimensions = Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: 1,
    };
    let angular = Fisheye {
        coefficients: [-0.03, 0.002, 0.00001, 0.],
        max_angle: if narrow { 45_f32 } else { 105_f32 }.to_radians(),
    };
    let calibrations = [
        angular
            .calibration(size, FisheyeFit::Elliptical, [0.03, -0.02], 1.03)
            .unwrap(),
        angular
            .calibration(size, FisheyeFit::Diagonal, [-0.05, 0.04], 0.98)
            .unwrap(),
        CameraCalibration::centered_pinhole(size[0], size[1], 1.2, size[0] as f32 / size[1] as f32)
            .unwrap(),
    ];
    let mut poses = [
        Transform::IDENTITY,
        Transform::from_xyz(0.7, 0.2, 0.3).with_rotation(Quat::from_rotation_y(0.25)),
        Transform::from_xyz(-0.4, -0.2, 0.1).with_rotation(Quat::from_rotation_y(-0.5)),
    ];
    let mut cameras = Vec::new();
    let mut copiers = Vec::new();
    for i in 0..3 {
        let (rgb, gt) = {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            let mut rgb =
                Image::new_target_texture(size[0], size[1], TextureFormat::Rgba32Float, None);
            rgb.texture_descriptor.usage |= TextureUsages::COPY_SRC;
            let rgb = images.add(rgb);
            let mut gt = GroundTruthCamera::new(&mut images, UVec2::from_array(size));
            gt.calibration = Some(calibrations[i].clone());
            gt.enable_flow(&mut images);
            gt.enable_co_visibility(&mut images, UVec2::from_array(size));
            (rgb, gt)
        };
        let copier = ImageCopier::for_targets(
            vec![
                rgb.clone(),
                gt.world_depth.clone(),
                gt.normal_semantic.clone(),
                gt.flow.clone().unwrap(),
                gt.co_visibility.as_ref().unwrap().image.clone(),
            ],
            dimensions,
            TextureFormat::Rgba32Float,
            app.world().resource::<RenderDevice>(),
        );
        let camera = app
            .world_mut()
            .spawn((
                (
                    Camera3d::default(),
                    Hdr,
                    Exposure::INDOOR,
                    bevy_zeroverse::render::RenderMode::Color.tonemapping(),
                    bevy_zeroverse::render::RenderMode::Color.msaa(),
                    ZeroverseCamera {
                        fisheye: (i < 2).then_some(bevy_zeroverse::calibration::FisheyeSensor {
                            lens: angular,
                            fit: if i == 0 {
                                FisheyeFit::Elliptical
                            } else {
                                FisheyeFit::Diagonal
                            },
                            anisotropy: if i == 0 { 1.03 } else { 0.98 },
                        }),
                        principal_point_offset: match i {
                            0 => Vec2::new(0.03, -0.02),
                            1 => Vec2::new(-0.05, 0.04),
                            _ => Vec2::ZERO,
                        },
                        override_transform: Some(poses[i]),
                        resolution: Some(UVec2::from_array(size)),
                        ..default()
                    },
                ),
                Camera::default(),
                CaptureCameraIndex(i),
                RenderTarget::Image(ImageRenderTarget::from(rgb)),
                Projection::Perspective(PerspectiveProjection {
                    fov: 1.2,
                    near: 0.1,
                    far: 50.,
                    ..default()
                }),
                poses[i],
                gt,
                copier.clone(),
            ))
            .id();
        cameras.push(camera);
        copiers.push(copier);
    }
    let mut previous: Option<Vec<CapturedImages>> = None;
    let mut max_position = 0_f32;
    let mut max_flow = 0_f32;
    let mut compared = 0;
    let mut cov_compared = 0;
    let mut cov_wrong = 0;
    let mut rgb_compared = 0usize;
    let mut rgb_wrong = 0usize;
    let mut rgb_foreground = 0usize;
    for stamp in 1..=3 {
        for _ in 0..5 {
            app.update();
        }
        for c in &copiers {
            c.request(stamp);
        }
        let start = Instant::now();
        let mut packets: [Option<CapturedImages>; 3] = [None, None, None];
        while packets.iter().any(Option::is_none) {
            app.update();
            for (i, c) in copiers.iter().enumerate() {
                assert!(c.failure().is_none(), "{:?}", c.failure());
                if packets[i].is_none() {
                    packets[i] = c.take(stamp);
                }
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "analytic fisheye capture stalled"
            );
        }
        let packets: Vec<_> = packets.into_iter().map(Option::unwrap).collect();
        let old_worlds: Option<Vec<_>> = previous
            .as_ref()
            .map(|p| p.iter().map(|p| pixels(&p.planes[1])).collect());
        let old_normals: Option<Vec<_>> = previous
            .as_ref()
            .map(|p| p.iter().map(|p| pixels(&p.planes[2])).collect());
        let worlds: Vec<_> = packets.iter().map(|p| pixels(&p.planes[1])).collect();
        let normals: Vec<_> = packets.iter().map(|p| pixels(&p.planes[2])).collect();
        for i in 0..3 {
            let c = &calibrations[i];
            let cov = pixels(&packets[i].planes[4]);
            let flow = pixels(&packets[i].planes[3]);
            let rgb = pixels(&packets[i].planes[0]);
            for y in 0..size[1] as usize {
                for x in 0..size[0] as usize {
                    let id = y * size[0] as usize + x;
                    let p = worlds[i][id];
                    let Some(expected) = point(poses[i], c, x, y, (stamp - 1) as f32 * 0.08) else {
                        assert_eq!(p[3], 0.);
                        assert_eq!(cov[id][2], 0.);
                        assert_eq!(rgb[id], [0.; 4], "invalid lens pixel must be clear");
                        continue;
                    };
                    let actual = Vec3::from_array(p[..3].try_into().unwrap());
                    max_position = max_position.max(actual.distance(expected));
                    if x >= 2 && y >= 2 && x + 2 < size[0] as usize && y + 2 < size[1] as usize {
                        let label = normals[i][id][3];
                        let core = (-2_i32..=2).all(|dy| {
                            (-2_i32..=2).all(|dx| {
                                normals[i][(y as i32 + dy) as usize * size[0] as usize
                                    + (x as i32 + dx) as usize][3]
                                    == label
                            })
                        });
                        if core {
                            let is_box = label
                                == bevy_zeroverse::render::ground_truth::semantic_id(
                                    &SemanticLabel::Table,
                                ) as f32;
                            rgb_compared += 1;
                            rgb_foreground += usize::from(is_box);
                            let expected_channel = if is_box { 0 } else { 2 };
                            rgb_wrong += usize::from(
                                (rgb[id][0] > rgb[id][2]) != is_box
                                    || rgb[id][expected_channel] < 0.005
                                    || rgb[id][3] < 0.99,
                            );
                        }
                    }
                    assert!(p[3] > 0.);
                    let local = poses[i].to_matrix().inverse().transform_point3(actual);
                    assert!((c.depth(local.to_array()) - p[3]).abs() < 0.001);
                    for j in 0..3 {
                        if i == j {
                            continue;
                        }
                        let local = poses[j].to_matrix().inverse().transform_point3(actual);
                        let Some(q) = calibrations[j]
                            .project(local.to_array())
                            .filter(|q| calibrations[j].in_image(*q))
                        else {
                            assert_eq!((cov[id][0] as u32) & (1 << j), 0);
                            continue;
                        };
                        let tx = q[0] as usize;
                        let ty = q[1] as usize;
                        if tx < 1
                            || ty < 1
                            || tx + 1 >= size[0] as usize
                            || ty + 1 >= size[1] as usize
                        {
                            continue;
                        }
                        let center = normals[j][ty * size[0] as usize + tx];
                        let uniform = (-1_i32..=1).all(|dy| {
                            (-1_i32..=1).all(|dx| {
                                normals[j][((ty as i32 + dy) as usize) * size[0] as usize
                                    + (tx as i32 + dx) as usize]
                                    == center
                            })
                        });
                        if !uniform {
                            continue;
                        }
                        let hit = Vec3::from_array(
                            worlds[j][ty * size[0] as usize + tx][..3]
                                .try_into()
                                .unwrap(),
                        );
                        let n = Vec3::from_array(center[..3].try_into().unwrap()) * 2. - Vec3::ONE;
                        let n = poses[j].rotation * n;
                        let expected_visible = (hit - actual).dot(n).abs() < 0.0005;
                        let visible = (cov[id][0] as u32) & (1 << j) != 0;
                        cov_compared += 1;
                        cov_wrong += usize::from(visible != expected_visible);
                    }
                    if let Some(old) = &old_worlds {
                        let p = old[i][id];
                        if p[3] <= 0. {
                            continue;
                        }
                        let displacement = if old_normals.as_ref().unwrap()[i][id][3]
                            == bevy_zeroverse::render::ground_truth::semantic_id(
                                &SemanticLabel::Table,
                            ) as f32
                        {
                            Vec3::X * 0.08
                        } else {
                            Vec3::ZERO
                        };
                        let local = poses[i].to_matrix().inverse().transform_point3(
                            Vec3::from_array(p[..3].try_into().unwrap()) + displacement,
                        );
                        if let Some(target) = c.project(local.to_array()) {
                            if c.depth(local.to_array()) > 0.1 {
                                assert_eq!(flow[id][2], 1.);
                                let expected = Vec2::from_array(target)
                                    - Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                                max_flow = max_flow
                                    .max(expected.distance(Vec2::new(flow[id][0], flow[id][1])));
                                compared += 1;
                            }
                        }
                    } else {
                        assert_eq!(flow[id], [0.; 4]);
                    }
                }
            }
        }
        previous = Some(packets);
        app.world_mut()
            .get_mut::<Transform>(moving_box.unwrap())
            .unwrap()
            .translation
            .x += 0.08;
        for (i, e) in cameras.iter().enumerate() {
            poses[i].translation.x += 0.02;
            poses[i].rotation *= Quat::from_rotation_y(0.015);
            *app.world_mut().get_mut::<Transform>(*e).unwrap() = poses[i];
        }
    }
    assert!(max_position < 0.001, "position {max_position}");
    assert!(max_flow < 0.01, "flow {max_flow}");
    assert!(compared > 5000);
    assert!(rgb_compared > 5000);
    assert!(
        rgb_foreground > 10,
        "RGB oracle must include the moving foreground"
    );
    assert_eq!(rgb_wrong, 0, "RGB/geometry alignment");
    assert!(cov_compared > 5000);
    assert_eq!(cov_wrong, 0, "co-visibility mismatch away from silhouettes");
    std::fs::create_dir_all("out/fisheye").unwrap();
    std::fs::write(if narrow { "out/fisheye/analytic-narrow.json" } else { "out/fisheye/analytic.json" },serde_json::to_vec_pretty(&serde_json::json!({"provenance":bevy_zeroverse::provenance::capture_provenance(),"position_max_metres":max_position,"flow_max_pixels":max_flow,"flow_correspondences":compared,"co_visibility_relations":cov_compared,"co_visibility_mismatches":cov_wrong,"mixed_lenses":true,"single_face_case":narrow,"rgb_core_pixels":rgb_compared,"rgb_core_mismatches":rgb_wrong,"rgb_foreground_pixels":rgb_foreground,"moving_occluder_metres_per_step":0.08,"size":size})).unwrap()).unwrap();
    println!("Analytic fisheye: max position {max_position:.8}m, max flow {max_flow:.6}px over {compared} correspondences; {cov_compared} co-visibility relations, {cov_wrong} mismatches");
}
