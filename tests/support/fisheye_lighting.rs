//! Render the same world rays on both sides of every cube-face boundary.
//! Separate direct-light cases retain shadows; combined shading includes
//! reflection, transmission, ambient occlusion, bloom and non-default exposure.
use bevy::{
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
    core_pipeline::tonemapping::DebandDither,
    light::{AmbientLight, CascadeShadowConfigBuilder, GlobalAmbientLight, ShadowFilteringMethod},
    pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceTransmission},
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        render_resource::{Extent3d, TextureFormat, TextureUsages},
        renderer::RenderDevice,
        view::NoIndirectDrawing,
    },
};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    calibration::{Fisheye, FisheyeFit, FisheyeSensor},
    camera::{lens::LensCamera, ZeroverseCamera},
    headless::{create_app, setup_globals},
    io::image_copy::{CapturedImages, ImageCopier},
    render::{ground_truth::GroundTruthCamera, RenderMode},
    sample::SamplerState,
};
#[path = "fisheye_seam_probe.rs"]
mod seam_probe;
pub use seam_probe::validate_shader;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

fn pixels(p: &CapturedImages) -> Vec<[f32; 4]> {
    p.planes[0]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| bytemuck::pod_read_unaligned(b))
        .collect()
}
fn save(path: &std::path::Path, size: UVec2, p: &[[f32; 4]]) {
    let mut image = image::RgbImage::new(size.x, size.y);
    for (dst, src) in image.pixels_mut().zip(p) {
        dst.0 = std::array::from_fn(|c| {
            let x = src[c].clamp(0., 1.);
            let s = if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1. / 2.4) - 0.055
            };
            (s * 255.).round() as u8
        });
    }
    image.save(path).unwrap();
}
fn sample(image: &[[f32; 4]], side: u32, uv: Vec2) -> Vec3 {
    let p = uv * side as f32 - Vec2::splat(0.5);
    let lo = p.floor().as_ivec2();
    let f = p.fract();
    let at = |x: i32, y: i32| {
        Vec3::from_array(
            image[(y.clamp(0, side as i32 - 1) as u32 * side + x.clamp(0, side as i32 - 1) as u32)
                as usize][..3]
                .try_into()
                .unwrap(),
        )
    };
    at(lo.x, lo.y)
        .lerp(at(lo.x + 1, lo.y), f.x)
        .lerp(at(lo.x, lo.y + 1).lerp(at(lo.x + 1, lo.y + 1), f.x), f.y)
}
fn capture(app: &mut App, copiers: &[ImageCopier], id: u64) -> Vec<Vec<[f32; 4]>> {
    // Warm every newly specialized material/postprocess pipeline before requesting.
    for _ in 0..12 {
        app.update();
    }
    for c in copiers {
        c.request(id);
    }
    let start = Instant::now();
    let mut packets: Vec<_> = copiers.iter().map(|_| None).collect();
    while packets.iter().any(Option::is_none) {
        app.update();
        for (i, c) in copiers.iter().enumerate() {
            assert!(c.failure().is_none(), "{:?}", c.failure());
            if packets[i].is_none() {
                packets[i] = c.take(id);
            }
        }
        assert!(
            start.elapsed() < Duration::from_secs(40),
            "lighting capture timed out"
        );
    }
    packets
        .iter()
        .map(|p| pixels(p.as_ref().unwrap()))
        .collect()
}
fn mesh(app: &mut App, size: Vec3, tf: Transform, material: StandardMaterial) {
    let h = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::from_size(size));
    let m = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(material);
    app.world_mut().spawn((Mesh3d(h), MeshMaterial3d(m), tf));
}
pub fn run() {
    let output = std::env::args()
        .find_map(|a| a.strip_prefix("--output=").map(PathBuf::from))
        .unwrap_or_else(|| "out/fisheye/lighting".into());
    std::fs::create_dir_all(&output).unwrap();
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
    app.world_mut().resource_mut::<SamplerState>().enabled = false;
    app.insert_resource(GlobalAmbientLight::NONE);
    let matte = StandardMaterial {
        base_color: Color::srgb(0.58, 0.61, 0.65),
        perceptual_roughness: 0.8,
        ..default()
    };
    for (size, xyz) in [
        (Vec3::new(18., 0.2, 18.), Vec3::new(0., -2., 0.)),
        (Vec3::new(18., 8., 0.2), Vec3::new(0., 1., -7.)),
        (Vec3::new(0.2, 8., 18.), Vec3::new(-7., 1., 0.)),
        (Vec3::new(0.2, 8., 18.), Vec3::new(7., 1., 0.)),
    ] {
        mesh(
            &mut app,
            size,
            Transform::from_translation(xyz),
            matte.clone(),
        );
    }
    for (i, xyz) in [
        Vec3::new(2., -1., -3.),
        Vec3::new(-3., -1., -3.),
        Vec3::new(4., -0.2, -1.5),
        Vec3::new(-1., 2.5, -4.),
    ]
    .into_iter()
    .enumerate()
    {
        mesh(
            &mut app,
            Vec3::new(0.8, 1.8, 0.8),
            Transform::from_translation(xyz).with_rotation(Quat::from_rotation_y(0.31 * i as f32)),
            StandardMaterial {
                base_color: Color::srgb(0.65, 0.27, 0.08),
                ..matte.clone()
            },
        );
    }
    // A bright emitter straddles the front/right boundary, not a face center.
    mesh(
        &mut app,
        Vec3::splat(0.28),
        Transform::from_xyz(2.2, 1.6, -2.2),
        StandardMaterial {
            emissive: LinearRgba::rgb(90., 70., 45.),
            ..default()
        },
    );
    mesh(
        &mut app,
        Vec3::new(1.6, 1.2, 0.08),
        Transform::from_xyz(3., 0., -3.).with_rotation(Quat::from_rotation_y(-0.7)),
        StandardMaterial {
            base_color: Color::srgb(0.94, 0.98, 1.),
            specular_transmission: 0.85,
            thickness: 0.04,
            ior: 1.5,
            perceptual_roughness: 0.12,
            ..default()
        },
    );
    let sphere = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Sphere::new(0.7).mesh().uv(48, 24));
    let chrome = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::srgb(0.7, 0.7, 0.7),
            metallic: 1.,
            perceptual_roughness: 0.24,
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(sphere),
        MeshMaterial3d(chrome),
        Transform::from_xyz(-2., 0., -2.),
    ));
    let sun = app
        .world_mut()
        .spawn((
            DirectionalLight {
                illuminance: 0.,
                shadow_maps_enabled: true,
                ..default()
            },
            CascadeShadowConfigBuilder {
                first_cascade_far_bound: 8.,
                maximum_distance: 45.,
                ..default()
            }
            .build(),
            Transform::from_xyz(-3., 5., 2.).looking_at(Vec3::ZERO, Vec3::Y),
        ))
        .id();
    let point = app
        .world_mut()
        .spawn((
            PointLight {
                intensity: 0.,
                shadow_maps_enabled: true,
                range: 30.,
                radius: 0.05,
                ..default()
            },
            Transform::from_xyz(0., 2., -1.),
        ))
        .id();
    let spot = app
        .world_mut()
        .spawn((
            SpotLight {
                intensity: 0.,
                shadow_maps_enabled: true,
                range: 30.,
                inner_angle: 0.65,
                outer_angle: 1.2,
                ..default()
            },
            Transform::from_xyz(3., 3., 1.).looking_at(Vec3::new(0., -2., -3.), Vec3::Y),
        ))
        .id();
    let size = UVec2::splat(384);
    let (rgb, gt) = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut im = Image::new_target_texture(size.x, size.y, TextureFormat::Rgba32Float, None);
        im.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        (images.add(im), GroundTruthCamera::new(&mut images, size))
    };
    let sensor = FisheyeSensor {
        lens: Fisheye {
            coefficients: [-0.03, 0.002, 0., 0.],
            max_angle: 105_f32.to_radians(),
        },
        fit: FisheyeFit::Circular,
        anisotropy: 1.,
    };
    let copier = ImageCopier::for_targets(
        vec![rgb.clone()],
        Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        TextureFormat::Rgba32Float,
        app.world().resource::<RenderDevice>(),
    );
    // A continuous constant-radiance cube isolates view-setting inheritance
    // from any seams in an input environment asset.
    let environment = {
        use bevy::{
            asset::RenderAssetUsages,
            render::render_resource::{
                TextureDimension, TextureViewDescriptor, TextureViewDimension,
            },
        };
        let texel: Vec<u8> = [0.65_f32, 0.78, 1., 1.]
            .into_iter()
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let mut image = Image::new(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            texel.repeat(4 * 4 * 6),
            TextureFormat::Rgba16Float,
            RenderAssetUsages::default(),
        );
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
        let handle = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        EnvironmentMapLight {
            diffuse_map: handle.clone(),
            specular_map: handle,
            intensity: 80.,
            ..default()
        }
    };
    let owner = app
        .world_mut()
        .spawn((
            (
                Camera3d::default(),
                Camera::default(),
                Hdr,
                Exposure { ev100: 9. },
                AmbientLight {
                    brightness: 600.,
                    ..default()
                },
                ShadowFilteringMethod::Gaussian,
                NoIndirectDrawing,
            ),
            (
                RenderMode::Color.tonemapping(),
                DebandDither::Disabled,
                RenderMode::Color.msaa(),
                ScreenSpaceTransmission::default(),
            ),
            ZeroverseCamera {
                fisheye: Some(sensor),
                override_transform: Some(Transform::IDENTITY),
                resolution: Some(size),
                ..default()
            },
            RenderTarget::Image(ImageRenderTarget::from(rgb)),
            Projection::Perspective(PerspectiveProjection {
                near: 0.1,
                far: 50.,
                ..default()
            }),
            Transform::IDENTITY,
            gt,
            copier.clone(),
        ))
        .id();
    app.world_mut()
        .entity_mut(owner)
        .insert((environment, bevy::camera::ShadowLodOrigin));
    app.update();
    app.update();
    let lens = app.world().get::<LensCamera>(owner).unwrap().clone();
    let face_size = app
        .world()
        .resource::<Assets<Image>>()
        .get(&lens.faces[0])
        .unwrap()
        .size();
    let mut copiers = vec![copier];
    let mut face_entities = vec![(owner, Transform::IDENTITY)];
    let mut q = app
        .world_mut()
        .query::<(Entity, &bevy_zeroverse::camera::lens::LensFace, &Transform)>();
    let mut children: Vec<_> = q
        .iter(app.world())
        .filter(|(_, face, _)| face.owner == owner)
        .map(|(e, _, t)| (e, *t))
        .collect();
    // Associate each face by its target rather than entity allocation order.
    for handle in lens.faces.iter().take(5).skip(1) {
        let at=children.iter().position(|(e,_)|matches!(app.world().get::<RenderTarget>(*e),Some(RenderTarget::Image(t)) if t.handle==*handle)).unwrap();
        face_entities.push(children.remove(at));
    }
    for (i, (entity, _)) in face_entities.iter().enumerate() {
        let c = ImageCopier::for_targets(
            vec![lens.faces[i].clone()],
            Extent3d {
                width: face_size.x,
                height: face_size.y,
                depth_or_array_layers: 1,
            },
            TextureFormat::Rgba32Float,
            app.world().resource::<RenderDevice>(),
        );
        // Owner keeps its output copier, so use an inert copy-only entity.
        app.world_mut().spawn(c.clone());
        copiers.push(c);
        assert!(app.world().get::<NoIndirectDrawing>(*entity).is_some());
    }
    let scale = if let Projection::Perspective(p) = app.world().get::<Projection>(owner).unwrap() {
        1. / (p.fov * 0.5).tan()
    } else {
        unreachable!()
    };
    let mut reports = Vec::new();
    let mut passed = true;
    for (id, name) in ["ambient", "directional", "point", "spot", "combined"]
        .into_iter()
        .enumerate()
    {
        app.world_mut()
            .get_mut::<DirectionalLight>(sun)
            .unwrap()
            .illuminance = if name == "directional" || name == "combined" {
            12000.
        } else {
            0.
        };
        app.world_mut()
            .get_mut::<PointLight>(point)
            .unwrap()
            .intensity = if name == "point" || name == "combined" {
            150000.
        } else {
            0.
        };
        app.world_mut()
            .get_mut::<SpotLight>(spot)
            .unwrap()
            .intensity = if name == "spot" || name == "combined" {
            150000.
        } else {
            0.
        };
        app.world_mut()
            .get_mut::<AmbientLight>(owner)
            .unwrap()
            .brightness = if name == "ambient" { 600. } else { 35. };
        if name == "combined" {
            app.world_mut().entity_mut(owner).insert((
                Bloom {
                    intensity: 0.035,
                    ..Bloom::NATURAL
                },
                ScreenSpaceAmbientOcclusion::default(),
            ));
        }
        let images = capture(&mut app, &copiers, id as u64 + 1);
        save(&output.join(format!("{name}.png")), size, &images[0]);
        for (i, image) in images.iter().skip(1).enumerate() {
            save(
                &output.join(format!("{name}_face_{i}.png")),
                face_size,
                image,
            );
        }
        let continuity = seam_probe::run(
            app.world().resource(),
            app.world().resource(),
            &lens,
            &face_entities,
            scale,
            face_size,
            &images[1..],
        );
        let mut errors = Vec::new();
        let mut gradients = Vec::new();
        let mut boundaries = Vec::new();
        for (a, b) in [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
        ] {
            let mut pair_errors = Vec::new();
            let fa = face_entities[a].1.rotation * Vec3::NEG_Z;
            let fb = face_entities[b].1.rotation * Vec3::NEG_Z;
            let edge = fa.cross(fb).normalize();
            for j in -90..=90 {
                let ray = (fa + fb + edge * (j as f32 / 100.)).normalize();
                if (-ray.z).acos() > sensor.lens.max_angle {
                    continue;
                }
                let uv = |i: usize, d: Vec3| {
                    let p = face_entities[i].1.rotation.inverse() * d;
                    Vec2::new(p.x, -p.y) / -p.z * scale * 0.5 + Vec2::splat(0.5)
                };
                let ca = sample(&images[a + 1], face_size.x, uv(a, ray));
                let cb = sample(&images[b + 1], face_size.x, uv(b, ray));
                errors.push((ca - cb).abs().max_element());
                pair_errors.push((ca - cb).abs().max_element());
                let adjacent = sample(
                    &images[a + 1],
                    face_size.x,
                    uv(a, (ray + edge * 0.002).normalize()),
                );
                gradients.push((ca - adjacent).abs().max_element());
            }
            pair_errors.sort_by(f32::total_cmp);
            let mean = pair_errors.iter().sum::<f32>() / pair_errors.len() as f32;
            let p95 = pair_errors[pair_errors.len() * 95 / 100];
            passed &= mean < 0.025 && p95 < 0.08;
            boundaries.push(serde_json::json!({"faces":[a,b],"rays":pair_errors.len(),"mean":mean,"p95":p95,"max":pair_errors.last()}));
        }
        errors.sort_by(f32::total_cmp);
        gradients.sort_by(f32::total_cmp);
        let mean = errors.iter().sum::<f32>() / errors.len() as f32;
        let p95 = errors[(errors.len() as f32 * 0.95) as usize];
        let max = *errors.last().unwrap();
        println!("{name}: shared-ray seam mean={mean:.6} p95={p95:.6} max={max:.6}");
        reports.push(serde_json::json!({"case":name,"shared_boundary_rays":errors.len(),"linear_rgb_abs_error_mean":mean,"linear_rgb_abs_error_p95":p95,"linear_rgb_abs_error_max":max,"adjacent_gradient_p95":gradients[(gradients.len() as f32*0.95) as usize],"boundaries":boundaries,"composite_continuity":continuity}));
        // Do not hide silhouette/interpolation outliers behind a maximum-only
        // criterion. A widespread shading step must fail even with shadows on.
        passed &= mean < 0.025 && p95 < 0.08;
    }
    std::fs::write(output.join("report.json"),serde_json::to_vec_pretty(&serde_json::json!({"provenance":bevy_zeroverse::provenance::capture_provenance(),"passed":passed,"size":size.to_array(),"face_size":face_size.to_array(),"boundary_policy":"compare identical world rays across face pairs, linear RGB; fixed exposure, shadow-enabled directional/point/spot cases, combined SSAO/bloom/transmission", "cases":reports})).unwrap()).unwrap();
    assert!(
        passed,
        "one or more lighting boundaries failed; inspect report.json and captures"
    );
}
