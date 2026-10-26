//! Real StandardMaterial instances from IndoorMaterials, under one fixed rig.
use super::{audit, Category, RESOLUTION, SEEDS};
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{semantic::SemanticLabel, RenderMode},
    sample::SamplerState,
    scene::{
        procedural_indoor::{
            appearance::AppearanceSettings,
            geometry::Geometry,
            layout::ObjectKind,
            layout::{IndoorLayout, IndoorManifest},
            materials::{rugs, IndoorMaterials, Surface},
            objects,
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

#[derive(Resource)]
struct Batch {
    handles: Vec<Handle<StandardMaterial>>,
    leaves: Vec<bool>,
    names: Vec<String>,
    environment: EnvironmentMapLight,
    install: bool,
}
#[derive(Component)]
struct Root;
fn install(
    mut commands: Commands,
    mut batch: ResMut<Batch>,
    roots: Query<Entity, With<Root>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut sampler: ResMut<SamplerState>,
) {
    if !batch.install {
        return;
    }
    for root in &roots {
        commands.entity(root).despawn();
    }
    let root = commands
        .spawn((
            Root,
            ZeroverseSceneRoot,
            SceneAabbNode,
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();
    for (i, material) in batch.handles.iter().enumerate() {
        let offset = Vec3::X * i as f32 * 6.;
        let mut g = Geometry::default();
        if batch.leaves[i] {
            // A real production blade preserves the atlas' base-to-tip UV domain.
            g.leaf(
                0.62,
                0.26,
                0.10,
                Transform::from_xyz(-0.18, 0.09, 0.12).with_rotation(Quat::from_rotation_x(0.95)),
            );
            g.leaf(
                0.54,
                0.23,
                -0.06,
                Transform::from_xyz(0.17, 0.10, 0.10)
                    .with_rotation(Quat::from_rotation_y(0.2) * Quat::from_rotation_x(1.05)),
            );
        } else if matches!(
            batch.names[i].as_str(),
            "Screen" | "PhoneScreen" | "Whiteboard" | "Television" | "Art"
        ) {
            // Content is a complete surface image, not a metric repeating swatch.
            let (w, h) = if batch.names[i] == "PhoneScreen" {
                (0.30, 0.56)
            } else {
                (0.88, 0.52)
            };
            g.mesh(
                Rectangle::new(w, h).mesh().build(),
                Transform::from_xyz(0., 0.30, 0.),
            );
        } else if batch.names[i] == "Rug" {
            let mut o = objects::audit::fixture(ObjectKind::Rug, 19);
            o.size = Vec3::new(0.95, 0.009, 0.72);
            let assembly = objects::build_object(&o);
            for ((_, label), part) in assembly.parts {
                if rugs::key(&label).is_some_and(|(_, binding)| !binding) {
                    g.mesh(part.into_mesh(), Transform::from_xyz(0., 0.025, 0.));
                }
            }
        } else {
            let mut sphere = Sphere::new(0.23).mesh().uv(64, 32);
            if let Some(bevy::mesh::VertexAttributeValues::Float32x2(uvs)) =
                sphere.attribute_mut(Mesh::ATTRIBUTE_UV_0)
            {
                for uv in uvs {
                    uv[0] *= std::f32::consts::TAU * 0.23;
                    uv[1] *= std::f32::consts::PI * 0.23;
                }
            }
            g.mesh(sphere, Transform::from_xyz(-0.27, 0.26, 0.04));
            g.cuboid(
                Vec3::new(0.46, 0.49, 0.025),
                0.012,
                Transform::from_xyz(0.29, 0.255, -0.05),
            );
        }
        commands.spawn((
            Mesh3d(meshes.add(g.into_mesh())),
            MeshMaterial3d(material.clone()),
            SemanticLabel::OtherProp,
            Transform::from_translation(offset),
            ChildOf(root),
        ));
        let target = offset + Vec3::new(0., 0.27, 0.);
        let from = offset + Vec3::new(0.75, 0.80, 1.65);
        commands.spawn((
            ZeroverseCamera {
                resolution: Some(UVec2::splat(RESOLUTION)),
                perspective_sampler: PerspectiveSampler::exact(40.),
                trajectory: TrajectorySampler::Static {
                    start: ExtrinsicsSampler {
                        looking_at: LookingAtSampler::Exact(target),
                        position: ExtrinsicsSamplerType::Transform(Transform::from_translation(
                            from,
                        )),
                        ..default()
                    },
                },
                ..default()
            },
            CaptureCameraIndex(i),
            batch.environment.clone(),
            ChildOf(root),
        ));
    }
    *sampler = SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color, RenderMode::Normal],
        warmup_frames: 18,
        frames: 3,
        timesteps: vec![],
        ..default()
    };
    batch.install = false;
}

fn maps(
    output: &Path,
    name: &str,
    index: usize,
    m: &StandardMaterial,
    images: &Assets<Image>,
) -> Result<Value> {
    let mut details = serde_json::Map::new();
    for (mode, handle, channel) in [
        ("albedo", m.base_color_texture.as_ref(), None),
        ("normal_map", m.normal_map_texture.as_ref(), None),
        (
            "roughness",
            m.metallic_roughness_texture.as_ref(),
            Some(1usize),
        ),
    ] {
        let (n, bytes, levels) = if let Some(handle) = handle {
            let image = images
                .get(handle)
                .ok_or_else(|| anyhow::anyhow!("missing material image"))?;
            let n = image.width();
            ensure!(image.height() == n, "non-square study texture");
            let data = image
                .data
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("study requires CPU reference maps"))?;
            ensure!(data.len() >= (n * n * 4) as usize, "short texture");
            let levels = image.texture_descriptor.mip_level_count;
            let expected: usize = (0..levels)
                .map(|level| ((n >> level).max(1).pow(2) * 4) as usize)
                .sum();
            ensure!(data.len() == expected, "incomplete material mip chain");
            let bytes: Vec<u8> = data[..(n * n * 4) as usize]
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    if let Some(c) = channel {
                        vec![p[c]; 3]
                    } else {
                        p[..3].to_vec()
                    }
                })
                .collect();
            (n, bytes, image.texture_descriptor.mip_level_count)
        } else {
            let color = match mode {
                "albedo" => {
                    let c = m.base_color.to_srgba();
                    [c.red, c.green, c.blue].map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
                }
                "normal_map" => [128, 128, 255],
                _ => [(m.perceptual_roughness * 255.).round() as u8; 3],
            };
            (1, color.to_vec(), 1)
        };
        image::RgbImage::from_raw(n, n, bytes.clone())
            .unwrap()
            .save(output.join(format!("{name}-{index}-{mode}.png")))?;
        let mean = bytes.iter().map(|v| f64::from(*v) / 255.).sum::<f64>() / bytes.len() as f64;
        let sd = (bytes
            .iter()
            .map(|v| (f64::from(*v) / 255. - mean).powi(2))
            .sum::<f64>()
            / bytes.len() as f64)
            .sqrt();
        details.insert(mode.into(),json!({"width":n,"mip_levels":levels,"has_texture":handle.is_some(),"mean_encoded":mean,"std_encoded":sd}));
    }
    let color = m.base_color.to_srgba();
    let attenuation = m.attenuation_color.to_linear();
    Ok(
        json!({"maps":details,"base_color_srgba":[color.red,color.green,color.blue,color.alpha],
        "roughness_multiplier":m.perceptual_roughness,"metallic":m.metallic,"reflectance":m.reflectance,
        "clearcoat":m.clearcoat,"clearcoat_roughness":m.clearcoat_perceptual_roughness,
        "anisotropy":m.anisotropy_strength,"ior":m.ior,"thickness_m":m.thickness,
        "specular_transmission":m.specular_transmission,"diffuse_transmission":m.diffuse_transmission,
        "attenuation_linear_rgb":[attenuation.red,attenuation.green,attenuation.blue],
        "attenuation_distance_m":m.attenuation_distance.is_finite().then_some(m.attenuation_distance),
        "emissive_linear_rgb":[m.emissive.red,m.emissive.green,m.emissive.blue],
        "uv_transform":m.uv_transform.to_cols_array(),"double_sided":m.double_sided}),
    )
}

pub fn run(output: &Path, categories: &[Category]) -> Result<()> {
    setup_globals(None);
    let mut app = create_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::Custom,
            initialize_scene: false,
            headless: true,
            editor: false,
            gizmos: false,
            keybinds: false,
            image_copiers: true,
            num_cameras: 6,
            width: RESOLUTION as f32,
            height: RESOLUTION as f32,
            render_modes: vec![RenderMode::Color, RenderMode::Normal],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    let mut meshes = app.world_mut().remove_resource::<Assets<Mesh>>().unwrap();
    let mut materials = app
        .world_mut()
        .remove_resource::<Assets<StandardMaterial>>()
        .unwrap();
    let floor = materials.add(StandardMaterial {
        base_color: Color::srgb(0.24, 0.26, 0.29),
        perceptual_roughness: 0.85,
        ..default()
    });
    app.world_mut().spawn((
        Mesh3d(meshes.add(Cuboid::new(80., 0.1, 80.))),
        MeshMaterial3d(floor),
        Transform::from_xyz(15., -0.05, 0.),
    ));
    // The same neutral reference pattern behind every sample reveals refraction.
    for i in 0..6 {
        for x in 0..8 {
            for y in 0..6 {
                let c = if (x + y) % 2 == 0 { 0.20 } else { 0.64 };
                let h = materials.add(StandardMaterial {
                    base_color: Color::srgb(c, c, c),
                    perceptual_roughness: 0.8,
                    ..default()
                });
                app.world_mut().spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.15, 0.15, 0.025))),
                    MeshMaterial3d(h),
                    Transform::from_xyz(
                        i as f32 * 6. - 0.525 + x as f32 * 0.15,
                        0.075 + y as f32 * 0.15,
                        -0.46,
                    ),
                ));
            }
        }
    }
    for (p, lux) in [
        (Vec3::new(-3., 6., 5.), 1400.),
        (Vec3::new(5., 2., -1.), 250.),
    ] {
        app.world_mut().spawn((
            DirectionalLight {
                illuminance: lux,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_translation(p).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }
    app.world_mut().insert_resource(meshes);
    app.world_mut().insert_resource(materials);
    let template = IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.5, 0, 0.)
        .map_err(anyhow::Error::msg)?;
    let mut environment = None;
    let mut records = Vec::new();
    for (index, seed) in SEEDS.into_iter().enumerate() {
        let start = Instant::now();
        let mut images = app.world_mut().remove_resource::<Assets<Image>>().unwrap();
        let mut materials = app
            .world_mut()
            .remove_resource::<Assets<StandardMaterial>>()
            .unwrap();
        let mut handles = Vec::new();
        let mut row_metadata = Vec::new();
        for floor_style in 0..3 {
            if !categories.iter().any(|c| c.floor == floor_style) {
                continue;
            }
            let mut scene = template.clone();
            scene.floor_style = floor_style;
            scene
                .apply_appearance(AppearanceSettings {
                    material_seed: Some(seed),
                    ..default()
                })
                .map_err(anyhow::Error::msg)?;
            if categories.iter().any(|c| c.name == "Rug") {
                let mut rug = objects::audit::fixture(ObjectKind::Rug, seed);
                rug.finish_seed = Some(seed);
                scene.objects.push(rug);
            }
            let labels: Vec<_> = categories
                .iter()
                .filter(|c| c.floor == floor_style)
                .map(|c| {
                    let label = if c.name == "Rug" {
                        format!("floormat#rug{}", rugs::slot(seed))
                    } else {
                        format!("swatch#finish{}", audit::finish_slot(c, seed))
                    };
                    (c.surface, label)
                })
                .collect();
            let palette = IndoorMaterials::build_for_parts(
                &scene,
                &mut images,
                &mut materials,
                labels.iter().map(|(s, l)| (*s, l.as_str())),
            );
            if environment.is_none() {
                // The palette's environment handles are placeholders until the
                // room probe bake runs. A studio uses a fixed calibrated cube.
                use bevy::{
                    asset::RenderAssetUsages,
                    render::render_resource::{
                        Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor,
                        TextureViewDimension,
                    },
                };
                let bytes: Vec<u8> = [0.35_f32, 0.35, 0.35, 1.]
                    .into_iter()
                    .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
                    .collect();
                let mut cube = Image::new_fill(
                    Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 6,
                    },
                    TextureDimension::D2,
                    &bytes,
                    TextureFormat::Rgba16Float,
                    RenderAssetUsages::default(),
                );
                cube.texture_view_descriptor = Some(TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::Cube),
                    ..default()
                });
                let cube = images.add(cube);
                environment = Some(EnvironmentMapLight {
                    diffuse_map: cube.clone(),
                    specular_map: cube,
                    intensity: 1.,
                    ..default()
                });
            }
            for c in categories.iter().filter(|c| c.floor == floor_style) {
                let label = if c.name == "Rug" {
                    format!("floormat#rug{}", rugs::slot(seed))
                } else {
                    format!("swatch#finish{}", audit::finish_slot(c, seed))
                };
                let h = palette.for_part(c.surface, &label);
                let m = materials.get(&h).unwrap();
                if c.name == "Rug" {
                    ensure!(
                        m.base_color == Color::WHITE
                            && images
                                .get(
                                    m.base_color_texture
                                        .as_ref()
                                        .ok_or_else(|| anyhow::anyhow!("missing rug atlas"))?
                                )
                                .is_some_and(|i| i.width() == 512),
                        "rug material did not resolve to its dedicated atlas"
                    );
                }
                let actual = maps(output, &c.name, index, m, &images)?;
                if matches!(
                    c.surface,
                    Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
                ) {
                    ensure!(
                        m.uv_transform == bevy::math::Affine2::IDENTITY && m.double_sided,
                        "leaf atlas binding"
                    );
                }
                let mut record = audit::record(c, seed);
                record["actual_material"] = actual;
                record["name"] = json!(c.name);
                record["column"] = json!(index);
                row_metadata.push(record);
                handles.push((c.clone(), h));
            }
        }
        app.world_mut().insert_resource(images);
        app.world_mut().insert_resource(materials);
        for batch in handles.chunks(6) {
            let mut hs: Vec<_> = batch.iter().map(|(_, h)| h.clone()).collect();
            let mut leaves: Vec<_> = batch
                .iter()
                .map(|(c, _)| {
                    matches!(
                        c.surface,
                        Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
                    )
                })
                .collect();
            let mut names: Vec<_> = batch.iter().map(|(c, _)| c.name.clone()).collect();
            while hs.len() < 6 {
                hs.push(hs[0].clone());
                leaves.push(leaves[0]);
                names.push(names[0].clone());
            }
            app.world_mut().insert_resource(Batch {
                handles: hs,
                leaves,
                names,
                environment: environment.clone().unwrap(),
                install: true,
            });
            let capture_start = Instant::now();
            loop {
                app.update();
                ensure!(
                    capture_start.elapsed() < Duration::from_secs(180),
                    "material capture timeout"
                );
                if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                    .unwrap()
                    .lock()
                    .unwrap()
                    .try_recv()
                {
                    ensure!(sample.views.len() == 6, "incomplete material views");
                    for ((c, _), v) in batch.iter().zip(&sample.views) {
                        for (mode, data) in [("color", &v.color), ("normal", &v.normal)] {
                            ensure!(
                                data.len() == (RESOLUTION * RESOLUTION * 16) as usize,
                                "short rendered channel"
                            );
                            let floats: &[f32] = bytemuck::cast_slice(data);
                            let mut bytes = Vec::new();
                            for p in floats.as_chunks::<4>().0 {
                                for value in &p[..3] {
                                    ensure!(value.is_finite(), "nonfinite render");
                                    let v = if mode == "color" {
                                        bevy_zeroverse::render::color::linear_to_srgb(*value)
                                    } else {
                                        *value
                                    };
                                    bytes.push((v.clamp(0., 1.) * 255.).round() as u8);
                                }
                            }
                            image::RgbImage::from_raw(RESOLUTION, RESOLUTION, bytes)
                                .unwrap()
                                .save(output.join(format!("{}-{index}-{mode}.png", c.name)))?;
                        }
                    }
                    break;
                }
            }
        }
        records.extend(row_metadata);
        println!(
            "Material studio seed {seed}: {} swatches in {:.2}s",
            categories.len(),
            start.elapsed().as_secs_f64()
        );
    }
    fs::write(
        output.join("renders.json"),
        serde_json::to_vec_pretty(&json!({"schema_version":1,
        "build_provenance":bevy_zeroverse::provenance::capture_provenance(),"tool_sources":super::tool_sources(),
        "adapter":app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|a|format!("{:?}",**a)),
        "selected_seeds":SEEDS,"image_size":[RESOLUTION,RESOLUTION],
        "map_baker":"production CPU reference maps; native Bevy PBR rendering", "rig":"fixed 1400/250 lux directional lights, shadows, neutral 0.35 cd/m2 reflection cube, default capture exposure, 40 degree vertical FOV; metric sphere/beveled panel for repeating materials, normalized planes for screen/phone/TV/board/art, production whole-rug fields, normalized production leaf blades",
        "records":records}))?,
    )?;
    galleries(output, categories)?;
    Ok(())
}

fn galleries(output: &Path, categories: &[Category]) -> Result<()> {
    let mut html = String::from("<!doctype html><meta charset=utf-8><title>Production PBR surfaces</title><style>body{background:#171b23;color:#eef0f5;font:16px system-ui;margin:32px;max-width:1200px}img{width:100%}a{color:#a5ceff}section{margin:40px 0}</style><h1>Production PBR surfaces</h1><p>Native Bevy rendering, fixed lighting and camera rig. Six material seeds per category: 0, 1, 4, 19, 53, 101 (left to right, top to bottom). Content is sampled from the production scene palette. <a href=renders.json>Parameters, maps and renderer provenance</a>.</p>");
    for category in categories {
        for mode in ["color", "normal"] {
            let mut grid = image::RgbImage::new(RESOLUTION * 3, RESOLUTION * 2);
            for index in 0..SEEDS.len() {
                let capture =
                    image::open(output.join(format!("{}-{index}-{mode}.png", category.name)))?
                        .to_rgb8();
                image::imageops::replace(
                    &mut grid,
                    &capture,
                    (index % 3) as i64 * i64::from(RESOLUTION),
                    (index / 3) as i64 * i64::from(RESOLUTION),
                );
            }
            grid.save(output.join(format!("{}-{mode}-gallery.png", category.name)))?;
        }
        html.push_str(&format!("<section><h2>{0}</h2><img src={0}-color-gallery.png><details><summary>Aligned world-normal annotation</summary><img src={0}-normal-gallery.png></details></section>",category.name));
    }
    fs::write(output.join("index.html"), html)?;
    Ok(())
}
