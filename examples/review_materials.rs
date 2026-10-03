//! Fixed metric swatches under grazing light, using the production PBR/capture path.
//! `cargo run --example review_materials -- out/materials 81`
//! Add `--no-normal-prepass` to check forward/prepass PBR agreement.
//! Add `--close-up` for 25 cm samples that resolve the textile/leather grain.
#![recursion_limit = "256"]
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{semantic::SemanticLabel, RenderMode},
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            geometry::Geometry,
            layout::{IndoorLayout, IndoorManifest},
            materials::{IndoorMaterials, Surface},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

const SURFACES: [Surface; 20] = [
    Surface::Wood,
    Surface::Fabric,
    Surface::FabricAlt,
    Surface::Leather,
    Surface::Paint,
    Surface::Concrete,
    Surface::Ceramic,
    Surface::Metal,
    Surface::Chrome,
    Surface::Plastic,
    Surface::Floor,
    Surface::Accent,
    Surface::Ceiling,
    Surface::WoodEdge,
    Surface::Soil,
    Surface::Leaf,
    Surface::LeafLight,
    Surface::LeafVariegated,
    Surface::Terracotta,
    Surface::Bark,
];

fn factor(name: &str) -> anyhow::Result<Option<f32>> {
    let prefix = format!("--{name}=");
    let v: Option<f32> = std::env::args()
        .find_map(|s| s.strip_prefix(&prefix).map(str::to_owned))
        .map(|s| s.parse())
        .transpose()?;
    anyhow::ensure!(
        v.is_none_or(|v| v.is_finite() && (0.0..=1.).contains(&v)),
        "{name} must be 0..1"
    );
    Ok(v)
}

fn main() -> anyhow::Result<()> {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let output = PathBuf::from(std::env::args().nth(1).unwrap_or("out/materials".into()));
    let seed = std::env::args().nth(2).unwrap_or("81".into()).parse()?;
    std::fs::create_dir_all(&output)?;
    let normal_prepass = !std::env::args().any(|arg| arg == "--no-normal-prepass");
    let ibl_only = std::env::args().any(|arg| arg == "--ibl-only");
    let variant: Option<usize> = std::env::args()
        .find_map(|arg| arg.strip_prefix("--variant=").map(str::to_owned))
        .map(|s| s.parse())
        .transpose()?;
    anyhow::ensure!(variant.is_none_or(|v| v < 12), "variant must be 0..12");
    let floor_style: Option<u32> = std::env::args()
        .find_map(|s| s.strip_prefix("--floor-style=").map(str::to_owned))
        .map(|s| s.parse())
        .transpose()?;
    anyhow::ensure!(
        floor_style.is_none_or(|s| s < 3),
        "floor-style must be 0..2"
    );
    let stone_mix = factor("stone-mix")?;
    let wall_texture = factor("wall-texture")?;
    let wood_stain = factor("wood-stain")?;
    anyhow::ensure!(
        variant.is_none()
            || (stone_mix.is_none() && wall_texture.is_none() && wood_stain.is_none()),
        "finish-slot resampling cannot be combined with substrate factor overrides"
    );
    let scale = if std::env::args().any(|arg| arg == "--close-up") {
        0.25
    } else {
        1.0
    };
    std::fs::write(
        output.join("review.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "seed": seed, "sample_width_m": scale, "normal_prepass": normal_prepass,
            "surfaces": SURFACES, "image_size": [512, 512], "png_encoding": "sRGB",
            "key_fill_lux": if ibl_only { [0, 0] } else { [2400, 350] },
            "reflection_target_lux": 500, "reflection_sky_radiance": [450, 500, 600],
            "uniform_ambient": false, "ibl_only": ibl_only, "camera_fovy_degrees": 39,
            "finish_slot":variant,
            "floor_style":floor_style,"stone_mix":stone_mix,"wall_texture":wall_texture,"wood_stain":wood_stain,
        }))?,
    )?;
    let mut scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.5, 0, 0.0)
            .map_err(anyhow::Error::msg)?;
    if let Some(s) = floor_style {
        scene.floor_style = s;
        let p = scene.program.as_mut().unwrap();
        // Replace only the floor recipe before its one-time specialization.
        p.materials[Surface::Floor as usize] =
            bevy_zeroverse::scene::procedural_indoor::materials::program::sample_with_floor(
                scene.seed, s,
            )
            .map_err(anyhow::Error::msg)?[Surface::Floor as usize]
                .clone();
    }
    for r in &mut scene.program.as_mut().unwrap().materials {
        if let Some(w) = r.wood.as_mut() {
            if let Some(v) = wood_stain {
                w.stain_strength = v;
            }
        }
        if let Some(c) = r.coating.as_mut() {
            if let Some(v) = wall_texture {
                c.texture_mix = v;
                if r.surface != Surface::Ceramic {
                    r.relief_m = 0.000018 + v.powi(2) * 0.00065;
                }
            }
        }
        if r.surface == Surface::Floor {
            if let Some(v) = stone_mix {
                r.mineral.as_mut().unwrap().marble_mix = v;
            }
        }
    }
    scene
        .program
        .as_ref()
        .unwrap()
        .validate(&scene)
        .map_err(anyhow::Error::msg)?;
    // Isolate finish variation from scene photometry. Seed 81 otherwise samples
    // an almost unlit room, obscuring IBL behind the review's separate key/fill.
    if let Some(domain) = scene.program.as_mut().and_then(|p| p.domain.as_mut()) {
        domain.target_lux = 500.;
        domain.photometry.sun_lux = 0.;
        domain.photometry.sky_radiance = Vec3::new(450., 500., 600.);
        domain.photometry.environment_intensity = 35.;
        domain.photometry.active_fraction = 1.;
        domain.photometry.circuit_contrast = 0.;
    }
    scene.target_lux = 500.;
    scene.light_kelvin = 4500.;
    std::fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&scene)?,
    )?;
    std::fs::write(
        output.join("engine.json"),
        serde_json::to_vec_pretty(&bevy_zeroverse::CAPTURE_ENGINE_IDENTITY)?,
    )?;
    std::fs::write(
        output.join("build_provenance.json"),
        serde_json::to_vec_pretty(&bevy_zeroverse::provenance::capture_provenance())?,
    )?;
    setup_globals(None);
    let modes = vec![RenderMode::Color, RenderMode::Normal];
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
            num_cameras: SURFACES.len(),
            width: 512.0,
            height: 512.0,
            render_modes: modes.clone(),
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    if !normal_prepass {
        app.add_systems(PreUpdate, |mut commands: Commands,
            cameras: Query<Entity, With<bevy::core_pipeline::prepass::NormalPrepass>>| {
            for entity in &cameras {
                commands.entity(entity).remove::<bevy::core_pipeline::prepass::NormalPrepass>();
            }
        });
    }
    let maps_output = output.join("maps");
    std::fs::create_dir_all(&maps_output)?;
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut meshes: ResMut<Assets<Mesh>>,
              mut images: ResMut<Assets<Image>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            let set = IndoorMaterials::build(&scene, &mut *images, &mut *materials);
            let handle = |surface| {
                variant.map_or_else(
                    || set.get(surface),
                    |slot| set.for_part(surface, &format!("review#finish{slot}")),
                )
            };
            let environment = images.get(&set.environment.specular_map).unwrap();
            std::fs::write(
                maps_output.join("environment.rgba16f"),
                environment.data.as_ref().unwrap(),
            )
            .unwrap();
            for surface in SURFACES {
                let material = materials.get(&handle(surface)).unwrap();
                for (name, handle) in [
                    ("albedo", &material.base_color_texture),
                    ("normal", &material.normal_map_texture),
                    ("roughness", &material.metallic_roughness_texture),
                ] {
                    if let Some(image) = handle.as_ref().and_then(|h| images.get(h)) {
                        let size = image.texture_descriptor.size;
                        let bytes = image.data.as_ref().unwrap()
                            [..(size.width * size.height * 4) as usize]
                            .to_vec();
                        image::RgbaImage::from_raw(size.width, size.height, bytes)
                            .unwrap()
                            .save(maps_output.join(format!("{surface:?}.{name}.png")))
                            .unwrap();
                    }
                }
            }
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            let ground = materials.add(StandardMaterial {
                base_color: Color::srgb(0.24, 0.25, 0.27),
                perceptual_roughness: 0.92,
                ..default()
            });
            for (i, surface) in SURFACES.into_iter().enumerate() {
                let center = Vec3::X * i as f32 * 4.0;
                // One-metre panel and a rounded sample expose both texture scale and BRDF.
                let mut geometry = Geometry::default();
                let leaf = matches!(
                    surface,
                    Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated
                );
                if leaf {
                    geometry.leaf(
                        0.65 * scale,
                        0.42 * scale,
                        0.04 * scale,
                        Transform::from_translation(center + Vec3::Y * 0.16 * scale)
                            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                    );
                } else {
                    geometry.cuboid(
                        Vec3::new(1.0, 0.65, 0.09) * scale,
                        0.022 * scale,
                        Transform::from_translation(center + Vec3::Y * 0.49 * scale),
                    );
                }
                commands.spawn((
                    Mesh3d(meshes.add(geometry.into_mesh())),
                    MeshMaterial3d(handle(surface)),
                    SemanticLabel::Wall,
                    ChildOf(root),
                ));
                if !leaf {
                    let radius = 0.15 * scale;
                    let mut sphere = Sphere::new(radius).mesh().uv(64, 32);
                    // Production panels use metre UVs. Give the BRDF sphere the
                    // same physical scale instead of stretching a 0..1 texture.
                    if let Some(bevy::mesh::VertexAttributeValues::Float32x2(uvs)) =
                        sphere.attribute_mut(Mesh::ATTRIBUTE_UV_0)
                    {
                        for uv in uvs {
                            uv[0] *= std::f32::consts::TAU * radius;
                            uv[1] *= std::f32::consts::PI * radius;
                        }
                    }
                    sphere.generate_tangents().unwrap();
                    commands.spawn((
                        Mesh3d(meshes.add(sphere)),
                        MeshMaterial3d(handle(surface)),
                        SemanticLabel::Wall,
                        Transform::from_translation(center + Vec3::new(0.24, 0.15, 0.27) * scale),
                        ChildOf(root),
                    ));
                }
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(2.2, 0.08, 1.5))),
                    MeshMaterial3d(ground.clone()),
                    SemanticLabel::Floor,
                    Transform::from_translation(center - Vec3::Y * 0.04),
                    ChildOf(root),
                ));
                let target = center + Vec3::Y * 0.42 * scale;
                commands.spawn((
                    CaptureCameraIndex(i),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(39.0),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(
                                        target + Vec3::new(0.22, 0.22, 1.8) * scale,
                                    ),
                                ),
                                looking_at: LookingAtSampler::Exact(target),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    set.environment.clone(),
                    ChildOf(root),
                ));
            }
            for (from, lux) in [
                (Vec3::new(-4., 3., 2.), 2400.),
                (Vec3::new(2., 1., 4.), 350.),
            ] {
                if ibl_only {
                    continue;
                }
                commands.spawn((
                    DirectionalLight {
                        illuminance: lux,
                        shadow_maps_enabled: true,
                        ..default()
                    },
                    Transform::from_translation(from).looking_at(Vec3::ZERO, Vec3::Y),
                ));
            }
        },
    );
    app.finish();
    app.cleanup();
    app.insert_resource(bevy::light::GlobalAmbientLight::NONE);
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: modes,
        warmup_frames: 25,
        frames: 3,
        timesteps: vec![],
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
        anyhow::ensure!(
            start.elapsed() < Duration::from_secs(120),
            "capture timed out"
        );
        anyhow::ensure!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "capture failed"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            anyhow::ensure!(sample.views.len() == SURFACES.len(), "missing swatch views");
            for (surface, view) in SURFACES.into_iter().zip(sample.views) {
                let rgba: &[f32] = bytemuck::cast_slice(&view.color);
                anyhow::ensure!(rgba.iter().all(|v| v.is_finite()), "non-finite RGB");
                let rgb = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.)
                                .round() as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(512, 512, rgb)
                    .unwrap()
                    .save(output.join(format!("{surface:?}.png")))?;
                std::fs::write(
                    output.join(format!("{surface:?}.normal.rgba32f")),
                    view.normal,
                )?;
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
