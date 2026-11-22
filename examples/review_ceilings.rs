//! Local geometry/PBR/annotation contact sheets for production ceiling modules.
//! Geometry is selected from generated rooms, isolated and oriented for inspection.
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
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            architecture::ceiling::{CeilingModule, CeilingShape},
            layout::{IndoorLayout, IndoorManifest},
            materials::{IndoorMaterials, Surface},
            objects::Assembly,
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use clap::Parser;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
const SIZE: u32 = 512;
#[derive(Component)]
struct ReviewRoot;
#[derive(Resource)]
struct Batch {
    assemblies: Vec<Assembly>,
    targets: Vec<(Vec3, Vec3)>,
    materials: std::sync::Arc<IndoorMaterials>,
    install: bool,
}

fn install(
    mut commands: Commands,
    mut batch: ResMut<Batch>,
    roots: Query<Entity, With<ReviewRoot>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut sampler: ResMut<SamplerState>,
) {
    if !batch.install {
        return;
    }
    for e in &roots {
        commands.entity(e).despawn();
    }
    let root = commands
        .spawn((
            ReviewRoot,
            ZeroverseSceneRoot,
            SceneAabbNode,
            Transform::IDENTITY,
            Visibility::default(),
        ))
        .id();
    for (i, a) in std::mem::take(&mut batch.assemblies)
        .into_iter()
        .enumerate()
    {
        a.spawn(root, &mut commands, &mut meshes, &batch.materials);
        let (eye, target) = batch.targets[i];
        commands.spawn((
            ZeroverseCamera {
                resolution: Some(UVec2::splat(SIZE)),
                perspective_sampler: PerspectiveSampler::exact(48.),
                trajectory: TrajectorySampler::Static {
                    start: ExtrinsicsSampler {
                        looking_at: LookingAtSampler::Exact(target),
                        position: ExtrinsicsSamplerType::Transform(Transform::from_translation(
                            eye,
                        )),
                        ..default()
                    },
                },
                ..default()
            },
            CaptureCameraIndex(i),
            batch.materials.environment.clone(),
            ChildOf(root),
        ));
    }
    *sampler = SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![
            RenderMode::Color,
            RenderMode::Normal,
            RenderMode::Depth,
            RenderMode::Semantic,
        ],
        warmup_frames: 20,
        frames: 1,
        timesteps: vec![],
        ..default()
    };
    batch.install = false;
}
#[derive(Parser)]
struct Args {
    output: PathBuf,
    /// Show each installed mineral-fiber edge and facing at its original metric scale.
    #[arg(long)]
    tile_edges: bool,
}

fn main() -> Result<()> {
    let Args { output, tile_edges } = Args::parse();
    ensure!(!output.exists(), "use a new output directory");
    fs::create_dir_all(&output)?;
    let mut selected: BTreeMap<String, Vec<(IndoorManifest, CeilingModule)>> = BTreeMap::new();
    let (families, examples) = if tile_edges { (9, 1) } else { (13, 2) };
    for seed in 0..512 {
        let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
            .map_err(anyhow::Error::msg)?;
        for module in &scene.ceiling_installation.as_ref().unwrap().modules {
            let name = if tile_edges {
                let Some(CeilingShape::LayIn { tile, .. }) =
                    module.detail.as_ref().map(|d| &d.shape)
                else {
                    continue;
                };
                format!("{:?}-{:?}", tile.edge, tile.facing)
            } else {
                format!("{:?}", module.system)
            };
            let entries = selected.entry(name).or_default();
            if entries.len() < examples && entries.iter().all(|(s, _)| s.seed != seed) {
                entries.push((scene.clone(), module.clone()));
            }
        }
        if selected.len() == families && selected.values().all(|v| v.len() == examples) {
            break;
        }
    }
    ensure!(
        selected.len() == families && selected.values().all(|v| v.len() == examples),
        "missing ceiling types"
    );
    let cases: Vec<_> = selected.into_values().flatten().collect();
    let batch_size = if tile_edges { 3 } else { 6 };
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
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
            num_cameras: batch_size,
            width: SIZE as f32,
            height: SIZE as f32,
            playback_steps: 1,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Normal,
                RenderMode::Depth,
                RenderMode::Semantic,
            ],
            ..default()
        }),
        false,
    );
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    let lights = if tile_edges {
        // Raking illumination resolves millimetre-scale edges and pore relief.
        [
            (Vec3::new(-3.6, -1.55, 1.), 1100.),
            (Vec3::new(1.5, -2., -1.), 110.),
        ]
    } else {
        [
            (Vec3::new(-3., -5., 5.), 1400.),
            (Vec3::new(2., -2., -4.), 600.),
        ]
    };
    for (p, lux) in lights {
        app.world_mut().spawn((
            DirectionalLight {
                illuminance: lux,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_translation(p).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }
    let palette_scene =
        IndoorManifest::generate_with_humans(41, IndoorLayout::Conference, 0.5, 0, 0.)
            .map_err(anyhow::Error::msg)?;
    let mut images = app.world_mut().remove_resource::<Assets<Image>>().unwrap();
    let mut materials = app
        .world_mut()
        .remove_resource::<Assets<StandardMaterial>>()
        .unwrap();
    let mut keys: BTreeSet<_> = cases
        .iter()
        .flat_map(|(scene, module)| module.geometry(scene).parts.into_keys())
        .collect();
    keys.insert((Surface::Concrete, "ceiling".into()));
    let mut palette = IndoorMaterials::build_for_parts(
        &palette_scene,
        &mut images,
        &mut materials,
        keys.iter().map(|(surface, key)| (*surface, key.as_str())),
    );
    if tile_edges {
        palette.environment.intensity *= 0.2;
    }
    let palette = std::sync::Arc::new(palette);
    app.world_mut().insert_resource(images);
    app.world_mut().insert_resource(materials);
    let mut records = Vec::new();
    for (batch, chunk) in cases.chunks(batch_size).enumerate() {
        let mut assemblies = Vec::new();
        let mut targets = Vec::new();
        for (i, (scene, module)) in chunk.iter().enumerate() {
            let mut timings = Vec::new();
            for _ in 0..32 {
                let before = Instant::now();
                std::hint::black_box(module.geometry(scene));
                timings.push(before.elapsed().as_secs_f64() * 1e6);
            }
            timings.sort_by(f64::total_cmp);
            let mut a = module.geometry(scene);
            let triangles: usize = a.parts.values().map(|g| g.indices.len() / 3).sum();
            let shift = Vec3::new(i as f32 * 20., 3., 0.);
            let rotation = Quat::from_rotation_y(if module.size.x < module.size.z {
                std::f32::consts::FRAC_PI_2
            } else {
                0.
            });
            for g in a.parts.values_mut() {
                for p in &mut g.positions {
                    *p = (rotation * (Vec3::from_array(*p) - module.center) + shift).to_array();
                }
                for n in &mut g.normals {
                    *n = (rotation * Vec3::from_array(*n)).to_array();
                }
            }
            // Small slab patch from the same analytic roof. Diagnostic cameras
            // look upward at the real underside, never through the module.
            let roof = scene.ceiling_height(module.center.xz());
            let h_x = (scene.ceiling_height(module.center.xz() + Vec2::X * 0.01) - roof) / 0.01;
            let h_z = (scene.ceiling_height(module.center.xz() + Vec2::Y * 0.01) - roof) / 0.01;
            let n = rotation * Vec3::new(-h_x, 1., -h_z).normalize();
            let slab_size = Vec3::new(
                module.size.x.max(module.size.z) * 1.04,
                0.08,
                module.size.x.min(module.size.z) * 1.08,
            );
            a.part(Surface::Concrete, "ceiling").cuboid(
                slab_size,
                0.,
                Transform::from_translation(shift + Vec3::Y * (roof - module.center.y + 0.04))
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, n)),
            );
            let span = module.size.max_element().max(1.4);
            let target = if tile_edges {
                shift - Vec3::Y * module.size.y * 0.5
            } else {
                shift
            };
            let offset = if tile_edges {
                Vec3::new(0.28, -0.64, 0.37)
            } else {
                Vec3::new(span * 0.25, -span * 0.58, span * 1.10)
            };
            targets.push((target + offset, target));
            records.push(
                json!({"index":batch*batch_size+i,"system":module.system,"room_seed":scene.seed,
                "module":module,"triangles":triangles,"material_batches":a.parts.len(),
                "median_build_microseconds":timings[timings.len()/2]}),
            );
            assemblies.push(a);
        }
        app.world_mut().insert_resource(Batch {
            assemblies,
            targets,
            materials: palette.clone(),
            install: true,
        });
        let started = Instant::now();
        loop {
            app.update();
            ensure!(
                started.elapsed() < Duration::from_secs(180),
                "ceiling capture timeout for batch {batch}, readiness {:?}",
                app.world()
                    .resource::<bevy_zeroverse::sample::CaptureReadiness>()
            );
            ensure!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "ceiling capture failure: {:?}",
                app.world().resource::<CaptureFailure>().0
            );
            if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                ensure!(
                    sample.views.len() == chunk.len(),
                    "incomplete ceiling capture"
                );
                for (i, v) in sample.views.iter().enumerate() {
                    let mut visible = 0;
                    let color = SemanticLabel::Ceiling.color().to_linear();
                    for (mode, bytes) in [
                        ("color", &v.color),
                        ("normal", &v.normal),
                        ("depth", &v.depth),
                        ("semantic", &v.semantic),
                    ] {
                        ensure!(
                            bytes.len() == (SIZE * SIZE * 16) as usize,
                            "incorrect {mode} length"
                        );
                        let values: &[f32] = bytemuck::cast_slice(bytes);
                        ensure!(values.iter().all(|x| x.is_finite()), "nonfinite {mode}");
                        let pixels: Vec<u8> = values
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| {
                                if mode == "semantic"
                                    && (p[0] - color.red).abs() < 0.002
                                    && (p[1] - color.green).abs() < 0.002
                                    && (p[2] - color.blue).abs() < 0.002
                                {
                                    visible += 1;
                                }
                                [0, 1, 2].map(|j| {
                                    let v = match mode {
                                        "color" | "semantic" => {
                                            bevy_zeroverse::render::color::linear_to_srgb(p[j])
                                        }
                                        "depth" => p[0] / 15.,
                                        _ => p[j],
                                    };
                                    (v.clamp(0., 1.) * 255.).round() as u8
                                })
                            })
                            .collect();
                        image::RgbImage::from_raw(SIZE, SIZE, pixels)
                            .unwrap()
                            .save(
                                output.join(format!("{:02}-{mode}.png", batch * batch_size + i)),
                            )?;
                    }
                    ensure!(visible > 100, "missing ceiling pixels");
                    records[batch * batch_size + i]["ceiling_pixels"] = json!(visible);
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        println!(
            "Ceiling batch {batch}: {:.2}s",
            started.elapsed().as_secs_f64()
        );
    }
    fs::write(
        output.join("review.json"),
        serde_json::to_vec_pretty(&json!({
            "provenance":bevy_zeroverse::provenance::capture_provenance(),"image_size":[SIZE,SIZE],
            "protocol":"Production modules under fixed inspection lighting and palette, at their installed metric scale.","tile_edges":tile_edges,"records":records
        }))?,
    )?;
    Ok(())
}
