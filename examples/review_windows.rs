//! Controlled production window assemblies: every treatment, unequal panes,
//! normal/grazing views, and paired solid/through annotation policies.
//! `cargo run --example review_windows -- --output out/windows [--open] [--night]`
#![recursion_limit = "256"]
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::{glass::AnnotationGlass, semantic::SemanticLabel, RenderMode},
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            architecture::{
                self,
                facade::{ExteriorProgram, Facade, FacadeSide, Shade, WindowOpening},
            },
            layout::{IndoorLayout, IndoorManifest},
            materials::{glazing::GlazingProgram, IndoorMaterials, Surface},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use clap::Parser;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    open: bool,
    #[arg(long)]
    night: bool,
    /// Move the calibrated backdrop inside and observe the exterior pane face.
    #[arg(long)]
    exterior: bool,
    #[arg(long, value_enum, default_value = "surface")]
    annotation_glass: AnnotationGlass,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.output.exists(), "use a new output directory");
    std::fs::create_dir_all(&args.output)?;
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let mut scene = IndoorManifest::generate_with_humans(81, IndoorLayout::Conference, 0.5, 0, 0.)
        .map_err(anyhow::Error::msg)?;
    // These are isolated diagnostic bays, not sampled distribution rooms. Extract
    // the actual facade meshes after construction; do not reimplement sashes/UVs.
    scene.room_size = Vec3::new(30., 3.2, 4.);
    scene.envelope = None;
    scene.objects.clear();
    scene.humans.clear();
    scene.world_yaw = 0.;
    let p = scene.program.as_mut().unwrap();
    p.partitions.clear();
    p.zones.clear();
    if let Some(domain) = &mut p.domain {
        domain.target_lux = 500.;
        domain.photometry.sky_radiance = if args.night {
            Vec3::new(0.1, 0.15, 0.25)
        } else {
            Vec3::new(450., 500., 600.)
        };
        domain.photometry.sun_lux = 0.;
        domain.photometry.environment_intensity = 35.;
    }
    let mut openings = Vec::new();
    for treatment in 0..7 {
        let x = (treatment as f32 - 3.) * 4.;
        let mut glazing = GlazingProgram::sample(81, treatment, true);
        glazing.treatment = treatment as u8;
        glazing.opening = if args.open { 1. } else { 0. };
        glazing.band_m = Vec2::new(0.8, 1.7);
        glazing.pitch_m = 0.12;
        glazing.duty = 0.42;
        glazing.frost_roughness = 0.40;
        glazing.validate().map_err(anyhow::Error::msg)?;
        // Off-center transom and different opening heights test metric anchoring.
        openings.push(WindowOpening {
            glazing: Some(glazing),
            min: Vec2::new(x - 1.15, 0.30),
            max: Vec2::new(x + 1.15, 2.6 + 0.12 * (treatment % 2) as f32),
            columns: 2,
            transom: 0.69,
            transoms: vec![],
            operable: true,
        });
    }
    scene.exterior = Some(ExteriorProgram {
        exposure_probability: 1.,
        facades: vec![Facade {
            joinery: None,
            side: FacadeSide::Rear,
            openings,
            frame: Surface::Metal,
            frame_width: 0.05,
            frame_depth: 0.09,
            recess: 0.065,
            sill_projection: 0.12,
            treatment: None,
            shade: Shade::None,
            shade_coverage: 0.,
            shade_tilt: 0.,
        }],
    });
    std::fs::write(
        args.output.join("fixture.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "build":bevy_zeroverse::provenance::capture_provenance(), "facade":scene.exterior,
            "image_size":[512,512], "open":args.open, "night":args.night, "exterior":args.exterior,
            "annotation_glass":args.annotation_glass, "ev100":10.,
            "protocol":"7 treatment fields x normal/grazing incidence; production construction and materials, controlled emissive checker backdrop; repeated start pose then a 4 mm lateral camera move test stationary stability and near-static edges. These isolated bays are not population samples or a transport-accuracy reference."
        }))?,
    )?;
    setup_globals(None);
    let modes = vec![RenderMode::Color, RenderMode::Depth, RenderMode::Semantic];
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
            num_cameras: 14,
            width: 512.,
            height: 512.,
            render_modes: modes.clone(),
            playback_steps: 3,
            annotation_glass: args.annotation_glass,
            ..default()
        }),
        false,
    );
    let night = args.night;
    let exterior = args.exterior;
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut meshes: ResMut<Assets<Mesh>>,
              mut images: ResMut<Assets<Image>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            let set = IndoorMaterials::build(&scene, &mut *images, &mut *materials);
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            let mut facade = architecture::architecture(&scene);
            facade
                .parts
                .retain(|(_, label), _| label.starts_with("window"));
            for g in facade.parts.values_mut() {
                // Exclude the separate neighboring-room glazing from this fixture.
                g.indices = g
                    .indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter(|t| t.iter().all(|&i| g.positions[i as usize][2] < -1.7))
                    .flatten()
                    .copied()
                    .collect();
            }
            facade.spawn(root, &mut commands, &mut meshes, &set);
            let cell = meshes.add(Cuboid::new(0.249, 0.249, 0.1));
            let board_materials: Vec<_> = [
                Vec3::splat(0.025),
                Vec3::splat(0.8),
                Vec3::new(0.1, 0.25, 0.6),
            ]
            .iter()
            .map(|&c| {
                materials.add(StandardMaterial {
                    base_color: Color::BLACK,
                    emissive: LinearRgba::rgb(c.x, c.y, c.z) * if night { 15. } else { 1500. },
                    emissive_exposure_weight: 1.,
                    perceptual_roughness: 0.9,
                    ..default()
                })
            })
            .collect();
            for i in 0..7 {
                let x = (i as f32 - 3.) * 4.;
                for column in 0..14 {
                    for row in 0..14 {
                        commands.spawn((
                            Mesh3d(cell.clone()),
                            MeshMaterial3d(board_materials[(column + row) % 3].clone()),
                            SemanticLabel::Wall,
                            Transform::from_xyz(
                                x + (column as f32 - 6.5) * 0.25,
                                0.05 + row as f32 * 0.25,
                                if exterior { -1.3 } else { -2.8 },
                            ),
                            ChildOf(root),
                        ));
                    }
                }
                let facing = if exterior { -1. } else { 1. };
                for (angle, offset) in [
                    Vec3::new(0., 0., facing * 3.),
                    Vec3::new(2.8, 0., facing * 1.2),
                ]
                .into_iter()
                .enumerate()
                {
                    let target = Vec3::new(x, 1.45, -2.065);
                    commands.spawn((
                        CaptureCameraIndex(i * 2 + angle),
                        ZeroverseCamera {
                            perspective_sampler: PerspectiveSampler::exact(52.),
                            trajectory: TrajectorySampler::Linear {
                                start: ExtrinsicsSampler {
                                    position: ExtrinsicsSamplerType::Transform(
                                        Transform::from_translation(target + offset),
                                    ),
                                    looking_at: LookingAtSampler::Exact(target),
                                    ..default()
                                },
                                end: ExtrinsicsSampler {
                                    position: ExtrinsicsSamplerType::Transform(
                                        Transform::from_translation(
                                            target + offset + Vec3::X * 0.004,
                                        ),
                                    ),
                                    looking_at: LookingAtSampler::Exact(target + Vec3::X * 0.004),
                                    ..default()
                                },
                            },
                            ..default()
                        },
                        set.environment.clone(),
                        ChildOf(root),
                    ));
                }
            }
            commands.spawn((
                DirectionalLight {
                    illuminance: if night { 1400. } else { 500. },
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-3., 5., if exterior { -10. } else { 8. }).looking_at(
                    if exterior {
                        Vec3::new(0., 0., -2.)
                    } else {
                        Vec3::ZERO
                    },
                    Vec3::Y,
                ),
                ChildOf(root),
            ));
        },
    );
    app.add_systems(
        Last,
        |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
            for e in &cameras {
                commands
                    .entity(e)
                    .insert(bevy::camera::Exposure { ev100: 10. });
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
        timesteps: vec![0., 1.],
        ..default()
    });
    let start = Instant::now();
    let sample = loop {
        app.update();
        ensure!(
            start.elapsed() < Duration::from_secs(240),
            "window capture timeout"
        );
        ensure!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "window capture failed"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            break sample;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    ensure!(sample.views.len() == 42, "incomplete window capture");
    let mut records = Vec::new();
    for (index, view) in sample.views.iter().enumerate() {
        let rgba: &[f32] = bytemuck::cast_slice(&view.color);
        ensure!(rgba.iter().all(|v| v.is_finite()), "nonfinite window RGB");
        let rgb = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| {
                p[..3].iter().map(|v| {
                    (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.).round()
                        as u8
                })
            })
            .collect();
        image::RgbImage::from_raw(512, 512, rgb)
            .unwrap()
            .save(args.output.join(format!("view_{index:02}.png")))?;
        std::fs::write(
            args.output
                .join(format!("view_{index:02}.semantic.rgba32f")),
            &view.semantic,
        )?;
        std::fs::write(
            args.output.join(format!("view_{index:02}.depth.rgba32f")),
            &view.depth,
        )?;
        records.push(serde_json::json!({"index":index,"time":view.time,"world_from_view":view.world_from_view,"fovy":view.fovy}));
    }
    std::fs::write(
        args.output.join("capture.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "elapsed_seconds":start.elapsed().as_secs_f64(), "annotation_precision":sample.annotation_precision,
            "views":records, "completed":true
        }))?,
    )?;
    Ok(())
}
