//! Local matched window-covering review; production geometry and PBR materials.
//! `cargo run --example review_window_treatments -- out/window_treatments/renders`
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
            architecture::facade::{
                self, treatments::ShadeProgram, Facade, FacadeSide, Shade, WindowOpening,
            },
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
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

/// Render one installation at different pixel densities, distances and view angles.
#[derive(Parser, Resource)]
struct Review {
    #[arg(default_value = "out/window_treatments/renders")]
    output: PathBuf,
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(32..=2048))]
    size: u32,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=4))]
    ssaa: u32,
    #[arg(long, default_value_t = 4.535416)]
    distance: f32,
    #[arg(long, default_value_t = 14.036243)]
    yaw: f32,
    /// Vertical displacement in metres for measuring subpixel phase changes.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    rise: f32,
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..=3))]
    row: Option<u32>,
    #[arg(long, value_parser = ["venetian", "roller", "vertical", "roman", "pleated", "curtain", "paneltrack"])]
    kind: Option<String>,
}

const KINDS: [Shade; 7] = [
    Shade::Venetian,
    Shade::Roller,
    Shade::Vertical,
    Shade::Roman,
    Shade::Pleated,
    Shade::Curtain,
    Shade::PanelTrack,
];
#[derive(Component)]
struct ReviewRoot;
#[derive(Resource)]
struct Batch {
    assemblies: Vec<Assembly>,
    materials: IndoorMaterials,
    install: bool,
}

fn install(
    mut commands: Commands,
    review: Res<Review>,
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
        let target = Vec3::new(i as f32 * 8., 1.55 + review.rise, 0.);
        let (sin, cos) = review.yaw.to_radians().sin_cos();
        commands.spawn((
            ZeroverseCamera {
                resolution: Some(UVec2::splat(review.size)),
                perspective_sampler: PerspectiveSampler::exact(49.),
                trajectory: TrajectorySampler::Static {
                    start: ExtrinsicsSampler {
                        looking_at: LookingAtSampler::Exact(target),
                        position: ExtrinsicsSamplerType::Transform(Transform::from_translation(
                            target + Vec3::new(sin * review.distance, 0.08, cos * review.distance),
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
        frames: 3,
        timesteps: vec![],
        ..default()
    };
    batch.install = false;
}
fn main() -> Result<()> {
    let review = Review::parse();
    ensure!(
        review.distance.is_finite()
            && review.distance > 0.
            && review.yaw.is_finite()
            && review.rise.is_finite(),
        "invalid camera rig"
    );
    let size = review.size;
    let row_filter = review.row;
    let output = review.output.clone();
    let kinds: Vec<_> = KINDS
        .into_iter()
        .filter(|kind| {
            review
                .kind
                .as_ref()
                .is_none_or(|k| k == &format!("{kind:?}").to_ascii_lowercase())
        })
        .collect();
    let rig = json!({"distance_m":review.distance,"yaw_deg":review.yaw,"rise_m":review.rise,"ssaa":review.ssaa});
    fs::create_dir_all(&output)?;
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
            num_cameras: kinds.len(),
            ssaa_factor: review.ssaa,
            width: size as f32,
            height: size as f32,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Normal,
                RenderMode::Depth,
                RenderMode::Semantic,
            ],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.insert_resource(review);
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    for (p, lux) in [
        (Vec3::new(-3., 5., 5.), 1600.),
        (Vec3::new(2., 3., -4.), 800.),
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
    let mut records = Vec::new();
    for (row, (seed, coverage)) in [(41, 0.08), (41, 0.55), (41, 0.98), (101, 0.65)]
        .into_iter()
        .enumerate()
    {
        if row_filter.is_some_and(|selected| selected as usize != row) {
            continue;
        }
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.5, 0, 0.)
                .map_err(anyhow::Error::msg)?;
        scene.humans.clear();
        scene.objects.clear();
        let mut images = app.world_mut().remove_resource::<Assets<Image>>().unwrap();
        let mut materials = app
            .world_mut()
            .remove_resource::<Assets<StandardMaterial>>()
            .unwrap();
        let palette = IndoorMaterials::build(&scene, &mut images, &mut materials);
        app.world_mut().insert_resource(images);
        app.world_mut().insert_resource(materials);
        let mut assemblies = Vec::new();
        let first_record = records.len();
        for (i, &kind) in kinds.iter().enumerate() {
            let (_, treatment) = (seed..)
                .map(ShadeProgram::sample)
                .find(|(selected, _)| *selected == kind)
                .unwrap();
            let mut f = Facade {
                joinery: None,
                side: FacadeSide::Rear,
                openings: vec![WindowOpening {
                    glazing: None,
                    min: Vec2::new(-1.35, 0.30),
                    max: Vec2::new(1.35, 2.80),
                    columns: 2,
                    transom: 0.,
                    transoms: vec![],
                    operable: false,
                }],
                frame: Surface::Metal,
                frame_width: 0.045,
                frame_depth: 0.08,
                recess: 0.07,
                sill_projection: 0.075,
                treatment: Some(treatment),
                shade: kind,
                shade_coverage: coverage,
                shade_tilt: -0.65,
            };
            if row == 3 && kind == Shade::Curtain {
                let p = f.treatment.as_mut().unwrap();
                p.stack = facade::treatments::Stack::Pair;
                p.tieback = 0.85;
            }
            let mut a = Assembly::default();
            let tf = Transform::from_xyz(i as f32 * 8., 0., 0.);
            let before = Instant::now();
            facade::glazing_at(&mut a, tf, &f);
            let micros = before.elapsed().as_secs_f64() * 1e6;
            let triangles: usize = a.parts.values().map(|g| g.indices.len() / 3).sum();
            for (p, s) in [
                (Vec3::new(-1.62, 1.6, -0.06), Vec3::new(0.54, 3.2, 0.12)),
                (Vec3::new(1.62, 1.6, -0.06), Vec3::new(0.54, 3.2, 0.12)),
                (Vec3::new(0., 0.15, -0.06), Vec3::new(2.7, 0.30, 0.12)),
                (Vec3::new(0., 3.0, -0.06), Vec3::new(2.7, 0.4, 0.12)),
            ] {
                a.part(Surface::Paint, "wall").cuboid(
                    s,
                    0.,
                    tf.mul_transform(Transform::from_translation(p)),
                );
            }
            a.part(Surface::Floor, "floor").cuboid(
                Vec3::new(4., 0.1, 4.),
                0.,
                tf.mul_transform(Transform::from_xyz(0., -0.05, 1.)),
            );
            // Bounded exterior reference geometry gives visible depth through the opening.
            a.part(Surface::Concrete, "wall").cuboid(
                Vec3::new(6., 4., 0.1),
                0.,
                tf.mul_transform(Transform::from_xyz(0., 2., -3.5)),
            );
            for x in [-2., -1., 0., 1., 2.] {
                a.part(Surface::Metal, "window").cuboid(
                    Vec3::new(0.04, 3., 0.03),
                    0.,
                    tf.mul_transform(Transform::from_xyz(x, 1.6, -3.40)),
                );
            }
            records.push(json!({"row":row,"column":i,"seed":seed,"facade":f,"window_triangles":triangles,"window_construction_microseconds":micros}));
            assemblies.push(a);
        }
        app.world_mut().insert_resource(Batch {
            assemblies,
            materials: palette,
            install: true,
        });
        let started = Instant::now();
        loop {
            app.update();
            ensure!(
                started.elapsed() < Duration::from_secs(180),
                "window treatment capture timeout"
            );
            ensure!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "window treatment capture failed"
            );
            if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                ensure!(sample.views.len() == kinds.len(), "missing window views");
                for (i, v) in sample.views.iter().enumerate() {
                    let expected = if kinds[i] == Shade::Curtain {
                        SemanticLabel::Curtain
                    } else {
                        SemanticLabel::Blinds
                    };
                    let semantic = expected.color().to_linear();
                    let mut visible = 0;
                    for (mode, bytes) in [
                        ("color", &v.color),
                        ("normal", &v.normal),
                        ("depth", &v.depth),
                        ("semantic", &v.semantic),
                    ] {
                        ensure!(
                            bytes.len() == (size * size * 16) as usize,
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
                                    && (p[0] - semantic.red).abs() < 0.002
                                    && (p[1] - semantic.green).abs() < 0.002
                                    && (p[2] - semantic.blue).abs() < 0.002
                                {
                                    visible += 1;
                                }
                                [0, 1, 2].map(|j| {
                                    let value = match mode {
                                        "color" | "semantic" => {
                                            bevy_zeroverse::render::color::linear_to_srgb(p[j])
                                        }
                                        "depth" => p[0] / 12.,
                                        _ => p[j],
                                    };
                                    (value.clamp(0., 1.) * 255.).round() as u8
                                })
                            })
                            .collect();
                        image::RgbImage::from_raw(size, size, pixels)
                            .unwrap()
                            .save(output.join(format!("{row}-{:?}-{mode}.png", kinds[i])))?;
                    }
                    ensure!(
                        visible > 20,
                        "missing covering semantic pixels: row {row} {:?}",
                        kinds[i]
                    );
                    records[first_record + i]["visible_covering_pixels"] = json!(visible);
                }
                break;
            }
        }
        println!(
            "Covering review row {row}: {:.2}s",
            started.elapsed().as_secs_f64()
        );
    }
    fs::write(
        output.join("review.json"),
        serde_json::to_vec_pretty(
            &json!({"source":bevy_zeroverse::provenance::capture_provenance(),"image_size":[size,size],"rig":rig,"records":records}),
        )?,
    )?;
    Ok(())
}
