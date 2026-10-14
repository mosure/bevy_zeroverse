//! Large-area floor/rug review using the production material resolver and PBR.
//! cargo run --example review_flooring -- out/flooring
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
            geometry::Geometry,
            layout::{IndoorLayout, IndoorManifest, ObjectKind},
            materials::{program, rugs, IndoorMaterials, Surface},
            objects::{self, audit},
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
const SIZE: u32 = 640;
const SEEDS: [u64; 4] = [1, 19, 53, 101];
#[derive(Component)]
struct ReviewRoot;
#[derive(Resource)]
struct Batch {
    floors: Vec<Handle<StandardMaterial>>,
    rug: Vec<(Geometry, Handle<StandardMaterial>)>,
    environment: EnvironmentMapLight,
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
    for i in 0..4 {
        let offset = Vec3::X * (i as f32 * 14.);
        let mut g = Geometry::default();
        g.cuboid(
            Vec3::new(6., 0.1, 6.),
            0.,
            Transform::from_xyz(0., -0.05, 0.),
        );
        commands.spawn((
            Mesh3d(meshes.add(g.into_mesh())),
            MeshMaterial3d(batch.floors[i].clone()),
            SemanticLabel::Floor,
            Transform::from_translation(offset),
            ChildOf(root),
        ));
        if i == 3 {
            for (g, m) in &batch.rug {
                commands.spawn((
                    Mesh3d(meshes.add(g.clone().into_mesh())),
                    MeshMaterial3d(m.clone()),
                    SemanticLabel::from_label("floormat").unwrap(),
                    Transform::from_translation(offset + Vec3::Y * 0.001),
                    ChildOf(root),
                ));
            }
        }
        let from = offset + Vec3::new(4.1, 6.8, 4.6);
        commands.spawn((
            ZeroverseCamera {
                resolution: Some(UVec2::splat(SIZE)),
                perspective_sampler: PerspectiveSampler::exact(48.),
                trajectory: TrajectorySampler::Static {
                    start: ExtrinsicsSampler {
                        looking_at: LookingAtSampler::Exact(offset),
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
fn main() -> Result<()> {
    let output = PathBuf::from(std::env::args().nth(1).unwrap_or("out/flooring".into()));
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
            num_cameras: 4,
            width: SIZE as f32,
            height: SIZE as f32,
            render_modes: vec![RenderMode::Color, RenderMode::Normal],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.add_systems(PreUpdate, install);
    app.finish();
    app.cleanup();
    for (p, lux) in [
        (Vec3::new(-3., 6., 5.), 1800.),
        (Vec3::new(5., 2., -1.), 300.),
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
    let mut environment = None;
    for seed in SEEDS {
        let start = Instant::now();
        let mut images = app.world_mut().remove_resource::<Assets<Image>>().unwrap();
        let mut materials = app
            .world_mut()
            .remove_resource::<Assets<StandardMaterial>>()
            .unwrap();
        let mut floors = Vec::new();
        let mut rug = Vec::new();
        for style in 0..3 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.5, 0, 0.)
                    .map_err(anyhow::Error::msg)?;
            scene.floor_style = style;
            scene.program.as_mut().unwrap().materials =
                program::sample_with_floor(scene.material_seed(), style)
                    .map_err(anyhow::Error::msg)?;
            let mut object = audit::fixture(ObjectKind::Rug, seed);
            object.finish_seed = Some(seed);
            object.size = Vec3::new(3.8, 0.012, 2.7);
            if style == 0 {
                scene.objects.push(object.clone());
            }
            let palette = IndoorMaterials::build(&scene, &mut images, &mut materials);
            environment.get_or_insert_with(|| palette.environment.clone());
            floors.push(palette.get(Surface::Floor));
            if style == 0 {
                rug = objects::build_object(&object)
                    .parts
                    .into_iter()
                    .map(|((s, l), g)| (g, palette.for_part(s, &l)))
                    .collect();
                let r = rugs::recipe(
                    &scene.program.as_ref().unwrap().materials[Surface::FabricAlt as usize],
                    rugs::slot(seed),
                );
                records.push(json!({"seed":seed,"category":"rug","recipe":r}));
            }
            records.push(
                json!({"seed":seed,"category":(["wood","carpet","stone"][style as usize]),
                "recipe":scene.program.as_ref().unwrap().materials[Surface::Floor as usize]}),
            );
        }
        floors.push(materials.add(StandardMaterial {
            base_color: Color::srgb(0.38, 0.39, 0.40),
            perceptual_roughness: 0.88,
            ..default()
        }));
        app.world_mut().insert_resource(images);
        app.world_mut().insert_resource(materials);
        app.world_mut().insert_resource(Batch {
            floors,
            rug,
            environment: environment.clone().unwrap(),
            install: true,
        });
        let capture = Instant::now();
        loop {
            app.update();
            ensure!(
                capture.elapsed() < Duration::from_secs(180),
                "floor review capture timeout"
            );
            if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                ensure!(sample.views.len() == 4, "incomplete floor review");
                for (name, v) in ["wood", "carpet", "stone", "rug"]
                    .into_iter()
                    .zip(&sample.views)
                {
                    for (mode, data) in [("color", &v.color), ("normal", &v.normal)] {
                        ensure!(data.len() == (SIZE * SIZE * 16) as usize, "short render");
                        let floats: &[f32] = bytemuck::cast_slice(data);
                        ensure!(floats.iter().all(|v| v.is_finite()), "nonfinite render");
                        let pixels = floats
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|p| {
                                p[..3].iter().map(|v| {
                                    let v = if mode == "color" {
                                        bevy_zeroverse::render::color::linear_to_srgb(*v)
                                    } else {
                                        *v
                                    };
                                    (v.clamp(0., 1.) * 255.).round() as u8
                                })
                            })
                            .collect();
                        image::RgbImage::from_raw(SIZE, SIZE, pixels)
                            .unwrap()
                            .save(output.join(format!("{name}-{seed}-{mode}.png")))?;
                    }
                }
                break;
            }
        }
        println!(
            "Floor review seed {seed}: {:.2}s",
            start.elapsed().as_secs_f64()
        );
    }
    fs::write(
        output.join("review.json"),
        serde_json::to_vec_pretty(&json!({
            "source":bevy_zeroverse::provenance::capture_provenance(), "seeds":SEEDS,
            "floor_extent_m":[6,6],"rug_extent_m":[3.8,2.7],"image_size":[SIZE,SIZE],
            "rig":"fixed 1800/300 lux, shared environment, production CPU reference maps and native PBR",
            "records":records
        }))?,
    )?;
    Ok(())
}
