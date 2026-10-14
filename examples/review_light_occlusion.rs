//! Rendered direct-light control for production ceiling emitters after index 8.
//! A half-width opaque deck must darken one receiver while its peer stays lit.
use anyhow::{ensure, Result};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{architecture, layout::IndoorManifest, IndoorQuality},
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let output = PathBuf::from(std::env::args().nth(1).expect("output directory required"));
    ensure!(!output.exists(), "use a new output directory");
    std::fs::create_dir_all(&output)?;
    let scene = (0..64)
        .map(|seed| IndoorManifest::generate_with_humans(seed, Default::default(), 0.65, 0, 0.))
        .collect::<Result<Vec<_>, _>>()
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .find(|scene| architecture::fixture_positions(scene).len() > 16)
        .expect("fixture with more than sixteen lights");
    let seed = scene.seed;
    setup_globals(None);
    let modes = vec![RenderMode::Color];
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
            num_cameras: 2,
            playback_steps: 1,
            width: 128.,
            height: 128.,
            render_modes: modes.clone(),
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut meshes: ResMut<Assets<Mesh>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
            let root = commands
                .spawn((ZeroverseSceneRoot, SceneAabbNode, Transform::IDENTITY, Visibility::default()))
                .id();
            architecture::spawn_lights(&scene, IndoorQuality::Auto, root, &mut commands);
            commands.queue(|world: &mut World| {
                // Isolate an actual late-index emitter; keep production optical
                // and shadow settings. Moving it only defines the test fixture.
                let lights: Vec<_> = world
                    .query_filtered::<(Entity, &Name), Or<(With<SpotLight>, With<DirectionalLight>, With<PointLight>)>>()
                    .iter(world)
                    .map(|(e, name)| (e, name.to_string()))
                    .collect();
                for (e, name) in lights {
                    if name == "indoor_luminaire_12" {
                        world.entity_mut(e).insert(
                            Transform::from_xyz(0., 3., 0.).looking_at(Vec3::ZERO, Vec3::Z),
                        );
                    } else {
                        world.entity_mut(e).despawn();
                    }
                }
            });
            let white = materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.8, 0.8),
                perceptual_roughness: 1.,
                reflectance: 0.,
                ..default()
            });
            for (size, center) in [
                (Vec3::new(4., 0.1, 4.), Vec3::new(0., -0.05, 0.)),
                (Vec3::new(2.1, 0.1, 4.4), Vec3::new(-1.05, 1.4, 0.)),
            ] {
                commands.spawn((Mesh3d(meshes.add(Cuboid::from_size(size))), MeshMaterial3d(white.clone()), Transform::from_translation(center), ChildOf(root)));
            }
            for (index, x) in [-0.8, 0.8].into_iter().enumerate() {
                commands.spawn((
                    CaptureCameraIndex(index),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(30.),
                        trajectory: TrajectorySampler::Static { start: ExtrinsicsSampler {
                            position: ExtrinsicsSamplerType::Transform(Transform::from_xyz(x, 0.75, 0.1)),
                            looking_at: LookingAtSampler::Exact(Vec3::new(x, 0., 0.)),
                            ..default()
                        } },
                        ..default()
                    },
                    ChildOf(root),
                ));
            }
        },
    );
    app.add_systems(
        Last,
        |mut commands: Commands, cameras: Query<Entity, With<Camera3d>>| {
            for e in &cameras {
                commands
                    .entity(e)
                    .insert((
                        bevy::camera::Exposure { ev100: 8. },
                        bevy::core_pipeline::tonemapping::Tonemapping::None,
                        bevy::core_pipeline::tonemapping::DebandDither::Disabled,
                    ))
                    .remove::<(
                        bevy::post_process::bloom::Bloom,
                        bevy::anti_alias::fxaa::Fxaa,
                    )>();
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
        frames: 1,
        timesteps: Vec::new(),
        ..default()
    });
    let started = Instant::now();
    let sample = loop {
        app.update();
        ensure!(
            started.elapsed().as_secs() < 120,
            "occlusion capture timeout"
        );
        ensure!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "occlusion capture failed"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            break sample;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    ensure!(
        sample.views.len() == 2,
        "expected two receiver views, got {}",
        sample.views.len()
    );
    let mut means = Vec::new();
    for (index, view) in sample.views.iter().enumerate() {
        let rgba: &[f32] = bytemuck::cast_slice(&view.color);
        ensure!(
            rgba.iter().all(|x| x.is_finite()),
            "nonfinite receiver radiance"
        );
        let luminance: f64 = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| (p[0] * 0.2126 + p[1] * 0.7152 + p[2] * 0.0722) as f64)
            .sum::<f64>()
            / (128 * 128) as f64;
        means.push(luminance);
        std::fs::write(
            output.join(format!("receiver_{index}.rgba32f")),
            &view.color,
        )?;
    }
    let ratio = means[0] / means[1];
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "build":bevy_zeroverse::provenance::capture_provenance(), "source_seed":seed, "light":"indoor_luminaire_12", "ev100":8., "mean_exposed_linear_luminance":means,
            "blocked_over_clear":ratio, "passed":means[1] > 0.001 && ratio < 0.01,
            "scope":"Production late-index spot light; isolated direct receiver/deck geometry, no ambient, IBL or GI. This does not qualify indirect leakage or area-light equivalence."
        }))?,
    )?;
    ensure!(
        means[1] > 0.001 && ratio < 0.01,
        "occluded receiver failed: {means:?}"
    );
    Ok(())
}
