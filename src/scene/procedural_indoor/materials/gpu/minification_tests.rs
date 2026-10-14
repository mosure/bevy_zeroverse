//! Actual StandardMaterial/capture regression, including filtering and IBL.
use super::*;
use crate::{
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
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use bevy::camera::visibility::RenderLayers;
use std::time::{Duration, Instant};

#[derive(Resource)]
struct SwatchBakes(Requests);

#[test]
#[ignore = "requires native GPU; paired CPU/GPU PBR at three texture footprints"]
fn gpu_baked_fabric_matches_cpu_when_minified() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let output = std::path::Path::new("out/material_portability/pbr");
    std::fs::create_dir_all(output).unwrap();
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
            num_cameras: 18,
            width: 384.,
            height: 384.,
            render_modes: vec![RenderMode::Color],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
    app.add_systems(
        Startup,
        |mut commands: Commands,
         mut images: ResMut<Assets<Image>>,
         mut materials: ResMut<Assets<StandardMaterial>>,
         mut meshes: ResMut<Assets<Mesh>>| {
            let mut scene =
                IndoorManifest::generate_with_humans(202, IndoorLayout::Conference, 0.5, 0, 0.)
                    .unwrap();
            scene.program.as_mut().unwrap().materials[Surface::FabricAlt as usize]
                .textile
                .as_mut()
                .unwrap()
                .knit = true;
            let root = commands
                .spawn((
                    ZeroverseSceneRoot,
                    SceneAabbNode,
                    Transform::IDENTITY,
                    Visibility::default(),
                ))
                .id();
            for (backend, gpu) in [false, true].into_iter().enumerate() {
                let set = super::super::super::IndoorMaterials::build_with_selection(
                    &scene,
                    crate::scene::procedural_indoor::IndoorQuality::default(),
                    &mut *images,
                    &mut *materials,
                    None,
                    gpu,
                );
                for (surface_index, surface) in
                    [Surface::Fabric, Surface::FabricAlt, Surface::Leather]
                        .into_iter()
                        .enumerate()
                {
                    for (distance, scale) in [0.15, 1., 8.].into_iter().enumerate() {
                        let index = backend * 9 + surface_index * 3 + distance;
                        let layer = RenderLayers::layer(index + 2);
                        let mut g = Geometry::default();
                        g.cuboid(
                            Vec3::new(1., 0.65, 0.06) * scale,
                            0.02 * scale,
                            Transform::from_xyz(0., 0.4 * scale, 0.),
                        );
                        let radius = 0.15 * scale;
                        let mut sphere = Sphere::new(radius).mesh().uv(64, 32);
                        if let Some(bevy::mesh::VertexAttributeValues::Float32x2(uv)) =
                            sphere.attribute_mut(Mesh::ATTRIBUTE_UV_0)
                        {
                            for p in uv {
                                p[0] *= std::f32::consts::TAU * radius;
                                p[1] *= std::f32::consts::PI * radius;
                            }
                        }
                        sphere.generate_tangents().unwrap();
                        g.mesh(
                            sphere,
                            Transform::from_xyz(0.23 * scale, 0.18 * scale, 0.22 * scale),
                        );
                        commands.spawn((
                            Mesh3d(meshes.add(g.into_mesh())),
                            MeshMaterial3d(set.get(surface)),
                            SemanticLabel::Chair,
                            layer.clone(),
                            ChildOf(root),
                        ));
                        let target = Vec3::Y * 0.38 * scale;
                        commands.spawn((
                            CaptureCameraIndex(index),
                            ZeroverseCamera {
                                resolution: Some(UVec2::splat(384)),
                                perspective_sampler: PerspectiveSampler::exact(39.),
                                trajectory: TrajectorySampler::Static {
                                    start: ExtrinsicsSampler {
                                        position: ExtrinsicsSamplerType::Transform(
                                            Transform::from_translation(
                                                target + Vec3::new(0.2, 0.2, 1.8) * scale,
                                            ),
                                        ),
                                        looking_at: LookingAtSampler::Exact(target),
                                        ..default()
                                    },
                                },
                                ..default()
                            },
                            layer,
                            set.environment.clone(),
                            ChildOf(root),
                        ));
                    }
                }
                if gpu {
                    commands.insert_resource(SwatchBakes(set.gpu_bakes));
                }
            }
            for (from, lux) in [
                (Vec3::new(-4., 3., 2.), 2400.),
                (Vec3::new(2., 1., 4.), 350.),
            ] {
                commands.spawn((
                    DirectionalLight {
                        illuminance: lux,
                        // Swatches overlap in world space on distinct render
                        // layers. A shared shadow map would mix their blockers
                        // and change lighting across the texture footprints.
                        shadow_maps_enabled: false,
                        ..default()
                    },
                    Transform::from_translation(from).looking_at(Vec3::ZERO, Vec3::Y),
                    RenderLayers::from_layers(&(2..20).collect::<Vec<_>>()),
                ));
            }
        },
    );
    // This fixture uses a Custom scene. The room lifecycle clears its own bake
    // resource when leaving ProceduralIndoor, so install the fixture's requests
    // after that cleanup, before render extraction and readback readiness.
    app.add_systems(
        PostUpdate,
        |mut commands: Commands, bakes: Res<SwatchBakes>| {
            commands.insert_resource(bakes.0.clone());
        },
    );
    app.finish();
    app.cleanup();
    app.insert_resource(bevy::light::GlobalAmbientLight::NONE);
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color],
        warmup_frames: 25,
        frames: 3,
        timesteps: vec![],
        ..default()
    });
    let start = Instant::now();
    let sample = loop {
        app.update();
        assert!(
            start.elapsed() < Duration::from_secs(180),
            "PBR capture timed out"
        );
        assert!(
            app.world().resource::<CaptureFailure>().0.is_none(),
            "capture failed"
        );
        if let Ok(sample) = crate::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            break sample;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(sample.views.len(), 18);
    let mut measurements = Vec::new();
    for pair in 0..9 {
        let a: &[f32] = bytemuck::cast_slice(&sample.views[pair].color);
        let b: &[f32] = bytemuck::cast_slice(&sample.views[pair + 9].color);
        let mut error = 0.;
        let mut maximum = 0_f32;
        for (&a, &b) in a.iter().zip(b) {
            assert!(a.is_finite() && b.is_finite());
            error += f64::from(a - b).powi(2);
            maximum = maximum.max((a - b).abs());
        }
        let rmse = (error / a.len() as f64).sqrt();
        for (backend, data) in [("cpu", a), ("gpu", b)] {
            let rgb = data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    p[..3].iter().map(|v| {
                        (crate::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.).round()
                            as u8
                    })
                })
                .collect();
            image::RgbImage::from_raw(384, 384, rgb)
                .unwrap()
                .save(output.join(format!("{pair}-{backend}.png")))
                .unwrap();
        }
        measurements.push(serde_json::json!({"pair":pair,"rmse":rmse,"max":maximum}));
    }
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&measurements).unwrap(),
    )
    .unwrap();
    for measurement in measurements {
        assert!(
            measurement["rmse"].as_f64().unwrap() < 0.004
                && measurement["max"].as_f64().unwrap() < 0.20,
            "PBR mismatch: {measurement}"
        );
    }
}
