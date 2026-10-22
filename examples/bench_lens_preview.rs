//! Bounded, matched moving-view benchmark. No image readback in timed frames.
//! Keeps the normal viewer quality; --pinhole changes only the sampled lens.
#![recursion_limit = "256"]
use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{lens::LensSampling, CameraBuildSystems, Playback, PlaybackMode, ZeroverseCamera},
    headless::{create_app, setup_globals},
    render::RenderMode,
    scene::{procedural_indoor::cameras::CameraSettings, ZeroverseSceneType},
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let args: Vec<_> = std::env::args().collect();
    let option = |name: &str, fallback: &str| {
        args.iter()
            .find_map(|v| v.strip_prefix(&format!("--{name}=")))
            .unwrap_or(fallback)
            .to_owned()
    };
    let frames: usize = option("frames", "60").parse().unwrap();
    assert!(frames >= 10);
    let size: f32 = option("size", "512").parse().unwrap();
    let pinhole = args.iter().any(|v| v == "--pinhole");
    let cluster_budget: Option<u32> = args
        .iter()
        .find_map(|v| v.strip_prefix("--clusters="))
        .map(|v| v.parse().unwrap());
    let output = PathBuf::from(option("output", "out/lens-preview"));
    std::fs::create_dir_all(&output).unwrap();
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        image_copiers: false,
        num_cameras: option("cameras", "3").parse().unwrap(),
        width: size,
        height: size,
        indoor_seed: Some(option("seed", "400").parse().unwrap()),
        indoor_human_density: 0.,
        playback_mode: PlaybackMode::Still,
        indoor_camera: Some(
            serde_json::to_string(&CameraSettings {
                lenses: Some(LensSampling {
                    probability: 1.,
                    ..default()
                }),
                ..default()
            })
            .unwrap(),
        ),
        ..default()
    };
    setup_globals(None);
    let mut app = create_app(None, Some(config.clone()), false);
    if args.iter().any(|v| v == "--captures") {
        app.add_plugins(bevy_zeroverse::io::image_copy::ImageCopyPlugin);
    }
    if let Some(total) = cluster_budget {
        assert!((1..=4096).contains(&total));
        use bevy::light::{cluster::ClusterConfig, SimulationLightSystems};
        app.add_systems(
            PostUpdate,
            (move |mut views: Query<&mut ClusterConfig, Added<Camera3d>>| {
                for mut config in &mut views {
                    *config = if total == 1 {
                        ClusterConfig::Single
                    } else {
                        ClusterConfig::FixedZ {
                            total,
                            z_slices: total.min(8),
                            z_config: default(),
                            dynamic_resizing: true,
                        }
                    };
                }
            })
            .before(SimulationLightSystems::AssignLightsToClusters),
        );
    }
    if pinhole {
        // Admission and trajectories are identical to the angular workload.
        app.add_systems(
            Update,
            (|mut cameras: Query<&mut ZeroverseCamera, Added<Camera>>| {
                for mut camera in &mut cameras {
                    camera.fisheye = None;
                }
            })
            .after(CameraBuildSystems::Logical)
            .before(CameraBuildSystems::Lens),
        );
    }
    app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    app.finish();
    app.cleanup();
    let device = app.world().resource::<RenderDevice>().clone();
    let mut rows = Vec::new();
    let modes = if args.iter().any(|v| v == "--all-modes") {
        vec![
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::Semantic,
            RenderMode::Position,
            RenderMode::CoVisibility,
            RenderMode::OpticalFlow,
            RenderMode::Color,
        ]
    } else {
        vec![RenderMode::Color, RenderMode::Depth]
    };
    for mode in modes {
        if args.iter().any(|v| v == "--rgb-only") && mode != RenderMode::Color {
            continue;
        }
        app.world_mut()
            .resource_mut::<BevyZeroverseConfig>()
            .render_mode = mode.clone();
        for moving in [false, true] {
            let mut times = Vec::new();
            let mut poses = Vec::new();
            for frame in 0..(frames + 80) {
                app.world_mut().resource_mut::<Playback>().progress = if moving {
                    (frame % frames) as f32 / frames as f32
                } else {
                    0.5
                };
                let start = Instant::now();
                app.update();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if frame == 79 {
                    let wait = Instant::now();
                    while !app.world().resource::<bevy_zeroverse::sample::CaptureReadiness>().scene_ready()
                        || app.world().get_resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>()
                            .is_some_and(|r| !r.ready())
                    {
                        assert!(
                            wait.elapsed().as_secs() < 60,
                            "preview pipelines did not become ready"
                        );
                        app.update();
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    }
                }
                assert!(app
                    .world()
                    .resource::<bevy_zeroverse::sample::CaptureFailure>()
                    .0
                    .is_none());
                if frame >= 80 {
                    times.push(start.elapsed().as_secs_f64() * 1000.);
                }
                if frame == 80 || frame == frames + 79 {
                    let mut cameras: Vec<_> = app
                        .world_mut()
                        .query::<(&bevy_zeroverse::camera::CaptureCameraIndex, &Transform)>()
                        .iter(app.world())
                        .map(|(i, t)| (i.0, t.to_matrix().to_cols_array()))
                        .collect();
                    cameras.sort_by_key(|(i, _)| *i);
                    poses.push(cameras.into_iter().map(|(_, p)| p).collect::<Vec<_>>());
                }
            }
            times.sort_by(f64::total_cmp);
            let diagnostics: BTreeMap<_, _> = app
                .world()
                .resource::<bevy::diagnostic::DiagnosticsStore>()
                .iter()
                .map(|d| {
                    (
                        d.path().to_string(),
                        serde_json::json!({"mean":d.average(),"unit":d.suffix}),
                    )
                })
                .collect();
            let views: Vec<_> = app
                .world_mut()
                .query::<(&Camera, &bevy::camera::RenderTarget)>()
                .iter(app.world())
                .map(|(c, t)| serde_json::json!({"active":c.is_active,"target":format!("{t:?}")}))
                .collect();
            let mean = times.iter().sum::<f64>() / frames as f64;
            let side_motion_passes = app
                .world_mut()
                .query_filtered::<Entity, (
                    With<bevy_zeroverse::camera::lens::LensFace>,
                    With<bevy::core_pipeline::prepass::MotionVectorPrepass>,
                )>()
                .iter(app.world())
                .count();
            assert_eq!(side_motion_passes, 0);
            println!(
                "{mode:?} moving={moving}: mean {mean:.2} ms, median {:.2} ms, p95 {:.2} ms",
                times[frames / 2],
                times[frames * 95 / 100]
            );
            rows.push(serde_json::json!({"mode":mode,"moving":moving,"frames":frames,"mean_ms":mean,
                "median_ms":times[frames/2],"p95_ms":times[frames*95/100],"times_ms":times,"diagnostics":diagnostics,"views":views,"poses":poses}));
            if moving && args.iter().any(|v| v == "--captures") {
                save_previews(&mut app, &output, &format!("{mode:?}"));
            }
            app.insert_resource(bevy::diagnostic::DiagnosticsStore::default());
        }
    }
    let sizes: Vec<_> = app
        .world()
        .resource::<Assets<Image>>()
        .iter()
        .filter_map(|(_, im)| {
            let s = im.size();
            (s.x > 256 && s.y > 256).then_some(s.to_array())
        })
        .collect();
    let sources: Vec<_> = app.world_mut().query::<&bevy_zeroverse::camera::lens::LensCamera>()
        .iter(app.world()).flat_map(|l| l.faces.iter().enumerate()
            .filter(|(i,h)| *i == 0 || **h != l.faces[0]).map(|(_,h)| h.clone()).collect::<Vec<_>>())
        .map(|h| { let im = app.world().resource::<Assets<Image>>().get(&h).unwrap();
            serde_json::json!({"size":im.size().to_array(),"format":format!("{:?}", im.texture_descriptor.format)})
        }).collect();
    std::fs::write(output.join("report.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "pinhole":pinhole,"config":config,"cluster_budget":cluster_budget,"provenance":bevy_zeroverse::provenance::capture_provenance(),
        "policy":"Same scene and admitted trajectories. 80 warmup updates, device-synchronized frame wall time, no readback, concurrent training. Quality unchanged. This is not window presentation FPS.",
        "adapter":format!("{:?}", **app.world().resource::<bevy::render::renderer::RenderAdapterInfo>()),"images":sizes,"source_faces":sources,"rows":rows
    })).unwrap()).unwrap();
}

fn save_previews(app: &mut App, output: &std::path::Path, mode: &str) {
    use bevy::{camera::RenderTarget, render::render_resource::TextureFormat};
    use bevy_zeroverse::{
        camera::lens::LensCamera, io::image_copy::ImageCopier, render::ssaa::SsaaCamera,
    };
    app.world_mut().resource_mut::<Playback>().progress = 0.5;
    let device = app.world().resource::<RenderDevice>().clone();
    for _ in 0..6 {
        app.update();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let mut targets: Vec<_> = app
        .world_mut()
        .query_filtered::<(
            &bevy_zeroverse::camera::CaptureCameraIndex,
            &RenderTarget,
            Option<&LensCamera>,
            Option<&SsaaCamera>,
        ), With<ZeroverseCamera>>()
        .iter(app.world())
        .map(|(index, target, lens, ssaa)| {
            let RenderTarget::Image(target) = target else {
                panic!("offscreen camera");
            };
            (
                index.0,
                ssaa.map(|s| s.output.clone())
                    .or_else(|| lens.map(|l| l.output.clone()))
                    .unwrap_or(target.handle.clone()),
            )
        })
        .collect();
    targets.sort_by_key(|(i, _)| *i);
    for (i, target) in targets {
        let descriptor = app
            .world()
            .resource::<Assets<Image>>()
            .get(&target)
            .unwrap()
            .texture_descriptor
            .clone();
        assert_eq!(descriptor.format, TextureFormat::Rgba32Float);
        let copier =
            ImageCopier::for_targets(vec![target], descriptor.size, descriptor.format, &device);
        let entity = app.world_mut().spawn(copier.clone()).id();
        copier.request(1);
        let start = Instant::now();
        let pixels = loop {
            app.update();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            assert!(copier.failure().is_none(), "{:?}", copier.failure());
            if let Some(capture) = copier.take(1) {
                break capture.planes.into_iter().next().unwrap();
            }
            assert!(start.elapsed().as_secs() < 20);
        };
        app.world_mut().despawn(entity);
        std::fs::write(output.join(format!("{mode}_{i}.linear-rgba32f")), &pixels).unwrap();
        let mut image = image::RgbaImage::new(descriptor.size.width, descriptor.size.height);
        for (pixel, bytes) in image.pixels_mut().zip(pixels.as_chunks::<16>().0) {
            let rgba: [f32; 4] = bytemuck::pod_read_unaligned(bytes);
            assert!(rgba.iter().all(|v| v.is_finite()));
            pixel.0 = std::array::from_fn(|c| {
                let x = rgba[c].clamp(0., 1.);
                let value = if c == 3 || x <= 0.0031308 {
                    if c == 3 {
                        x
                    } else {
                        12.92 * x
                    }
                } else {
                    1.055 * x.powf(1. / 2.4) - 0.055
                };
                (value * 255.).round() as u8
            });
        }
        image.save(output.join(format!("{mode}_{i}.png"))).unwrap();
    }
}
