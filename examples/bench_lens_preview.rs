//! Bounded, matched moving-view benchmark. No image readback in timed frames.
//! Keeps the normal viewer quality; --pinhole changes only the sampled lens.
#![recursion_limit = "256"]
use bevy::{prelude::*, render::renderer::RenderDevice};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        lens::{FisheyeFit, LensSampling},
        CameraBuildSystems, Playback, PlaybackMode, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    scene::{procedural_indoor::cameras::CameraSettings, ZeroverseSceneType},
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[path = "support/lens_preview_clusters.rs"]
mod clusters;
#[path = "support/lens_preview_profile.rs"]
mod profile;

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
    let direct = args.iter().any(|v| v == "--direct");
    let indirect = args.iter().any(|v| v == "--indirect");
    assert!(!(direct && indirect));
    let fit = match option("fit", "sampled").as_str() {
        "sampled" => None,
        "circular" => Some(FisheyeFit::Circular),
        "diagonal" => Some(FisheyeFit::Diagonal),
        "elliptical" => Some(FisheyeFit::Elliptical),
        _ => panic!("fit must be sampled, circular, diagonal or elliptical"),
    };
    let fov: Option<f32> = args
        .iter()
        .find_map(|v| v.strip_prefix("--fov="))
        .map(|v| v.parse().unwrap());
    assert!(fov.is_none_or(|v| (40.0..=220.).contains(&v)));
    let anisotropy: f32 = option("anisotropy", "1.15").parse().unwrap();
    assert!((0.8..=1.25).contains(&anisotropy));
    let motion = option("motion", "both");
    let movement: &[bool] = match motion.as_str() {
        "static" => &[false],
        "moving" => &[true],
        "both" => &[false, true],
        _ => panic!("motion must be static, moving or both"),
    };
    let warmup: usize = option("warmup", "80").parse().unwrap();
    assert!(warmup >= 20);
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
        indoor_human_density: option("human-density", "0").parse().unwrap(),
        playback_mode: PlaybackMode::Still,
        indoor_camera: Some(
            if let Some(path) = args.iter().find_map(|v| v.strip_prefix("--camera-config=")) {
                let json = std::fs::read_to_string(path).unwrap();
                CameraSettings::parse(&json).expect("valid benchmark camera policy");
                json
            } else {
                serde_json::to_string(&CameraSettings {
                    lenses: Some(LensSampling {
                        probability: 1.,
                        ..default()
                    }),
                    ..default()
                })
                .unwrap()
            },
        ),
        ..default()
    };
    setup_globals(None);
    let mut app = create_app(None, Some(config.clone()), false);
    // Wall-clock latency must not change the animation/flow visualization
    // inputs being compared. Timing below still uses the real Instant clock.
    app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_secs_f64(1. / 60.),
    ));
    if indirect {
        app.insert_resource(bevy_zeroverse::camera::CaptureDrawPolicy { indirect: true });
    }
    if direct {
        app.add_systems(
            Update,
            (|mut commands: Commands, cameras: Query<Entity, Added<ZeroverseCamera>>| {
                for entity in &cameras {
                    commands
                        .entity(entity)
                        .insert(bevy::render::view::NoIndirectDrawing);
                }
            })
            .after(CameraBuildSystems::Logical)
            .before(CameraBuildSystems::Lens),
        );
    }
    // Readiness includes uploads and material/pipeline preparation even when
    // the timed workload does not request readback. No copier is spawned here.
    app.add_plugins(bevy_zeroverse::io::image_copy::ImageCopyPlugin);
    if let Some(total) = cluster_budget {
        clusters::install(&mut app, total);
    }
    if pinhole || fit.is_some() || fov.is_some() {
        // Admission and trajectories are identical to the angular workload.
        app.add_systems(
            Update,
            (move |mut cameras: Query<&mut ZeroverseCamera, Added<Camera>>| {
                for mut camera in &mut cameras {
                    if pinhole {
                        camera.fisheye = None;
                    } else if let Some(sensor) = &mut camera.fisheye {
                        if let Some(fit) = fit {
                            sensor.fit = fit;
                            sensor.anisotropy = if fit == FisheyeFit::Circular {
                                1.
                            } else {
                                anisotropy
                            };
                        }
                        if let Some(fov) = fov {
                            sensor.lens.max_angle = 0.5 * fov.to_radians();
                        }
                    }
                }
            })
            .after(CameraBuildSystems::Logical)
            .before(CameraBuildSystems::Lens),
        );
    }
    let diagnostics = args.iter().any(|v| v == "--diagnostics");
    if diagnostics {
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    }
    let profile = args
        .iter()
        .any(|v| v == "--profile")
        .then(|| profile::install(&mut app));
    app.finish();
    profile::FrameFence::install(&mut app);
    let serial_views = args.iter().any(|v| v == "--serial-views");
    let serial_render = args.iter().any(|v| v == "--serial-render");
    if serial_views || serial_render {
        use bevy::{
            ecs::schedule::SingleThreadedExecutor,
            render::{Render, RenderApp},
        };
        let mut schedules = app
            .sub_app_mut(RenderApp)
            .world_mut()
            .resource_mut::<Schedules>();
        if serial_views {
            schedules
                .get_mut(bevy::core_pipeline::schedule::Core3d)
                .unwrap()
                .set_executor(SingleThreadedExecutor::new());
        }
        if serial_render {
            schedules
                .get_mut(Render)
                .unwrap()
                .set_executor(SingleThreadedExecutor::new());
        }
    }
    if let Some(profile) = &profile {
        profile::instrument_resources(&mut app, profile);
    }
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
        for &moving in movement {
            let mut times = Vec::new();
            let mut poses = Vec::new();
            for frame in 0..(frames + warmup) {
                app.world_mut().resource_mut::<Playback>().progress = if moving {
                    (frame % frames) as f32 / frames as f32
                } else {
                    0.5
                };
                let start = Instant::now();
                if let Some(profile) = &profile {
                    profile.record(frame >= warmup);
                }
                profile::FrameFence::update(&mut app);
                let update_ms = start.elapsed().as_secs_f64() * 1000.;
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if let Some(profile) = &profile {
                    profile.add("app.update", update_ms);
                    profile.add(
                        "device.poll",
                        start.elapsed().as_secs_f64() * 1000. - update_ms,
                    );
                }
                if frame == warmup - 1 {
                    let wait = Instant::now();
                    while !app.world().resource::<bevy_zeroverse::sample::CaptureReadiness>().scene_ready()
                        || app.world().get_resource::<bevy_zeroverse::io::image_copy::CapturePipelineReadiness>()
                            .is_some_and(|r| !r.ready())
                    {
                        assert!(
                            wait.elapsed().as_secs() < 60,
                            "preview pipelines did not become ready"
                        );
                        profile::FrameFence::update(&mut app);
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    }
                }
                assert!(app
                    .world()
                    .resource::<bevy_zeroverse::sample::CaptureFailure>()
                    .0
                    .is_none());
                if frame >= warmup {
                    times.push(start.elapsed().as_secs_f64() * 1000.);
                }
                if frame == warmup || frame == frames + warmup - 1 {
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
                .query::<(&Camera, &bevy::camera::RenderTarget, Has<bevy::render::view::NoIndirectDrawing>, Has<bevy::core_pipeline::prepass::MotionVectorPrepass>)>()
                .iter(app.world())
                .map(|(c, t, direct, motion)| serde_json::json!({"active":c.is_active,"target":format!("{t:?}"),"direct":direct,"motion_prepass":motion}))
                .collect();
            let shadow_batches = app
                .world_mut()
                .query::<&Name>()
                .iter(app.world())
                .filter(|n| n.as_str() == "procedural_shadow_batch")
                .count();
            let clustered_objects = app
                .world_mut()
                .query_filtered::<Entity, Or<(
                    With<PointLight>,
                    With<SpotLight>,
                    With<bevy::light::RectLight>,
                    With<bevy::light::LightProbe>,
                    With<bevy::light::ClusteredDecal>,
                )>>()
                .iter(app.world())
                .count();
            let mean = times.iter().sum::<f64>() / frames as f64;
            let point_shadows = app
                .world_mut()
                .query::<&PointLight>()
                .iter(app.world())
                .filter(|l| l.shadow_maps_enabled)
                .count();
            let spot_shadows = app
                .world_mut()
                .query::<&SpotLight>()
                .iter(app.world())
                .filter(|l| l.shadow_maps_enabled)
                .count();
            let directional_shadows = app
                .world_mut()
                .query::<&DirectionalLight>()
                .iter(app.world())
                .filter(|l| l.shadow_maps_enabled)
                .count();
            let side_motion_passes = app
                .world_mut()
                .query_filtered::<Entity, (
                    With<bevy_zeroverse::camera::lens::LensFace>,
                    With<bevy::core_pipeline::prepass::MotionVectorPrepass>,
                )>()
                .iter(app.world())
                .count();
            assert_eq!(side_motion_passes, 0);
            let angular_motion_passes = app
                .world_mut()
                .query_filtered::<Entity, (
                    With<bevy_zeroverse::camera::lens::LensCamera>,
                    With<bevy::core_pipeline::prepass::MotionVectorPrepass>,
                )>()
                .iter(app.world())
                .count();
            assert_eq!(angular_motion_passes, 0);
            println!(
                "{mode:?} moving={moving}: mean {mean:.2} ms, median {:.2} ms, p95 {:.2} ms",
                times[frames / 2],
                times[frames * 95 / 100]
            );
            rows.push(serde_json::json!({"mode":mode,"moving":moving,"frames":frames,"mean_ms":mean,
                "median_ms":times[frames/2],"p95_ms":times[frames*95/100],"times_ms":times,"diagnostics":diagnostics,"views":views,"poses":poses,
                "cpu_phases":profile.as_ref().map(|p| p.take()),"cluster_views":clusters::snapshot(&mut app),"shadow_batches":shadow_batches,"clustered_objects":clustered_objects,
                "shadow_maps_enabled":[point_shadows,spot_shadows,directional_shadows]}));
            // Preserve completed phases if a bounded diagnostic is interrupted.
            // This file deliberately does not masquerade as a complete report.
            std::fs::write(
                output.join("partial.json"),
                serde_json::to_vec_pretty(&rows).unwrap(),
            )
            .unwrap();
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
        "simulation_timestep_seconds":1./60.,"serial_views":serial_views,"serial_render":serial_render,"pinhole":pinhole,"fit":fit,"fov":fov,"anisotropy":anisotropy,"warmup_frames":warmup,"render_diagnostics":diagnostics,"direct":direct,"indirect":indirect,"config":config,"cluster_budget":cluster_budget,"provenance":bevy_zeroverse::provenance::capture_provenance(),
        "policy":"Same scene and admitted trajectories; only the render lens changes. Device-synchronized frame wall time, no readback. Quality unchanged. This is not window presentation FPS; record concurrent workloads separately.",
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
        profile::FrameFence::update(app);
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
            profile::FrameFence::update(app);
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
