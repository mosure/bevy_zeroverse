#![recursion_limit = "256"]
#[path = "../tests/support/fisheye.rs"]
mod analytic;
#[path = "../tests/support/ceiling_capture.rs"]
mod ceiling_capture;
#[path = "../tests/support/fisheye_lighting.rs"]
mod lighting;
// Bounded analytic and real-scene qualification. No dataset filtering or
// quality settings are changed by the benchmark. Run with --cpu for math/WGSL.
#[path = "../tests/support/capture_deadline.rs"]
mod deadline;
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::lens::LensSampling,
    headless::{create_app, setup_globals, update_capture},
    io::channels,
    render::{depth::DepthFormat, RenderMode},
    sample::{CaptureFailure, SamplerState},
    scene::{
        procedural_indoor::{
            layout::{IndoorLayout, IndoorManifest},
            reset_indoor_sequence,
        },
        RegenerateSceneEvent, ZeroverseSceneType,
    },
};
use std::{path::PathBuf, time::Instant};

fn shader_checks() {
    lighting::validate_shader();
    let lens = include_str!("../src/camera/lens/lens.wgsl");
    let bvh = include_str!("../src/render/ground_truth/angular/bvh.wgsl");
    for (name, source, tree) in [
        (
            "rgb",
            include_str!("../src/camera/lens/resolve.wgsl"),
            false,
        ),
        (
            "geometry",
            include_str!("../src/render/ground_truth/angular/trace.wgsl"),
            true,
        ),
        (
            "flow",
            include_str!("../src/render/ground_truth/angular/flow.wgsl"),
            true,
        ),
        (
            "visibility",
            include_str!("../src/render/co_visibility/visibility.wgsl"),
            false,
        ),
    ] {
        let joined = [lens, if tree { bvh } else { "" }, source].join("\n");
        let source = joined
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}
fn main() {
    shader_checks();
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--lighting-fixture") {
        lighting::run();
        return;
    }
    if args.iter().any(|a| a == "--fixture") {
        analytic::run();
        return;
    }
    let option = |name: &str| {
        args.iter()
            .find_map(|a| a.strip_prefix(&format!("--{name}=")).map(str::to_string))
    };
    let pinhole = args.iter().any(|a| a == "--pinhole");
    let matched = args.iter().any(|a| a == "--matched");
    let rgb_only = args.iter().any(|a| a == "--rgb-only");
    let mixed = args.iter().any(|a| a == "--mixed");
    let ceiling = args.iter().any(|a| a == "--ceiling");
    let custom_camera = option("camera-policy").map(|s| {
        bevy_zeroverse::scene::procedural_indoor::cameras::CameraSettings::parse(&s).unwrap()
    });
    let output = PathBuf::from(option("output").unwrap_or_else(|| "out/fisheye".into()));
    std::fs::create_dir_all(&output).unwrap();
    let prior = if ceiling {
        bevy_zeroverse::scene::procedural_indoor::cameras::ceiling::CeilingCameraSettings::default()
            .lenses
    } else {
        LensSampling {
            probability: if mixed { 0.5 } else { 1. },
            ..default()
        }
    };
    let mut distribution = Vec::new();
    for seed in 400..408 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut settings = scene.camera_settings.clone();
        settings.lenses = Some(prior.clone());
        settings.motion_fraction = 0.;
        if ceiling {
            settings.ceiling_mount = Some(default());
        }
        if let Some(custom) = &custom_camera {
            settings = custom.clone();
        }
        let started = Instant::now();
        scene.resample_cameras(3, settings.clone(), 1.).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        bevy_zeroverse::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        for (i, c) in scene.cameras.iter().enumerate() {
            let lens = if settings.ceiling_mounted(seed, i) {
                Some(&settings.ceiling_mount.as_ref().unwrap().lenses)
            } else {
                settings.lenses.as_ref()
            };
            assert_eq!(c.fisheye, lens.and_then(|l| l.sample(seed, i)));
        }
        distribution.push(serde_json::json!({"seed":seed,"seconds":seconds,"cameras":scene.cameras,"diagnostics":scene.camera_diagnostics}));
    }
    std::fs::write(
        output.join("sampling.json"),
        serde_json::to_vec_pretty(&distribution).unwrap(),
    )
    .unwrap();
    if args.iter().any(|a| a == "--cpu") {
        use bevy_zeroverse::scene::procedural_indoor::cameras::CameraSettings;
        for invalid in [
            r#"{"lenses":{"probability":1.1}}"#,
            r#"{"lenses":{"fov_degrees":[220,140]}}"#,
            r#"{"lenses":{"anisotropy":[0.1,1.0]}}"#,
            r#"{"lenses":{"circular_probability":-1}}"#,
        ] {
            assert!(CameraSettings::parse(invalid).is_err());
        }
        let settings =
            CameraSettings::parse(r#"{"lenses":{"probability":0.5},"motion_fraction":0}"#).unwrap();
        let restored = CameraSettings::parse(&serde_json::to_string(&settings).unwrap()).unwrap();
        assert_eq!(settings.lenses, restored.lenses);
        assert_eq!(restored.motion_fraction, 0.);
        let mixed = LensSampling::default();
        let mut fish = 0usize;
        let mut circular = 0usize;
        let mut total = 0usize;
        for seed in 0..1024 {
            for i in 0..12 {
                total += 1;
                if let Some(l) = mixed.sample(seed, i) {
                    fish += 1;
                    circular +=
                        usize::from(l.fit == bevy_zeroverse::calibration::FisheyeFit::Circular);
                    assert_eq!(Some(l), mixed.sample(seed, i));
                    l.calibration([640, 400], [0.12, -0.08]).unwrap();
                }
            }
        }
        assert!((0.48..0.52).contains(&(fish as f64 / total as f64)));
        assert!((0.17..0.23).contains(&(circular as f64 / fish as f64)));
        std::fs::write(
            output.join("sampling-audit.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "provenance":bevy_zeroverse::provenance::capture_provenance(),
                "prior":mixed,"seed_start":0,"seeds":1024,"cameras_per_seed":12,
                "assignments":total,"angular":fish,"circular":circular,
                "replay_exact":true,"calibrations_valid":true,
                "scope":"unconditioned lens prior; pose admission preserves selected models"
            }))
            .unwrap(),
        )
        .unwrap();
        println!("Lens prior: {fish}/{total} angular, {circular}/{fish} circular; seed/index assignments repeat exactly");
        println!("Portable shaders and eight 3-camera sampling populations passed");
        return;
    }
    setup_globals(Some(env!("CARGO_MANIFEST_DIR").into()));
    let size: usize = option("size").map(|v| v.parse().unwrap()).unwrap_or(
        if args.iter().any(|a| a == "--bench") {
            512
        } else {
            192
        },
    );
    let rooms: u64 = option("rooms").map(|v| v.parse().unwrap()).unwrap_or(
        if args.iter().any(|a| a == "--bench") {
            8
        } else {
            2
        },
    );
    let steps: u32 = option("steps").map(|v| v.parse().unwrap()).unwrap_or(2);
    let seed_start: u64 = option("seed").map(|v| v.parse().unwrap()).unwrap_or(400);
    let warmup: u64 = option("warmup").map(|v| v.parse().unwrap()).unwrap_or(0);
    let mut camera_policy = bevy_zeroverse::scene::procedural_indoor::cameras::CameraSettings {
        lenses: if pinhole && !matched {
            None
        } else {
            Some(prior.clone())
        },
        ..default()
    };
    if ceiling {
        let mut mount = bevy_zeroverse::scene::procedural_indoor::cameras::ceiling::CeilingCameraSettings::default();
        if pinhole && !matched {
            mount.lenses.probability = 0.;
        }
        camera_policy.ceiling_mount = Some(mount);
    }
    if let Some(custom) = &custom_camera {
        camera_policy = custom.clone();
    }
    let config = BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        headless: true,
        editor: false,
        gizmos: false,
        keybinds: false,
        press_esc_close: false,
        image_copiers: true,
        initialize_scene: false,
        num_cameras: option("cameras").map(|v| v.parse().unwrap()).unwrap_or(3),
        width: size as f32,
        height: size as f32,
        indoor_seed: Some(seed_start),
        indoor_camera: Some(serde_json::to_string(&camera_policy).unwrap()),
        playback_steps: steps,
        playback_step: 0.1,
        depth_format: DepthFormat::Linear,
        render_modes: if rgb_only {
            vec![RenderMode::Color]
        } else {
            vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::Semantic,
                RenderMode::Position,
                RenderMode::OpticalFlow,
                RenderMode::CoVisibility,
            ]
        },
        ..default()
    };
    let mut app = create_app(None, Some(config.clone()), false);
    if args.iter().any(|a| a == "--diagnostics") {
        app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    }
    if pinhole && matched {
        // Both workloads sample/admit exactly the same trajectories with the
        // same lens prior. Only the capture projection changes for the control.
        // This benchmark hook is not enabled in any production API.
        app.add_systems(
            Update,
            matched_pinhole
                .after(bevy_zeroverse::camera::CameraBuildSystems::Logical)
                .before(bevy_zeroverse::camera::CameraBuildSystems::Lens),
        );
    }
    app.finish();
    app.cleanup();
    let capture_timeout = deadline::for_adapter(
        app.world()
            .get_resource::<bevy::render::renderer::RenderAdapterInfo>()
            .map(|a| a.device_type),
        120,
    );
    let reference_settling = args.iter().any(|a| a == "--reference-settling");
    if reference_settling {
        // A full-target explicit viewport renders the same pixels but opts out
        // of the specialized no-viewport capture fence. Preserve headless light
        // origins, quality and activity gating; only settling becomes conservative.
        app.add_systems(
            PostUpdate,
            full_target_viewports
                .after(bevy::camera::CameraUpdateSystems)
                .before(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
        );
    }
    let mut timings = Vec::new();
    for (iteration, seed) in (seed_start..seed_start + warmup)
        .chain(seed_start..seed_start + rooms)
        .enumerate()
    {
        let start = Instant::now();
        let first_frame = app.world().resource::<bevy::diagnostic::FrameCount>().0;
        reset_indoor_sequence(app.world_mut(), seed);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
        let mut sampler = SamplerState::from_config(&config);
        sampler.regenerate_scene = false;
        app.insert_resource(sampler);
        let mut next_progress = std::time::Duration::from_secs(30);
        let sample = loop {
            update_capture(&mut app);
            assert!(
                app.world().resource::<CaptureFailure>().0.is_none(),
                "{:?}",
                app.world().resource::<CaptureFailure>().0
            );
            if let Ok(s) = channels::sample_receiver()
                .unwrap()
                .lock()
                .unwrap()
                .try_recv()
            {
                break s;
            }
            let elapsed = start.elapsed();
            if elapsed >= next_progress || elapsed >= capture_timeout {
                let diagnostic = serde_json::json!({
                    "seed":seed, "elapsed_seconds":elapsed.as_secs_f64(),
                    "deadline_seconds":capture_timeout.as_secs(),
                    "frames":app.world().resource::<bevy::diagnostic::FrameCount>().0 - first_frame,
                    "readiness":app.world().get_resource::<bevy_zeroverse::sample::CaptureReadiness>(),
                    "sampler":format!("{:?}", app.world().resource::<SamplerState>()),
                });
                eprintln!("fisheye capture progress: {diagnostic}");
                std::fs::write(
                    output.join("capture-progress.json"),
                    serde_json::to_vec_pretty(&diagnostic).unwrap(),
                )
                .unwrap();
                next_progress = elapsed + std::time::Duration::from_secs(30);
            }
            assert!(
                elapsed < capture_timeout,
                "fisheye capture exceeded {capture_timeout:?}; see capture-progress.json"
            );
        };
        let elapsed = start.elapsed().as_secs_f64();
        println!("seed {seed}: {elapsed:.3}s");
        if iteration < warmup as usize {
            continue;
        }
        let frames = app.world().resource::<bevy::diagnostic::FrameCount>().0 - first_frame;
        let settling_counts = [
            "capture/settling_fence/prepared_count",
            "capture/settling_fence/rendered_count",
            "capture/settling_fence/hit_count",
        ]
        .map(|path| {
            app.world()
                .get_resource::<bevy::diagnostic::DiagnosticsStore>()
                .and_then(|d| d.get(&bevy::diagnostic::DiagnosticPath::const_new(path)))
                .and_then(|d| d.value())
        });
        if reference_settling {
            assert!(
                settling_counts.iter().all(|v| v.is_none_or(|v| v == 0.)),
                "reference must never use the fast capture fence: {settling_counts:?}"
            );
        }
        let mut measurements = Vec::new();
        for (view_id, view) in sample.views.iter().enumerate() {
            if args.iter().any(|a| a == "--raw") {
                for (mode, bytes) in [
                    ("rgb", &view.color),
                    ("depth", &view.depth),
                    ("normal", &view.normal),
                    ("semantic", &view.semantic),
                    ("position", &view.position),
                    ("flow", &view.optical_flow),
                    ("co_visibility", &view.co_visibility),
                ] {
                    if !bytes.is_empty() {
                        std::fs::write(
                            output.join(format!("room_{seed}_view_{view_id}_{mode}.rgba32f")),
                            bytes,
                        )
                        .unwrap();
                    }
                }
            }
            let c = view.calibration.as_ref().unwrap();
            c.validate().unwrap();
            if custom_camera.is_none() && (!mixed || pinhole) {
                assert_eq!(c.fisheye.is_none(), pinhole);
            }
            let read = |bytes: &[u8]| {
                bytes
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|b| bytemuck::pod_read_unaligned::<[f32; 4]>(b))
                    .collect::<Vec<_>>()
            };
            let depth = read(&view.depth);
            let position = read(&view.position);
            let normal = read(&view.normal);
            let semantic = read(&view.semantic);
            if !rgb_only {
                assert_eq!(depth.len(), size * size);
                assert_eq!(normal.len(), depth.len());
                assert_eq!(semantic.len(), depth.len());
            }
            let mut max_error = 0_f32;
            let mut hits = 0;
            let mut normal_error = 0_f32;
            let world = Mat4::from_cols_array_2d(&view.world_from_view);
            let inv = world.inverse();
            for (i, p) in position.iter().enumerate() {
                if p[3] == 0. {
                    continue;
                }
                let n = Vec3::from_array(normal[i][..3].try_into().unwrap()) * 2. - Vec3::ONE;
                normal_error = normal_error.max((n.length() - 1.).abs());
                let lo = Vec3::from_array(sample.aabb[0]);
                let hi = Vec3::from_array(sample.aabb[1]);
                let point = Vec3::new(p[0], p[1], p[2]) * (hi - lo) + lo;
                let local = inv.transform_point3(point);
                let pixel = c.project(local.to_array()).unwrap();
                max_error = max_error.max(
                    Vec2::from_array(pixel)
                        .distance(Vec2::new((i % size) as f32 + 0.5, (i / size) as f32 + 0.5)),
                );
                assert!((c.depth(local.to_array()) - depth[i][0]).abs() < 0.003);
                hits += 1;
            }
            if !rgb_only {
                assert!(hits > size * size / 3, "too few valid pixels");
            }
            assert!(normal_error < 1e-5, "non-unit normal {normal_error}");
            assert!(max_error < 0.01, "self reprojection {max_error}");
            measurements.push(serde_json::json!({"view":view_id,"valid_pixels":hits,"self_reprojection_max_pixels":max_error,"normal_length_max_error":normal_error,"calibration":c,"world_from_view":view.world_from_view,"progress":view.trajectory_progress,"time_seconds":view.time_seconds}));
            let mut image = image::RgbImage::new(size as u32, size as u32);
            for (p, rgba) in image.pixels_mut().zip(read(&view.color)) {
                p.0 = std::array::from_fn(|i| {
                    (bevy_zeroverse::render::color::linear_to_srgb(rgba[i]).clamp(0., 1.) * 255.)
                        .round() as u8
                });
            }
            image
                .save(output.join(format!("room_{seed}_view_{view_id}.png")))
                .unwrap();
            for (mode, bytes) in [
                ("depth", &view.depth),
                ("normal", &view.normal),
                ("semantic", &view.semantic),
                ("position", &view.position),
                ("co_visibility", &view.co_visibility),
                ("flow", &view.optical_flow),
            ] {
                if bytes.is_empty() {
                    continue;
                }
                let mut preview = image::RgbImage::new(size as u32, size as u32);
                for (p, v) in preview.pixels_mut().zip(read(bytes)) {
                    p.0 = if mode == "co_visibility" {
                        bevy_zeroverse_capture::mask_color(v[0] as u16, config.num_cameras)
                    } else if mode == "depth" {
                        if v[3] > 0. {
                            bevy_zeroverse_capture::depth::turbo_depth_rgb(v[0], 15.).map(|x| {
                                (bevy_zeroverse::render::color::linear_to_srgb(x).clamp(0., 1.)
                                    * 255.)
                                    .round() as u8
                            })
                        } else {
                            [0; 3]
                        }
                    } else if mode == "semantic" {
                        std::array::from_fn(|j| {
                            (bevy_zeroverse::render::color::linear_to_srgb(v[j]).clamp(0., 1.)
                                * 255.)
                                .round() as u8
                        })
                    } else if mode == "flow" {
                        if v[2] > 0. {
                            [
                                (127.5 + v[0] * 4.).clamp(0., 255.) as u8,
                                (127.5 + v[1] * 4.).clamp(0., 255.) as u8,
                                128,
                            ]
                        } else {
                            [0; 3]
                        }
                    } else {
                        std::array::from_fn(|j| (v[j].clamp(0., 1.) * 255.).round() as u8)
                    };
                }
                preview
                    .save(output.join(format!("room_{seed}_view_{view_id}_{mode}.png")))
                    .unwrap();
            }
            std::fs::write(
                output.join(format!("room_{seed}_view_{view_id}.json")),
                serde_json::to_vec_pretty(&serde_json::json!({"calibration":c,"world_from_view":view.world_from_view,"near":view.near,"far":view.far,"progress":view.trajectory_progress,"time_seconds":view.time_seconds,"aabb":sample.aabb})).unwrap(),
            )
            .unwrap();
        }
        std::fs::write(
            output.join(format!("room_{seed}_manifest.json")),
            serde_json::to_vec_pretty(&sample.indoor).unwrap(),
        )
        .unwrap();
        let fixed_timeline = (ceiling && custom_camera.is_none())
            .then(|| ceiling_capture::fixed_timeline(&sample.views, config.num_cameras));
        let mixed_timeline = custom_camera.as_ref().map(|_| {
            ceiling_capture::mixed_timeline(&sample.views, sample.indoor.as_ref().unwrap())
        });
        let ceiling_visibility =
            if (ceiling || custom_camera.is_some()) && !rgb_only && config.num_cameras > 1 {
                Some(ceiling_capture::visibility(
                    &sample.views,
                    config.num_cameras,
                    sample.aabb,
                ))
            } else {
                None
            };
        let mixed_flow = (custom_camera.is_some() && !rgb_only && steps > 1)
            .then(|| ceiling_capture::camera_flow(&sample.views, config.num_cameras, sample.aabb));
        let geometry = app
            .world()
            .resource::<bevy_zeroverse::render::ground_truth::GroundTruthDiagnostics>()
            .snapshot();
        timings.push(serde_json::json!({"seed":seed,"seconds":elapsed,"frames":frames,"settling_counts":settling_counts,"views":sample.views.len(),"measurements":measurements,"ceiling_visibility_oracle":ceiling_visibility,"fixed_timeline":fixed_timeline,"mixed_timeline":mixed_timeline,"mixed_flow":mixed_flow,"geometry_work":geometry,"metadata":sample.indoor_render_metadata,"overlap":bevy_zeroverse::sample::qualification::rendered_overlap(&sample.views,config.num_cameras,sample.aabb)}));
    }
    std::fs::write(output.join("protocol.json"),serde_json::to_vec_pretty(&serde_json::json!({"config":config,"provenance":bevy_zeroverse::provenance::capture_provenance(),"build":{"debug_assertions":cfg!(debug_assertions),"target_arch":std::env::consts::ARCH,"profile_opt_level_override":std::env::var("CARGO_PROFILE_DEV_OPT_LEVEL").ok()},"adapter":app.world().get_resource::<bevy::render::renderer::RenderAdapterInfo>().map(|a|format!("{:?}",**a)),"seeds":[seed_start,seed_start+rooms],"warmup_rooms":warmup,"capture_deadline_seconds":capture_timeout.as_secs(),"reference_settling":reference_settling,"concurrent_training":!args.iter().any(|a|a=="--isolated"),"timing":"completed multi-view capture excluding application startup; first room includes shader warmup; retained without filtering"})).unwrap()).unwrap();
    std::fs::write(
        output.join("timings.json"),
        serde_json::to_vec_pretty(&timings).unwrap(),
    )
    .unwrap();
}

fn matched_pinhole(
    mut cameras: Query<&mut bevy_zeroverse::camera::ZeroverseCamera, Added<Camera>>,
    mut scene: Option<ResMut<IndoorManifest>>,
) {
    let mut changed = false;
    for mut camera in &mut cameras {
        changed |= camera.fisheye.take().is_some();
    }
    if changed {
        if let Some(scene) = scene.as_mut() {
            for camera in &mut scene.cameras {
                camera.fisheye = None;
            }
        }
    }
}

fn full_target_viewports(
    images: Res<Assets<Image>>,
    mut cameras: Query<
        (&mut Camera, &bevy::camera::RenderTarget),
        With<bevy_zeroverse::camera::ZeroverseCamera>,
    >,
) {
    for (mut camera, target) in &mut cameras {
        if let bevy::camera::RenderTarget::Image(target) = target {
            if let Some(image) = images.get(&target.handle) {
                camera.viewport = Some(bevy::camera::Viewport {
                    physical_size: image.size(),
                    ..default()
                });
            }
        }
    }
}
