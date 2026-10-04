//! Small reproducible portrait/contact-sheet capture for human appearance review.
//! cargo run --example review_humans --features human_motion -- out/human_review
//! Add --standing after the output directory to review the same wardrobes upright.
//! --fit-review stages short, muscular bodies across all six garment cuts.
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler, PerspectiveSampler,
        TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    sample::SamplerState,
    scene::{
        procedural_indoor::{
            humans,
            layout::{IndoorLayout, IndoorManifest},
            materials::IndoorMaterials,
        },
        SceneAabbNode, ZeroverseSceneRoot, ZeroverseSceneType,
    },
};
use std::time::{Duration, Instant};

fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    setup_globals(None);
    let output =
        std::path::PathBuf::from(std::env::args().nth(1).unwrap_or("out/human_review".into()));
    std::fs::create_dir_all(&output).unwrap();
    let args: Vec<_> = std::env::args().collect();
    let first_seed: u64 = args
        .windows(2)
        .find(|p| p[0] == "--seed")
        .map_or(0, |p| p[1].parse().expect("--seed integer"));
    let hair_start: u8 = args
        .windows(2)
        .find(|p| p[0] == "--hair-start")
        .map_or(0, |p| p[1].parse().expect("--hair-start integer"));
    let hair_review = args.iter().any(|p| p == "--hair-start");
    let body_review = args.iter().any(|p| p == "--body-review");
    let fit_review = args.iter().any(|p| p == "--fit-review");
    std::fs::write(
        output.join("provenance.json"),
        serde_json::to_vec_pretty(&bevy_zeroverse::provenance::capture_provenance()).unwrap(),
    )
    .unwrap();
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.35, 0, 1.0).unwrap();
    scene.humans.clear();
    const PEOPLE: usize = 8;
    for seed in 0..PEOPLE as u64 {
        let mut source = IndoorManifest::generate_with_humans(
            first_seed
                + if body_review || fit_review || args.iter().any(|a| a == "--same-person") {
                    0
                } else {
                    seed
                },
            IndoorLayout::Mixed,
            0.35,
            0,
            1.0,
        )
        .unwrap();
        let mut h = source.humans.remove(0);
        if std::env::args().any(|a| a == "--standing") {
            h.pose = humans::HumanPoseKind::StandingRelaxed;
            let program = humans::poses::PoseProgram::sample(h.seed, h.pose);
            h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
            h.pose_program = Some(program);
        }
        h.position = Vec3::new(
            seed as f32 * if body_review || fit_review { 3.0 } else { 1.5 },
            0.0,
            0.0,
        );
        h.yaw = 0.0;
        h.glasses = seed % 2 == 0;
        h.hairstyle = (hair_start + seed as u8) % humans::hair::HairStyle::ALL.len() as u8;
        if args.iter().any(|a| a == "--female") {
            h.appearance.as_mut().unwrap().body_gender = Some(0.95);
        }
        if body_review {
            // Same wardrobe/pose/pigments and fixed full-body cameras: compare
            // dimensions without hiding differences through per-person zoom.
            h.stature = if seed & 1 == 0 { 1.50 } else { 1.95 };
            h.build = if seed & 2 == 0 { 0.82 } else { 1.22 };
            h.shoulder_width = humans::morphology::shoulder_span(h.stature, h.build, 0.37);
            let a = h.appearance.as_mut().unwrap();
            a.body_program = Some(humans::morphology::BodyProgram {
                gender: if seed & 4 == 0 { 0.05 } else { 0.95 },
                age: 0.70,
                muscle: 0.5,
                proportions: 0.5,
            });
            h.hairstyle = humans::hair::HairStyle::Bald as u8;
            h.glasses = false;
            h.outfit = humans::HumanOutfit::Tee;
            h.pose = humans::HumanPoseKind::StandingRelaxed;
            let program = humans::poses::PoseProgram::sample(h.seed, h.pose);
            h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
            h.pose_program = Some(program);
        }
        if fit_review {
            h.stature = 1.50;
            h.build = 1.22;
            h.shoulder_width = humans::morphology::shoulder_span(h.stature, h.build, 0.37);
            h.appearance.as_mut().unwrap().body_program = Some(humans::morphology::BodyProgram {
                gender: if seed < 6 { 0.05 } else { 0.95 },
                age: 0.70,
                muscle: 0.88,
                proportions: 0.5,
            });
            h.hairstyle = humans::hair::HairStyle::Bald as u8;
            h.glasses = false;
            h.outfit = [
                humans::HumanOutfit::Shirt,
                humans::HumanOutfit::Knitwear,
                humans::HumanOutfit::Blazer,
                humans::HumanOutfit::Tee,
                humans::HumanOutfit::Polo,
                humans::HumanOutfit::Cardigan,
            ][seed as usize % 6];
            h.pose = humans::HumanPoseKind::StandingRelaxed;
            let program = humans::poses::PoseProgram::sample(h.seed, h.pose);
            h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
            h.pose_program = Some(program);
        }
        h.id = seed as usize;
        scene.humans.push(h);
    }
    std::fs::write(
        output.join("people.json"),
        serde_json::to_vec_pretty(&scene.humans).unwrap(),
    )
    .unwrap();
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
            num_cameras: PEOPLE * 4,
            width: 640.0,
            height: 640.0,
            render_modes: vec![RenderMode::Color, RenderMode::Semantic],
            playback_steps: 1,
            ..default()
        }),
        false,
    );
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
            humans::spawn_people(
                &scene,
                root,
                &mut commands,
                &mut meshes,
                &mut materials,
                &set,
            );
            commands.spawn((
                DirectionalLight {
                    illuminance: 850.0,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-3.0, 6.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ));
            commands.spawn((
                DirectionalLight {
                    illuminance: 250.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(5.0, 2.0, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
            ));
            for (index, h) in scene.humans.iter().enumerate() {
                let dressed = humans::build_human(h);
                let hair_points: Vec<_> = dressed
                    .parts
                    .get(&humans::HumanSurface::Hair)
                    .into_iter()
                    .flat_map(|g| &g.positions)
                    .map(|p| Vec3::from_array(*p))
                    .collect();
                let (hair_lo, hair_hi) = if hair_points.is_empty() {
                    (
                        h.joints[4] - Vec3::splat(0.13),
                        h.joints[4] + Vec3::splat(0.13),
                    )
                } else {
                    hair_points.iter().copied().fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(lo, hi), p| (lo.min(p), hi.max(p)),
                    )
                };
                let hair_target = h.position + (hair_lo + hair_hi) * 0.5;
                let hair_distance =
                    (hair_hi - hair_lo).length() * 0.5 / 16_f32.to_radians().sin() * 1.12;
                for full_body in [false, true] {
                    let target = if full_body {
                        h.position
                            + if body_review || fit_review {
                                Vec3::Y
                            } else {
                                (h.joints[0] + h.joints[2]) * 0.5
                            }
                    } else {
                        h.position + h.joints[4] + Vec3::Y * 0.025
                    };
                    let mut from = target
                        + Vec3::new(
                            0.05,
                            0.025,
                            if full_body && (body_review || fit_review) {
                                -4.3
                            } else if full_body {
                                -3.7
                            } else {
                                -0.72
                            },
                        );
                    let target = if hair_review && !full_body {
                        from =
                            hair_target + Vec3::new(0.02, 0.015, -1.).normalize() * hair_distance;
                        hair_target
                    } else {
                        target
                    };
                    commands.spawn((
                        bevy_zeroverse::camera::CaptureCameraIndex(
                            index + if full_body { PEOPLE } else { 0 },
                        ),
                        ZeroverseCamera {
                            perspective_sampler: PerspectiveSampler::exact(32.0),
                            trajectory: TrajectorySampler::Static {
                                start: ExtrinsicsSampler {
                                    looking_at: LookingAtSampler::Exact(target),
                                    position: ExtrinsicsSamplerType::Transform(
                                        Transform::from_translation(from),
                                    ),
                                    ..default()
                                },
                            },
                            ..default()
                        },
                        set.environment.clone(),
                        ChildOf(root),
                    ));
                }
                let (lo, hi) = dressed
                    .parts
                    .iter()
                    .filter(|(s, _)| {
                        matches!(
                            s,
                            humans::HumanSurface::Shoes
                                | humans::HumanSurface::Sole
                                | humans::HumanSurface::ShoeDetail
                        )
                    })
                    .flat_map(|(_, g)| &g.positions)
                    .map(|p| Vec3::from_array(*p))
                    .fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(lo, hi), p| (lo.min(p), hi.max(p)),
                    );
                let target = h.position + (lo + hi) * 0.5;
                let radius = (hi - lo).length() * 0.5;
                let distance = radius / 21_f32.to_radians().sin() * 1.12;
                let from = target + Vec3::new(0.25, 0.22, -1.).normalize() * distance;
                commands.spawn((
                    bevy_zeroverse::camera::CaptureCameraIndex(PEOPLE * 2 + index),
                    ZeroverseCamera {
                        perspective_sampler: PerspectiveSampler::exact(42.0),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                looking_at: LookingAtSampler::Exact(target),
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(from),
                                ),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    set.environment.clone(),
                    ChildOf(root),
                ));
                {
                    let target = hair_target;
                    commands.spawn((
                        bevy_zeroverse::camera::CaptureCameraIndex(PEOPLE * 3 + index),
                        ZeroverseCamera {
                            perspective_sampler: PerspectiveSampler::exact(32.0),
                            trajectory: TrajectorySampler::Static {
                                start: ExtrinsicsSampler {
                                    looking_at: LookingAtSampler::Exact(target),
                                    position: ExtrinsicsSamplerType::Transform(
                                        Transform::from_translation(
                                            target
                                                + Vec3::new(0.5, 0.04, 0.87).normalize()
                                                    * hair_distance,
                                        ),
                                    ),
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
        },
    );
    app.finish();
    app.cleanup();
    app.insert_resource(SamplerState {
        enabled: true,
        regenerate_scene: false,
        render_modes: vec![RenderMode::Color, RenderMode::Semantic],
        warmup_frames: 20,
        frames: 3,
        timesteps: vec![],
        ..default()
    });
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            start.elapsed() < Duration::from_secs(180),
            "portrait capture timeout"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            assert_eq!(sample.object_obbs.len(), PEOPLE);
            for (i, view) in sample.views.iter().enumerate() {
                let values: &[f32] = bytemuck::cast_slice(&view.color);
                let rgb: Vec<u8> = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            let v = bevy_zeroverse::render::color::linear_to_srgb(*v);
                            (v.clamp(0.0, 1.0) * 255.0).round() as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(640, 640, rgb)
                    .unwrap()
                    .save(output.join(format!(
                        "{}_{index}.png",
                        if i < PEOPLE {
                            "portrait"
                        } else if i < PEOPLE * 2 {
                            "body"
                        } else if i < PEOPLE * 3 {
                            "footwear"
                        } else {
                            "rear"
                        },
                        index = if i < PEOPLE * 3 {
                            i % PEOPLE
                        } else {
                            i - PEOPLE * 3
                        }
                    )))
                    .unwrap();
                assert!(!view.semantic.is_empty());
            }
            println!(
                "saved {} portraits; {} person boxes",
                sample.views.len(),
                sample.object_obbs.len()
            );
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
