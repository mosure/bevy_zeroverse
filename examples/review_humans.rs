//! Small reproducible portrait/contact-sheet capture for human appearance review.
//! cargo run --example review_humans --features human_motion -- out/human_review
//! Add --standing after the output directory to review the same wardrobes upright.
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
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.35, 0, 1.0).unwrap();
    scene.humans.clear();
    const PEOPLE: usize = 8;
    for seed in 0..PEOPLE as u64 {
        let mut source =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 0, 1.0).unwrap();
        let mut h = source.humans.remove(0);
        if std::env::args().any(|a| a == "--standing") {
            h.pose = humans::HumanPoseKind::StandingRelaxed;
            let program = humans::poses::PoseProgram::sample(h.seed, h.pose);
            h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
            h.pose_program = Some(program);
        }
        h.position = Vec3::new(seed as f32 * 1.5, 0.0, 0.0);
        h.yaw = 0.0;
        h.glasses = seed % 2 == 0;
        h.hairstyle = seed as u8;
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
            num_cameras: PEOPLE * 2 + 2,
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
                for full_body in [false, true] {
                    let target = if full_body {
                        h.position + (h.joints[0] + h.joints[2]) * 0.5
                    } else {
                        h.position + h.joints[4] + Vec3::Y * 0.025
                    };
                    let from =
                        target + Vec3::new(0.05, 0.025, if full_body { -2.0 } else { -0.72 });
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
                        ChildOf(root),
                    ));
                }
                if h.hairstyle >= 6 {
                    let target = h.position + h.joints[4] + Vec3::Y * 0.025;
                    commands.spawn((
                        bevy_zeroverse::camera::CaptureCameraIndex(PEOPLE * 2 + index - 6),
                        ZeroverseCamera {
                            perspective_sampler: PerspectiveSampler::exact(32.0),
                            trajectory: TrajectorySampler::Static {
                                start: ExtrinsicsSampler {
                                    looking_at: LookingAtSampler::Exact(target),
                                    position: ExtrinsicsSamplerType::Transform(
                                        Transform::from_translation(
                                            target + Vec3::new(0.45, 0.04, 0.65),
                                        ),
                                    ),
                                    ..default()
                                },
                            },
                            ..default()
                        },
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
                        } else {
                            "rear"
                        },
                        index = if i < PEOPLE * 2 {
                            i % PEOPLE
                        } else {
                            6 + i - PEOPLE * 2
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
