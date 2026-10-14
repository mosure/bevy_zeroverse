//! Local RGB + semantic + pose qualification. No model inference is initialized.
//! cargo run --example review_static_poses -- out/static_pose_review
use bevy::prelude::*;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{
        CaptureCameraIndex, ExtrinsicsSampler, ExtrinsicsSamplerType, LookingAtSampler,
        PerspectiveSampler, TrajectorySampler, ZeroverseCamera,
    },
    headless::{create_app, setup_globals},
    render::RenderMode,
    sample::SamplerState,
    scene::{
        procedural_indoor::{
            humans::{self, poses::PoseProgram, HumanPoseKind},
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
    let output = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("out/static_pose_review".into()),
    );
    std::fs::create_dir_all(&output).unwrap();
    if !std::env::args().any(|arg| arg == "--skip-audit") {
        audit(&output);
    }
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.35, 0, 1.).unwrap();
    scene.humans.clear();
    const PEOPLE: usize = 12;
    for seed in 0..PEOPLE as u64 {
        let mut source =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 0, 1.).unwrap();
        let mut h = source.humans.remove(0);
        h.id = seed as usize;
        h.pose = [
            HumanPoseKind::StandingPresenting,
            HumanPoseKind::StandingConversation,
            HumanPoseKind::StandingRelaxed,
        ][seed as usize % 3];
        h.chair = None;
        h.worktop_contact = None;
        let program = PoseProgram::sample(h.seed, h.pose);
        h.joints = program.solve(h.stature, h.build, h.shoulder_width, false);
        h.pose_program = Some(program);
        h.position = Vec3::X * seed as f32 * 3.;
        h.yaw = 0.;
        scene.humans.push(h);
    }
    std::fs::write(
        output.join("people.json"),
        serde_json::to_vec_pretty(&scene.humans).unwrap(),
    )
    .unwrap();
    let expected_heads: Vec<_> = scene
        .humans
        .iter()
        .map(|h| humans::build_human(h).local_head_rotation)
        .collect();
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
            num_cameras: PEOPLE,
            width: 448.,
            height: 576.,
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
            for (index, human) in scene.humans.iter().enumerate() {
                let target = human.position + Vec3::Y * 0.95;
                commands.spawn((
                    CaptureCameraIndex(index),
                    ZeroverseCamera {
                        resolution: Some(UVec2::new(448, 576)),
                        perspective_sampler: PerspectiveSampler::exact(40.),
                        trajectory: TrajectorySampler::Static {
                            start: ExtrinsicsSampler {
                                position: ExtrinsicsSamplerType::Transform(
                                    Transform::from_translation(
                                        target + Vec3::new(0.3, 0.12, -3.2),
                                    ),
                                ),
                                looking_at: LookingAtSampler::Exact(target),
                                ..default()
                            },
                        },
                        ..default()
                    },
                    set.environment.clone(),
                    ChildOf(root),
                ));
            }
            commands.spawn((
                DirectionalLight {
                    illuminance: 850.,
                    shadow_maps_enabled: true,
                    ..default()
                },
                Transform::from_xyz(-3., 6., -5.).looking_at(Vec3::ZERO, Vec3::Y),
            ));
            commands.spawn((
                DirectionalLight {
                    illuminance: 250.,
                    ..default()
                },
                Transform::from_xyz(5., 2., -1.).looking_at(Vec3::ZERO, Vec3::Y),
            ));
        },
    );
    app.finish();
    app.cleanup();
    let mut sampler = SamplerState::from_config(app.world().resource::<BevyZeroverseConfig>());
    sampler.regenerate_scene = false;
    sampler.warmup_frames = 20;
    sampler.frames = 3;
    app.insert_resource(sampler);
    let start = Instant::now();
    loop {
        app.update();
        assert!(
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
                .is_none(),
            "capture failed: {:?}",
            app.world()
                .resource::<bevy_zeroverse::sample::CaptureFailure>()
                .0
        );
        assert!(
            start.elapsed() < Duration::from_secs(180),
            "pose capture timeout"
        );
        if let Ok(sample) = bevy_zeroverse::io::channels::sample_receiver()
            .unwrap()
            .lock()
            .unwrap()
            .try_recv()
        {
            assert_eq!(sample.views.len(), PEOPLE);
            assert_eq!(sample.human_poses, sample.human_pose_steps[0]);
            let mut annotations = Vec::new();
            for (index, view) in sample.views.iter().enumerate() {
                let annotation = sample.human_poses_for_view(index).unwrap();
                let person = &annotation.people[index];
                assert_eq!(person.instance_id, Some(index as i64));
                assert!(person.joints.iter().all(|p| p.in_frustum));
                assert!(
                    Quat::from_array(person.joints[4].world_rotation)
                        .angle_between(expected_heads[index])
                        < 0.001
                );
                assert!(sample
                    .object_obbs
                    .iter()
                    .any(|b| b.class_name == "person" && b.instance_id == person.instance_id));
                let rgba: &[f32] = bytemuck::cast_slice(&view.color);
                let rgb: Vec<_> = rgba
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|p| {
                        p[..3].iter().map(|v| {
                            (bevy_zeroverse::render::color::linear_to_srgb(*v).clamp(0., 1.) * 255.)
                                .round() as u8
                        })
                    })
                    .collect();
                image::RgbImage::from_raw(448, 576, rgb)
                    .unwrap()
                    .save(output.join(format!("pose_{index:02}.png")))
                    .unwrap();
                assert!(!view.semantic.is_empty());
                annotations.push(annotation);
            }
            std::fs::write(
                output.join("poses.json"),
                serde_json::to_vec_pretty(&annotations).unwrap(),
            )
            .unwrap();
            std::fs::write(
                output.join("provenance.json"),
                serde_json::to_vec_pretty(&bevy_zeroverse::provenance::capture_provenance())
                    .unwrap(),
            )
            .unwrap();
            println!("Validated {PEOPLE} rendered poses with calibrated joints, head rotations, and matching person OBB IDs in {:?}", start.elapsed());
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn audit(output: &std::path::Path) {
    use std::collections::{BTreeMap, HashSet};
    let start = Instant::now();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut unique = HashSet::new();
    let mut raised = [0usize; 2];
    let mut pointing = [0usize; 2];
    let mut turns = [0usize; 2];
    let mut palms = 0usize;
    let mut people = 0usize;
    let mut rejected = 0usize;
    for seed in 0..512 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.65).unwrap();
        bevy_zeroverse::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        rejected += scene.rejected_human_placements;
        for human in &scene.humans {
            people += 1;
            *counts.entry(format!("{:?}", human.pose)).or_default() += 1;
            let program = human.pose_program.as_ref().unwrap();
            unique.insert((
                program.arm_elevation.map(f32::to_bits),
                program.arm_sweep.map(f32::to_bits),
                program.torso_twist.to_bits(),
            ));
            for (i, base) in [5, 9].into_iter().enumerate() {
                raised[i] += usize::from(human.joints[base + 2].y > human.joints[base].y);
                pointing[i] += usize::from(program.index_extension[i] > 0.7);
            }
            turns[usize::from(program.torso_twist > 0.)] +=
                usize::from(program.torso_twist.abs() > 0.30);
            palms += usize::from(human.worktop_contact.is_some());
        }
    }
    assert_eq!(unique.len(), people);
    assert!(
        counts.len() == 8 && raised.iter().all(|&n| n > 20) && pointing.iter().all(|&n| n > 40)
    );
    std::fs::write(output.join("distribution.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "rooms":512, "first_seed":0, "human_density":0.65, "people":people, "activity_counts":counts,
        "unique_continuous_arm_and_torso_programs":unique.len(), "raised_wrists_left_right":raised,
        "extended_index_left_right":pointing, "torso_turns_beyond_0_3_rad_negative_positive":turns,
        "table_supported_people":palms, "rejected_placement_proposals":rejected,
        "manifest_and_layout_audit_seconds":start.elapsed().as_secs_f64(),
        "qualification":"layout/collision envelope audit; geometric mesh validation is covered by separate core tests"
    })).unwrap()).unwrap();
}
