//! Actual production window/trim/convectors in three controlled cutaway bays.
//! This is a geometry/material inspection, not a random-room realism benchmark.
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json LP_NUM_THREADS=2
//! cargo run --example review_window_sections
use bevy::{
    app::{ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin},
    camera::RenderTarget,
    prelude::*,
    render::{
        render_resource::TextureFormat,
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_zeroverse::scene::procedural_indoor::{
    architecture::{
        self,
        facade::{reveal::RevealProgram, ExteriorProgram, FacadeSide, Shade},
    },
    layout::{IndoorLayout, IndoorManifest},
    materials::{IndoorMaterials, Surface},
    objects::{audit, build_object, ObjectKind},
};

#[derive(Resource, Default)]
struct Review {
    frames: u32,
    images: Vec<Handle<Image>>,
    captured: usize,
}

fn main() {
    std::fs::create_dir_all("out/window_trim/renders").unwrap();
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(bevy::render::RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .set(TaskPoolPlugin {
                    task_pool_options: TaskPoolOptions::with_num_threads(2),
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_millis(20),
        ))
        .init_resource::<Review>()
        .add_systems(Startup, setup)
        .add_systems(Update, capture)
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut review: ResMut<Review>,
) {
    let mut records = Vec::new();
    for (i, (thickness, height, depth)) in
        [(0.18, 0.04, 0.01), (0.40, 0.16, 0.035), (0.70, 0.33, 0.065)]
            .into_iter()
            .enumerate()
    {
        let seed = 3 + i as u64 * 7;
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.5, 0, 0.)
                .unwrap();
        scene.room_size = Vec3::new(2.8, 2.8, 2.);
        scene.envelope = None;
        scene.objects.clear();
        scene.humans.clear();
        let p = scene.program.as_mut().unwrap();
        p.partitions.clear();
        p.zones.clear();
        let trim = p.finishes.as_mut().unwrap().trim.as_mut().unwrap();
        trim.base = Some(architecture::trim::TrimSection {
            height_m: height,
            depth_m: depth,
            shoulder: 0.6,
            tip_fraction: 0.35,
            profile: [
                architecture::trim::TrimProfile::Flat,
                architecture::trim::TrimProfile::Stepped,
                architecture::trim::TrimProfile::Cove,
            ][i],
            surface: Surface::Paint,
            finish_slot: 0,
        });
        trim.crown = None;
        let mut facade = ExteriorProgram::sample(seed, scene.room_size, 0.2)
            .facades
            .remove(0);
        facade.side = FacadeSide::Rear;
        facade.shade = Shade::None;
        facade.treatment = None;
        facade.frame_width = 0.06;
        facade.frame_depth = 0.08;
        facade.openings.truncate(1);
        let opening = &mut facade.openings[0];
        opening.min = Vec2::new(-0.95, 0.8);
        opening.max = Vec2::new(0.95, 2.35);
        opening.columns = 2;
        opening.transom = 0.;
        opening.transoms.clear();
        opening.operable = false;
        opening.glazing = None;
        let (reveal, _) =
            RevealProgram::sample(seed, thickness, facade.frame_depth, facade.frame_width);
        facade.recess = 0.04 + 0.012 + (thickness - 0.104) * 0.72;
        reveal.validate(facade.frame_depth, facade.recess).unwrap();
        facade.joinery.as_mut().unwrap().reveal = Some(reveal);
        scene.exterior = Some(ExteriorProgram {
            exposure_probability: 1.,
            facades: vec![facade],
        });
        let set = IndoorMaterials::build(&scene, &mut *images, &mut *materials);
        let shift = Vec3::X * i as f32 * 8.;
        let root = commands
            .spawn((Transform::from_translation(shift), Visibility::default()))
            .id();
        let mut assembly = architecture::architecture(&scene);
        // Keep the rear wall and its actual opening/trim meshes; remove the
        // other walls so the wall cross section is inspectable from this view.
        assembly.parts.retain(|(_, label), _| {
            !label.contains("exterior")
                && matches!(label.split('#').next(), Some("wall" | "window"))
        });
        for g in assembly.parts.values_mut() {
            g.indices = g
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter(|t| t.iter().all(|&v| g.positions[v as usize][2] < -0.70))
                .flatten()
                .copied()
                .collect();
        }
        assembly.spawn(root, &mut commands, &mut meshes, &set);
        if i > 0 {
            let mut heater = audit::fixture(ObjectKind::BaseboardHeater, seed);
            heater.size = Vec3::new(
                if i == 1 { 1.15 } else { 2.0 },
                if i == 1 { 0.22 } else { 0.40 },
                if i == 1 { 0.11 } else { 0.18 },
            );
            let holder = commands
                .spawn((
                    Transform::from_xyz(0., 0., -1. + depth + 0.01 + heater.size.z * 0.5),
                    Visibility::default(),
                    ChildOf(root),
                ))
                .id();
            build_object(&heater).spawn(holder, &mut commands, &mut meshes, &set);
            records.push(
                serde_json::json!({"bay":i,"heater":heater,"program":audit::program(&heater)}),
            );
        }
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(4., 0.1, 3.))),
            MeshMaterial3d(set.for_part(Surface::Floor, "floor")),
            Transform::from_xyz(0., -0.06, 0.),
            ChildOf(root),
        ));
        // A plain distant card supplies an inspectable transmitted backdrop.
        let card = materials.add(StandardMaterial {
            base_color: Color::srgb(0.24, 0.38, 0.46),
            perceptual_roughness: 0.9,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(3.0, 3., 0.05))),
            MeshMaterial3d(card),
            Transform::from_xyz(0., 1.4, -2.1),
            ChildOf(root),
        ));
        let image = images.add(Image::new_target_texture(
            640,
            800,
            TextureFormat::Rgba8UnormSrgb,
            None,
        ));
        commands.spawn((
            Camera3d::default(),
            Camera {
                clear_color: Color::srgb_u8(28, 33, 42).into(),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            Projection::Perspective(PerspectiveProjection {
                fov: 48_f32.to_radians(),
                ..default()
            }),
            Transform::from_translation(shift + Vec3::new(1.6, 1.85, 2.75))
                .looking_at(shift + Vec3::new(0., 1.25, -1.), Vec3::Y),
            bevy::camera::Exposure { ev100: 9. },
            set.environment.clone(),
        ));
        review.images.push(image);
        records.push(serde_json::json!({"bay":i,"facade":scene.exterior,"trim":scene.program.as_ref().unwrap().finishes.as_ref().unwrap().trim}));
    }
    commands.spawn((
        DirectionalLight {
            illuminance: 3500.,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(2., 4., 5.).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(bevy::light::GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 80.,
        ..default()
    });
    std::fs::write("out/window_trim/renders/fixtures.json",serde_json::to_vec_pretty(&serde_json::json!({"kind":"controlled production mesh/PBR cutaways","build":bevy_zeroverse::provenance::capture_provenance(),"records":records})).unwrap()).unwrap();
}

fn capture(mut commands: Commands, mut review: ResMut<Review>, mut exit: MessageWriter<AppExit>) {
    review.frames += 1;
    if review.frames == 50 {
        for (i, image) in review.images.iter().enumerate() {
            commands
                .spawn(Screenshot::image(image.clone()))
                .observe(save_to_disk(format!("out/window_trim/renders/bay-{i}.png")))
                .observe(
                    |capture: On<ScreenshotCaptured>,
                     mut review: ResMut<Review>,
                     mut exit: MessageWriter<AppExit>| {
                        let pixels = capture.image.data.as_ref().expect("readback pixels");
                        let colors: std::collections::BTreeSet<_> = pixels
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .step_by(67)
                            .map(|p| [p[0], p[1], p[2]])
                            .collect();
                        assert!(colors.len() > 32, "empty or unready cutaway capture");
                        review.captured += 1;
                        if review.captured == review.images.len() {
                            exit.write(AppExit::Success);
                        }
                    },
                );
        }
    }
    if review.frames > 160 {
        exit.write(AppExit::error());
    }
}
