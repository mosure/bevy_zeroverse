//! Bounded actual-renderer check of editor glyphs; no generated rooms/models.
//! Supports software Vulkan: VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
//! cargo run --example review_camera_gizmos -- 12
use bevy::{
    app::ScheduleRunnerPlugin,
    camera::{visibility::RenderLayers, RenderTarget},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    prelude::*,
    render::{
        render_resource::TextureFormat,
        view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_zeroverse::{
    annotation::schematic::{Camera as PredictedCamera, Overlay},
    app::{editor::schematic::Predictions, BevyZeroverseConfig},
    camera::{
        gizmo,
        lens::{self, FisheyeFit, LensSampling},
        CaptureCameraIndex, EditorCameraGizmoConfigGroup, EditorCameraMarker, ZeroverseCamera,
    },
};

#[derive(Resource)]
struct Review {
    count: usize,
    predictions: bool,
    image: Handle<Image>,
    frames: u32,
}

fn main() {
    let count = std::env::args()
        .nth(1)
        .map_or(12, |s| s.parse::<usize>().unwrap())
        .clamp(1, 16);
    std::fs::create_dir_all("out/editor_camera_gizmos").unwrap();
    App::new()
        .add_plugins(
            DefaultPlugins
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
        .insert_gizmo_config(EditorCameraGizmoConfigGroup, gizmo::config(false))
        .insert_gizmo_config(gizmo::PredictionCameraGizmoConfigGroup, gizmo::config(true))
        .insert_resource(BevyZeroverseConfig {
            gizmos: true,
            gizmos_alpha: 1.,
            camera_grid: false,
            room_schematic: false,
            ..default()
        })
        .insert_resource(Review {
            count,
            predictions: std::env::args().any(|a| a == "--predictions"),
            image: default(),
            frames: 0,
        })
        .add_systems(Startup, setup)
        .add_systems(
            PostUpdate,
            gizmo::draw_camera_gizmo.after(bevy::transform::TransformSystems::Propagate),
        )
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
    review.image = images.add(Image::new_target_texture(
        1280,
        800,
        TextureFormat::Rgba8UnormSrgb,
        None,
    ));
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: Color::srgb_u8(20, 25, 32).into(),
            ..default()
        },
        RenderTarget::Image(review.image.clone().into()),
        EditorCameraMarker::default(),
        Transform::from_xyz(6., 7., 9.).looking_at(Vec3::new(0., 1., 0.), Vec3::Y),
        RenderLayers::from_layers(&[0, 1]),
        Tonemapping::None,
        DebandDither::Disabled,
    ));
    let floor = materials.add(StandardMaterial {
        base_color: Color::srgb_u8(42, 51, 60),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(7., 0.08, 7.))),
        MeshMaterial3d(floor),
        Transform::from_xyz(0., -0.05, 0.),
    ));
    let table = materials.add(StandardMaterial {
        base_color: Color::srgb_u8(107, 96, 78),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.8, 0.1, 1.2))),
        MeshMaterial3d(table.clone()),
        Transform::from_xyz(0., 0.8, 0.),
    ));
    for x in [-0.7, 0.7] {
        for z in [-0.4, 0.4] {
            commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.08, 0.75, 0.08))),
                MeshMaterial3d(table.clone()),
                Transform::from_xyz(x, 0.375, z),
            ));
        }
    }
    let mut predictions = Overlay::default();
    for id in 0..review.count {
        let phi = id as f32 * std::f32::consts::TAU / review.count as f32;
        let ceiling = id % 3 == 2;
        let mut pose = Transform::from_xyz(
            phi.cos() * 2.6,
            if ceiling { 2.8 } else { 1.25 },
            phi.sin() * 2.6,
        );
        pose = if ceiling {
            pose.looking_to(Vec3::NEG_Y, Vec3::Z)
        } else {
            pose.looking_at(Vec3::new(0., 1., 0.), Vec3::Y)
        };
        let projection = Projection::Perspective(PerspectiveProjection {
            fov: if id % 3 == 0 {
                140_f32.to_radians()
            } else {
                60_f32.to_radians()
            },
            aspect_ratio: 1.,
            ..default()
        });
        let fisheye = if id % 3 == 0 {
            None
        } else {
            let mut lens = LensSampling {
                probability: 1.,
                fov_degrees: [200., 200.],
                ..default()
            }
            .sample(401, id)
            .unwrap();
            lens.fit = if ceiling {
                FisheyeFit::Circular
            } else {
                FisheyeFit::Diagonal
            };
            if ceiling {
                lens.anisotropy = 1.;
            }
            Some(lens)
        };
        let calibration = lens::calibration(&projection, fisheye, Vec2::ZERO, [512, 512]).unwrap();
        commands.spawn((
            Camera3d::default(),
            Camera {
                is_active: false,
                ..default()
            },
            projection,
            ZeroverseCamera {
                fisheye,
                resolution: Some(UVec2::splat(512)),
                ..default()
            },
            CaptureCameraIndex(id),
            pose,
        ));
        let mut prediction = pose;
        prediction.translation += Vec3::new(0.75, 0.2, 0.);
        prediction.rotation *= Quat::from_rotation_y(0.16);
        predictions.cameras.push(PredictedCamera {
            label: format!("Pred C{id}"),
            world_from_view: prediction.to_matrix().to_cols_array_2d(),
            fov_y: 1.,
            aspect: 1.,
            principal_point_offset: None,
            calibration: Some(calibration),
            path: vec![],
        });
    }
    predictions.cameras.reverse(); // Deliberately exercise ID matching, not list position.
    if !review.predictions {
        predictions.cameras.clear();
    }
    commands.insert_resource(Predictions(predictions));
}

fn capture(mut commands: Commands, mut review: ResMut<Review>, mut exit: MessageWriter<AppExit>) {
    review.frames += 1;
    if review.frames == 60 {
        commands
            .spawn(Screenshot::image(review.image.clone()))
            .observe(save_to_disk(format!(
                "out/editor_camera_gizmos/editor-{}{}.png",
                review.count,
                if review.predictions {
                    "-predictions"
                } else {
                    ""
                }
            )))
            .observe(
                |_: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
                    exit.write(AppExit::Success);
                },
            );
    }
    if review.frames > 180 {
        exit.write(AppExit::error());
    }
}
