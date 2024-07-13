use bevy::{
    prelude::*,
    app::AppExit,
    time::Stopwatch,
};
use bevy_args::{
    parse_args,
    BevyArgsPlugin,
    Deserialize,
    Parser,
    Serialize,
};
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_panorbit_camera::{
    PanOrbitCamera,
    PanOrbitCameraPlugin,
};

use bevy_zeroverse::{
    BevyZeroversePlugin,
    camera::EditorCameraMarker,
    material::{
        MaterialsLoadedEvent,
        ShuffleMaterialsEvent,
        ZeroverseMaterials,
    },
    plucker::PluckerOutput,
    primitive::PrimitiveSettings,
    scene::RegenerateSceneEvent,
};


#[derive(
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
    Reflect,
)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
struct BevyZeroverseViewer {
    #[arg(long, default_value = "true")]
    editor: bool,

    /// view available material basecolor textures in a grid
    #[arg(long, default_value = "false")]
    material_grid: bool,

    /// view plücker embeddings
    #[arg(long, default_value = "false")]
    plucker_visualization: bool,

    #[arg(long, default_value = "true")]
    press_esc_close: bool,

    #[arg(long, default_value = "1920.0")]
    width: f32,

    #[arg(long, default_value = "1080.0")]
    height: f32,

    // TODO: change to just `num_cameras`
    /// number of cameras to spawn in the grid x direction
    #[arg(long, default_value = "1")]
    cameras_x: u16,

    /// number of cameras to spawn in the grid y direction
    #[arg(long, default_value = "1")]
    cameras_y: u16,

    #[arg(long, default_value = "bevy_zeroverse")]
    name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[arg(long, default_value = "0")]
    regenerate_ms: u32,

    // TODO: figure out an abstraction for this (e.g. move along sampled surface)
    /// automatically rotate the camera yaw
    #[arg(long, default_value = "0.0")]
    yaw_speed: f32,
}

impl Default for BevyZeroverseViewer {
    fn default() -> BevyZeroverseViewer {
        BevyZeroverseViewer {
            editor: true,
            material_grid: false,
            plucker_visualization: false,
            press_esc_close: true,
            width: 1920.0,
            height: 1080.0,
            cameras_x: 1,
            cameras_y: 1,
            name: "bevy_zeroverse".to_string(),
            regenerate_ms: 0,
            yaw_speed: 0.0,
        }
    }
}


fn viewer_app() {
    let args = parse_args::<BevyZeroverseViewer>();
    info!("{:?}", args);

    let mut app = App::new();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: args.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: (args.width, args.height).into(),
        title: args.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    app.insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)));
    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    app.add_plugins(default_plugins);

    app.add_plugins(BevyArgsPlugin::<BevyZeroverseViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    app.insert_resource(Msaa::Sample8);

    if args.editor {
        app.register_type::<BevyZeroverseViewer>();
        app.register_type::<Image>();
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if args.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);
    app.init_resource::<PrimitiveSettings>();

    app.add_systems(Startup, setup_camera);

    app.add_systems(PreUpdate, press_m_shuffle_materials);
    app.add_systems(PreUpdate, setup_material_grid);
    app.add_systems(PreUpdate, regenerate_scene_system);

    app.add_systems(PostUpdate, setup_plucker_visualization);

    app.run();
}


fn setup_camera(
    mut commands: Commands,
) {
    commands.spawn((
        EditorCameraMarker,
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.0,
            pan_smoothness: 0.0,
            zoom_smoothness: 0.0,
            radius: Some(3.0),
            ..default()
        },
    ));
}

// TODO: setup grid view of sampled cameras


#[derive(Component)]
struct MaterialGrid;

fn setup_material_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
    standard_materials: Res<Assets<StandardMaterial>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
    material_grids: Query<Entity, With<MaterialGrid>>,
    mut materials_loaded: EventReader<MaterialsLoadedEvent>,
) {
    if materials_loaded.is_empty() {
        return;
    }
    materials_loaded.clear();

    if !material_grids.is_empty() {
        commands.entity(material_grids.single()).despawn_recursive();
    }

    if args.material_grid {
        let material_count = zeroverse_materials.materials.len();
        let rows = (material_count as f32).sqrt().ceil() as u16;
        let cols = (material_count as f32 / rows as f32).ceil() as u16;

        commands.spawn(NodeBundle {
            style: Style {
                display: Display::Grid,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_template_columns: RepeatedGridTrack::flex(cols, 1.0),
                grid_template_rows: RepeatedGridTrack::flex(rows, 1.0),
                ..default()
            },
            background_color: BackgroundColor(Color::BLACK),
            ..default()
        })
        .with_children(|builder| {
            for material in &zeroverse_materials.materials {
                let base_color_texture = standard_materials
                    .get(material)
                    .unwrap()
                    .base_color_texture
                    .clone()
                    .unwrap();

                builder.spawn(ImageBundle {
                    style: Style {
                        width: Val::Percent(100.0),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    image: UiImage {
                        texture: base_color_texture,
                        ..default()
                    },
                    ..default()
                });
            }
        }).insert(MaterialGrid);
    }
}

#[derive(Component)]
struct PluckerVisualization;

fn setup_plucker_visualization(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
    plucker_output: Query<(Entity, &Camera, &PluckerOutput)>,
    plucker_visualization: Query<
        Entity,
        With<PluckerVisualization>,
    >,
) {
    let visualization_active = !plucker_visualization.is_empty();

    if !args.plucker_visualization {
        if visualization_active {
            for entity in plucker_visualization.iter() {
                commands.entity(entity).despawn_recursive();
            }
        }
        return;
    }

    if visualization_active {
        return;
    }

    if plucker_output.is_empty() {
        warn_once!("PluckerOutput is not attached - enable the `plucker` feature");
        return;
    }

    let plucker_output = plucker_output
        .iter()
        .sort_by::<(&Camera, &PluckerOutput)>(|(camera_a, _), (camera_b, _)| {
            camera_a.order.cmp(&camera_b.order)
        });

    let plucker_width: f32 = 256.0 / args.cameras_x as f32;
    let plucker_height = 256.0 / args.cameras_y as f32;

    for (
        entity,
        _,
        plucker_output,
    ) in plucker_output {
        commands.spawn((
            ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(0.0),
                    right: Val::Px(0.0),
                    width: Val::Px(plucker_width),
                    height: Val::Px(plucker_height),
                    ..default()
                },
                image: UiImage {
                    texture: plucker_output.visualization.clone(),
                    ..default()
                },
                ..default()
            },
            PluckerVisualization,
            TargetCamera(entity),
        ));
    }
}


#[allow(clippy::too_many_arguments)]
fn regenerate_scene_system(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
    keys: Res<ButtonInput<KeyCode>>,
    clear_meshes: Query<Entity, With<Handle<Mesh>>>,
    clear_zeroverse_primitives: Query<Entity, With<PrimitiveSettings>>,
    time: Res<Time>,
    mut regenerate_stopwatch: Local<Stopwatch>,
    mut regenerate_event: EventWriter<RegenerateSceneEvent>,
) {
    if args.regenerate_ms > 0 {
        regenerate_stopwatch.tick(time.delta());
    }

    let mut regenerate_scene = regenerate_stopwatch.elapsed().as_millis() > args.regenerate_ms as u128;
    regenerate_scene |= keys.just_pressed(KeyCode::KeyR);

    if regenerate_scene {
        for entity in clear_meshes.iter() {
            commands.entity(entity).despawn_recursive();
        }

        for entity in clear_zeroverse_primitives.iter() {
            commands.entity(entity).despawn_recursive();
        }

        regenerate_event.send(RegenerateSceneEvent);
        regenerate_stopwatch.reset();
    }
}


fn press_m_shuffle_materials(
    keys: Res<ButtonInput<KeyCode>>,
    mut shuffle_events: EventWriter<ShuffleMaterialsEvent>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        shuffle_events.send(ShuffleMaterialsEvent);
    }
}


fn press_esc_close(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}


pub fn main() {
    viewer_app();
}
