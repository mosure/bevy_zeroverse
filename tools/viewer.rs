use bevy::{
    prelude::*,
    app::AppExit,
    core_pipeline::{
        bloom::BloomSettings,
        core_3d::ScreenSpaceTransmissionQuality,
        tonemapping::Tonemapping,
    },
    render::camera::Exposure,
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

#[cfg(not(all(feature = "webgl2", target_arch = "wasm32")))]
use bevy::core_pipeline::experimental::taa::{
    TemporalAntiAliasBundle, TemporalAntiAliasPlugin,
};

use bevy_zeroverse::{
    BevyZeroversePlugin,
    material::{
        MaterialsLoadedEvent,
        ShuffleMaterialsEvent,
        ZeroverseMaterials,
    },
    primitive::{
        PrimitiveBundle,
        PrimitiveSettings,
    },
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

    #[arg(long, default_value = "true")]
    press_esc_close: bool,

    #[arg(long, default_value = "1920.0")]
    width: f32,

    #[arg(long, default_value = "1080.0")]
    height: f32,

    #[arg(long, default_value = "bevy_zeroverse")]
    name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[arg(long, default_value = "0")]
    regenerate_ms: u32,
}

impl Default for BevyZeroverseViewer {
    fn default() -> BevyZeroverseViewer {
        BevyZeroverseViewer {
            editor: true,
            material_grid: false,
            press_esc_close: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_zeroverse".to_string(),
            regenerate_ms: 0,
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

    app.insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)));
    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    app.add_plugins(default_plugins);

    app.add_plugins(BevyArgsPlugin::<BevyZeroverseViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    #[cfg(not(all(feature = "webgl2", target_arch = "wasm32")))]
    app.insert_resource(Msaa::Off)
        .add_plugins(TemporalAntiAliasPlugin);

    if args.editor {
        app.register_type::<BevyZeroverseViewer>();
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if args.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);
    app.init_resource::<PrimitiveSettings>();

    app.add_systems(Startup, setup_camera);
    app.add_systems(Startup, setup_primitives);

    app.add_systems(PreUpdate, press_m_shuffle_materials);
    app.add_systems(PreUpdate, setup_material_grid);
    app.add_systems(PostUpdate, regenerate_scene_system);

    app.run();
}

fn setup_camera(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    #[allow(unused_mut, unused_variables)]
    let mut camera = commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            camera_3d: Camera3d {
                screen_space_specular_transmission_quality: ScreenSpaceTransmissionQuality::High,
                ..default()
            },
            exposure: Exposure::INDOOR,
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            tonemapping: Tonemapping::TonyMcMapface,
            ..default()
        },
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.0,
            pan_smoothness: 0.0,
            zoom_smoothness: 0.0,
            ..default()
        },
        // TODO: generate environment maps
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 900.0,
        },
        BloomSettings::default(),
    ));

    // note: disable TAA in headless output mode
    #[cfg(not(all(feature = "webgl2", target_arch = "wasm32")))]
    camera.insert((
        TemporalAntiAliasBundle::default(),
    ));


    // TODO: move lighting to procedural scene plugin
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_xyz(50.0, 50.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
        directional_light: DirectionalLight {
            illuminance: 1_500.,
            ..default()
        },
        ..default()
    });
}


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


fn setup_primitives(
    mut commands: Commands,
    primitive_settings: Res<PrimitiveSettings>,
) {
    commands.spawn(PrimitiveBundle {
        settings: primitive_settings.clone(),
        ..default()
    });
}

#[allow(clippy::too_many_arguments)]
fn regenerate_scene_system(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
    keys: Res<ButtonInput<KeyCode>>,
    clear_meshes: Query<Entity, With<Handle<Mesh>>>,
    clear_zeroverse_primitives: Query<Entity, With<PrimitiveSettings>>,
    primitive_settings: Res<PrimitiveSettings>,
    time: Res<Time>,
    mut regenerate_stopwatch: Local<Stopwatch>,
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

        setup_primitives(commands, primitive_settings);
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
