use bevy::{
    prelude::*,
    app::AppExit,
    core_pipeline::{
        bloom::BloomSettings,
        core_3d::ScreenSpaceTransmissionQuality,
        tonemapping::Tonemapping,
    },
    render::{
        camera::Exposure,
        view::ColorGrading,
    },
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
    material::ZeroverseMaterials,
    primitive::PrimitiveSettings,
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

    #[arg(long, default_value = "10")]
    primitive_count: usize,
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
            primitive_count: 10,
        }
    }
}


fn viewer_app() {
    let config = parse_args::<BevyZeroverseViewer>();
    info!("{:?}", config);

    let mut app = App::new();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        // fit_canvas_to_parent: true,
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: config.name.clone(),

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
        resolution: (config.width, config.height).into(),
        title: config.name.clone(),

        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,

        ..default()
    });

    app.insert_resource(ClearColor(Color::rgb_u8(0, 0, 0)));
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

    if config.editor {
        app.register_type::<BevyZeroverseViewer>();
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);
    app.add_systems(Startup, setup_camera);
    app.add_systems(Startup, setup_primitives);
    app.add_systems(Update, press_r_to_reset);

    app.run();
}

fn setup_camera(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    args: Res<BevyZeroverseViewer>,
    standard_materials: Res<Assets<StandardMaterial>>,
    zeroverse_materials: Res<ZeroverseMaterials>,
) {
    #[allow(unused_mut)]
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
            color_grading: ColorGrading {
                post_saturation: 1.2,
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

    let camera = camera.id();


    // TODO: move lighting to procedural scene plugin
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_xyz(50.0, 50.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
        directional_light: DirectionalLight {
            illuminance: 1_500.,
            ..default()
        },
        ..default()
    });


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
        })
        .insert(TargetCamera(camera));
    }
}


fn setup_primitives(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
) {
    commands.spawn(PrimitiveSettings::count(args.primitive_count));
}

fn press_r_to_reset(
    mut commands: Commands,
    args: Res<BevyZeroverseViewer>,
    keys: Res<ButtonInput<KeyCode>>,
    clear_meshes: Query<Entity, With<Handle<Mesh>>>,
    clear_zeroverse_primitives: Query<Entity, With<PrimitiveSettings>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        for entity in clear_meshes.iter() {
            commands.entity(entity).despawn_recursive();
        }

        for entity in clear_zeroverse_primitives.iter() {
            commands.entity(entity).despawn_recursive();
        }

        setup_primitives(commands, args);
    }
}


fn press_esc_close(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>
) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.send(AppExit);
    }
}


pub fn main() {
    viewer_app();
}
