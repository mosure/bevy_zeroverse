use bevy::{
    prelude::*,
    app::AppExit,
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

use bevy_zeroverse::BevyZeroversePlugin;


#[derive(
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
pub struct BevyZeroverseViewer {
    #[arg(long, default_value = "true")]
    pub editor: bool,

    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[arg(long, default_value = "bevy_zeroverse")]
    pub name: String,
}

impl Default for BevyZeroverseViewer {
    fn default() -> BevyZeroverseViewer {
        BevyZeroverseViewer {
            editor: true,
            press_esc_close: true,
            width: 1920.0,
            height: 1080.0,
            name: "bevy_zeroverse".to_string(),
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
        })
        .set(AssetPlugin {
            mode: AssetMode::Processed,
            ..default()
        });

    app.add_plugins(default_plugins);
    app.add_plugins(BevyArgsPlugin::<BevyZeroverseViewer>::default());
    app.add_plugins(PanOrbitCameraPlugin);

    if config.editor {
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if config.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);
    app.add_systems(Startup, setup_camera);

    app.run();
}

pub fn setup_camera(
    mut commands: Commands,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
            ..default()
        },
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.0,
            pan_smoothness: 0.0,
            zoom_smoothness: 0.0,
            ..default()
        },
    ));
}

pub fn press_esc_close(
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
