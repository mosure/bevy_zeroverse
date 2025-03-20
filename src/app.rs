use bevy::{
    prelude::*,
    app::AppExit,
    time::Stopwatch,
    render::{
        camera::RenderTarget,
        // render_resource::{
        //     Extent3d,
        //     TextureDescriptor,
        //     TextureDimension,
        //     TextureUsages,
        // },
        // renderer::RenderDevice,
        settings::{
            RenderCreation,
            WgpuFeatures,
            WgpuSettings,
        },
        // texture::{
        //     ImageLoaderSettings,
        //     ImageSampler,
        // },
        RenderPlugin,
    },
    winit::{
        WakeUp,
        WinitPlugin,
    },
};
use bevy_args::{
    parse_args,
    Deserialize,
    Parser,
    Serialize,
};
#[cfg(feature = "viewer")]
use bevy_inspector_egui::quick::WorldInspectorPlugin;
#[cfg(feature = "viewer")]
use bevy_panorbit_camera::{
    PanOrbitCamera,
    PanOrbitCameraPlugin,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{
    BevyZeroversePlugin,
    camera::{
        DefaultZeroverseCamera,
        EditorCameraMarker,
        Playback,
        PlaybackMode,
        ZeroverseCamera,
    },
    io,
    material::{
        MaterialsLoadedEvent,
        ShuffleMaterialsEvent,
        ZeroverseMaterials,
    },
    mesh::ShuffleMeshesEvent,
    // plucker::ZeroversePluckerSettings,
    primitive::ScaleSampler,
    render::RenderMode,
    scene::{
        room::ZeroverseRoomSettings,
        RegenerateSceneEvent,
        SceneLoadedEvent,
        ZeroverseSceneRoot,
        ZeroverseSceneSettings,
        ZeroverseSceneType,
    },
};


// TODO: add meta-derive macro to populate get/set methods
#[cfg(feature = "python")]
#[derive(
    Clone,
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
    Reflect,
)]
#[pyclass]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct BevyZeroverseConfig {
    /// enable the bevy inspector
    #[pyo3(get, set)]
    #[arg(long, default_value = "true")]
    pub editor: bool,

    /// enable gizmo drawing on editor camera
    #[pyo3(get, set)]
    #[arg(long, default_value = "true")]
    pub gizmos: bool,

    /// no window will be shown
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub headless: bool,

    /// whether or not zeroverse cameras receive image copiers
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub image_copiers: bool,

    /// view available material basecolor textures in a grid
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub material_grid: bool,

    /// view plücker embeddings
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub plucker_visualization: bool,

    /// enable closing the window with the escape key (doesn't work in web)
    #[pyo3(get, set)]
    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[pyo3(get, set)]
    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub num_cameras: usize,

    /// display a grid of Zeroverse cameras
    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub camera_grid: bool,

    /// window title
    #[pyo3(get, set)]
    #[arg(long, default_value = "bevy_zeroverse")]
    pub name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_ms: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_scene_material_shuffle_period: u32,

    /// afer this many scene regenerations, shuffle the meshes
    #[pyo3(get, set)]
    #[arg(long, default_value = "0")]
    pub regenerate_scene_mesh_shuffle_period: u32,

    /// automatically rotate the root scene object in the y axis
    #[pyo3(get, set)]
    #[arg(long, default_value = "0.0")]
    pub yaw_speed: f32,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = RenderMode::Color)]
    pub render_mode: RenderMode,

    #[pyo3(get, set)]
    pub render_modes: Vec<RenderMode>,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = ZeroverseSceneType::Object)]
    pub scene_type: ZeroverseSceneType,

    #[pyo3(get, set)]
    #[arg(long, default_value = "false")]
    pub rotation_augmentation: bool,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.0")]
    pub max_camera_radius: f32,

    #[pyo3(get, set)]
    #[arg(long, value_enum, default_value_t = PlaybackMode::Still)]
    pub playback_mode: PlaybackMode,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.2")]
    pub playback_speed: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "0.05")]
    pub playback_step: f32,

    #[pyo3(get, set)]
    #[arg(long, default_value = "5")]
    pub playback_steps: u32,
}

#[cfg(feature = "python")]
#[pymethods]
impl BevyZeroverseConfig {
    #[new]
    pub fn new() -> Self {
        Default::default()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[cfg(not(feature = "python"))]
#[derive(
    Clone,
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
    Reflect,
)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct BevyZeroverseConfig {
    /// enable the bevy inspector
    #[arg(long, default_value = "true")]
    pub editor: bool,

    /// enable gizmo drawing on editor camera
    #[arg(long, default_value = "true")]
    pub gizmos: bool,

    /// no window will be shown
    #[arg(long, default_value = "false")]
    pub headless: bool,

    /// whether or not zeroverse cameras receive image copiers
    #[arg(long, default_value = "false")]
    pub image_copiers: bool,

    /// view available material basecolor textures in a grid
    #[arg(long, default_value = "false")]
    pub material_grid: bool,

    /// view plücker embeddings
    #[arg(long, default_value = "false")]
    pub plucker_visualization: bool,

    /// enable closing the window with the escape key (doesn't work in web)
    #[arg(long, default_value = "true")]
    pub press_esc_close: bool,

    #[arg(long, default_value = "1920.0")]
    pub width: f32,

    #[arg(long, default_value = "1080.0")]
    pub height: f32,

    #[arg(long, default_value = "0")]
    pub num_cameras: usize,

    /// display a grid of Zeroverse cameras
    #[arg(long, default_value = "false")]
    pub camera_grid: bool,

    /// window title
    #[arg(long, default_value = "bevy_zeroverse")]
    pub name: String,

    /// move to the next scene after `regenerate_ms` milliseconds
    #[arg(long, default_value = "0")]
    pub regenerate_ms: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[arg(long, default_value = "0")]
    pub regenerate_scene_material_shuffle_period: u32,

    /// afer this many scene regenerations, shuffle the materials
    #[arg(long, default_value = "0")]
    pub regenerate_scene_mesh_shuffle_period: u32,

    /// automatically rotate the root scene object in the y axis
    #[arg(long, default_value = "0.0")]
    pub yaw_speed: f32,

    #[arg(long, value_enum, default_value_t = RenderMode::Color)]
    pub render_mode: RenderMode,

    pub render_modes: Vec<RenderMode>,

    #[arg(long, value_enum, default_value_t = ZeroverseSceneType::Object)]
    pub scene_type: ZeroverseSceneType,

    #[arg(long, default_value = "true")]
    pub rotation_augmentation: bool,

    #[arg(long, default_value = "0.0")]
    pub max_camera_radius: f32,

    #[arg(long, value_enum, default_value_t = PlaybackMode::PingPong)]
    pub playback_mode: PlaybackMode,

    #[arg(long, default_value = "0.2")]
    pub playback_speed: f32,

    #[arg(long, default_value = "0.05")]
    pub playback_step: f32,

    #[arg(long, default_value = "5")]
    pub playback_steps: u32,
}

impl Default for BevyZeroverseConfig {
    fn default() -> BevyZeroverseConfig {
        BevyZeroverseConfig {
            editor: true,
            gizmos: true,
            headless: false,
            image_copiers: false,
            material_grid: false,
            plucker_visualization: false,
            press_esc_close: true,
            width: 1920.0,
            height: 1080.0,
            num_cameras: 0,
            camera_grid: false,
            name: "bevy_zeroverse".to_string(),
            regenerate_ms: 0,
            regenerate_scene_material_shuffle_period: 0,
            regenerate_scene_mesh_shuffle_period: 0,
            yaw_speed: 0.0,
            render_mode: Default::default(),
            render_modes: vec![],
            scene_type: Default::default(),
            rotation_augmentation: true,
            max_camera_radius: 0.0,
            playback_mode: PlaybackMode::PingPong,
            playback_speed: 0.2,
            playback_step: 0.05,
            playback_steps: 5,
        }
    }
}


pub fn viewer_app(
    override_args: Option<BevyZeroverseConfig>,
) -> App {
    let args = match override_args {
        Some(args) => args,
        None => parse_args::<BevyZeroverseConfig>(),
    };

    let mut app = App::new();

    info!("args: {:?}", args);
    app.insert_resource(args.clone());


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

    let mut winit_plugin = WinitPlugin::<WakeUp>::default();
    winit_plugin.run_on_any_thread = true;

    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | WgpuFeatures::SHADER_F16,
                ..Default::default()
            }),
            ..Default::default()
        })
        .set(winit_plugin);

    let default_plugins = if args.headless {
        default_plugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: bevy::window::ExitCondition::DontExit,
                close_when_requested: false,
            })
    } else {
        default_plugins
            .set(WindowPlugin {
                primary_window,
                ..default()
            })
    };

    app.add_plugins(default_plugins);

    if args.image_copiers {
        app.add_plugins(io::image_copy::ImageCopyPlugin);
        app.add_plugins(io::prepass_copy::PrepassCopyPlugin);
    }

    #[cfg(feature = "viewer")]
    app.add_plugins(PanOrbitCameraPlugin);

    #[cfg(feature = "viewer")]
    if args.editor {
        app.register_type::<BevyZeroverseConfig>();
        app.add_plugins(WorldInspectorPlugin::new());
    }

    if args.press_esc_close {
        app.add_systems(Update, press_esc_close);
    }

    app.add_plugins(BevyZeroversePlugin);

    app.insert_resource(args.render_mode);  // TODO: replicate on args changed

    app.insert_resource(DefaultZeroverseCamera {
        resolution: UVec2::new(args.width as u32, args.height as u32).into(),
    });

    app.add_systems(PostStartup, (
        propagate_cli_settings,
        setup_scene,
    ));

    app.add_systems(PreUpdate, press_m_shuffle_materials_and_meshes);

    #[cfg(feature = "viewer")]
    {
        app.add_systems(PreUpdate, setup_material_grid);
        app.add_systems(PostUpdate, setup_camera_grid);
    }

    app.add_systems(Update, rotate_scene);

    app.add_systems(PostUpdate, (
        propagate_cli_settings,
        regenerate_scene_system,
        setup_camera,
    ));

    app
}


#[derive(Component, Debug, Reflect)]
struct MaterialGridCameraMarker;

fn setup_camera(
    args: Res<BevyZeroverseConfig>,
    mut commands: Commands,
    material_grid_cameras: Query<Entity, With<MaterialGridCameraMarker>>,
    editor_cameras: Query<Entity, With<EditorCameraMarker>>,
    room_settings: Res<ZeroverseRoomSettings>,
) {
    if !args.is_changed() {
        return;
    }

    material_grid_cameras.iter().for_each(|entity| {
        commands.entity(entity).despawn_recursive();
    });

    editor_cameras.iter().for_each(|entity| {
        commands.entity(entity).despawn_recursive();
    });

    // TODO: add a check for dataloader/headless mode
    if args.camera_grid {
        commands.spawn(Camera2d)
            .insert(MaterialGridCameraMarker);
        return;
    }

    let camera_offset = Vec3::new(0.0, 0.0, 3.5);
    let camera_offset = match args.scene_type {
        ZeroverseSceneType::SemanticRoom => {
            let max_room_size = match room_settings.room_size {
                ScaleSampler::Bounded(_min, max) => max * Vec3::new(1.0, 2.0, 1.0),
                ScaleSampler::Exact(size) => size,
            };
            max_room_size + camera_offset
        },
        ZeroverseSceneType::CornellCube => camera_offset,
        ZeroverseSceneType::Custom => camera_offset,
        ZeroverseSceneType::Human => camera_offset,
        ZeroverseSceneType::Object => camera_offset,
        ZeroverseSceneType::Room => {
            let max_room_size = match room_settings.room_size {
                ScaleSampler::Bounded(_min, max) => max * Vec3::new(1.0, 2.0, 1.0),
                ScaleSampler::Exact(size) => size,
            };
            max_room_size + camera_offset
        },
    };

    commands.spawn((
        EditorCameraMarker {
            transform: Transform::from_translation(camera_offset)
                .looking_at(Vec3::ZERO, Vec3::Y)
                .into(),
        },
        #[cfg(feature = "viewer")]
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.8,
            pan_smoothness: 0.6,
            zoom_smoothness: 0.8,
            ..default()
        },
    ));
}

#[derive(Component)]
pub struct CameraGrid;

#[cfg(feature = "viewer")]
fn setup_camera_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
    camera_grids: Query<Entity, With<CameraGrid>>,
    zeroverse_cameras: Query<
        (Entity, &Camera),
        With<ZeroverseCamera>,
    >,
    mut scene_loaded: EventReader<SceneLoadedEvent>,
) {
    if zeroverse_cameras.is_empty() {
        return;
    }

    if scene_loaded.is_empty() && !args.is_changed() {
        return;
    }
    scene_loaded.clear();

    if !camera_grids.is_empty() {
        commands.entity(camera_grids.single()).despawn_recursive();
    }

    if args.camera_grid {
        let camera_count = zeroverse_cameras.iter().count();
        let rows = (camera_count as f32).sqrt().ceil() as u16;
        let cols = (camera_count as f32 / rows as f32).ceil() as u16;

        commands.spawn((
            CameraGrid,
            Name::new("camera_grid"),
            Node {
                display: Display::Grid,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_template_columns: RepeatedGridTrack::flex(cols, 1.0),
                grid_template_rows: RepeatedGridTrack::flex(rows, 1.0),
                ..default()
            },
            BackgroundColor(Color::BLACK),
        )).with_children(|builder| {
            for (_, camera) in zeroverse_cameras.iter() {
                let texture = match camera.target.clone() {
                    RenderTarget::Image(texture) => texture,
                    _ => continue,
                };

                builder.spawn(ImageNode {
                    image: texture,
                    ..default()
                });
            }
        });
    }
}


#[derive(Component)]
pub struct MaterialGrid;

#[cfg(feature = "viewer")]
fn setup_material_grid(
    mut commands: Commands,
    args: Res<BevyZeroverseConfig>,
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

        commands.spawn((
            MaterialGrid,
            Name::new("material_grid"),
            Node {
                display: Display::Grid,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                grid_template_columns: RepeatedGridTrack::flex(cols, 1.0),
                grid_template_rows: RepeatedGridTrack::flex(rows, 1.0),
                ..default()
            }
        ))
        .with_children(|builder| {
            for material in &zeroverse_materials.materials {
                let base_color_texture = standard_materials
                    .get(material)
                    .unwrap()
                    .base_color_texture
                    .clone()
                    .unwrap();

                builder.spawn(ImageNode {
                    image: base_color_texture,
                    ..default()
                });
            }
        });
    }
}


fn setup_scene(
    mut regenerate_event: EventWriter<RegenerateSceneEvent>,
) {
    regenerate_event.send(RegenerateSceneEvent);
}

#[allow(clippy::too_many_arguments)]
fn regenerate_scene_system(
    args: Res<BevyZeroverseConfig>,
    keys: Res<ButtonInput<KeyCode>>,
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
        regenerate_event.send(RegenerateSceneEvent);
        regenerate_stopwatch.reset();
    }
}


fn rotate_scene(
    time: Res<Time>,
    args: Res<BevyZeroverseConfig>,
    mut scene_roots: Query<&mut Transform, With<ZeroverseSceneRoot>>,
) {
    for mut transform in scene_roots.iter_mut() {
        let delta_rot = args.yaw_speed * time.delta_secs();
        transform.rotate(Quat::from_rotation_y(delta_rot));
    }
}


fn propagate_cli_settings(
    args: Res<BevyZeroverseConfig>,
    // mut plucker_settings: ResMut<ZeroversePluckerSettings>,
    mut playback: ResMut<Playback>,
    mut render_mode: ResMut<RenderMode>,
    mut scene_settings: ResMut<ZeroverseSceneSettings>,
) {
    if args.is_changed() {
        // plucker_settings.enabled = args.plucker_visualization;

        playback.mode = args.playback_mode;
        playback.speed = args.playback_speed;

        *render_mode = args.render_mode.clone();

        scene_settings.num_cameras = args.num_cameras;
        scene_settings.rotation_augmentation = args.rotation_augmentation;
        scene_settings.scene_type = args.scene_type.clone();
        scene_settings.max_camera_radius = args.max_camera_radius;
    }
}


fn press_m_shuffle_materials_and_meshes(
    keys: Res<ButtonInput<KeyCode>>,
    mut shuffle_material_events: EventWriter<ShuffleMaterialsEvent>,
    mut shuffle_meshes_events: EventWriter<ShuffleMeshesEvent>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        shuffle_material_events.send(ShuffleMaterialsEvent);
        shuffle_meshes_events.send(ShuffleMeshesEvent);
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
