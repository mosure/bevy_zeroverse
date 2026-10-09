#![cfg(feature = "viewer")]

use bevy::prelude::*;
use bevy_zeroverse::{
    app::{viewer_app, BevyZeroverseConfig},
    camera::{EditorCameraMarker, ProcessedEditorCameraMarker},
    scene::{procedural_indoor::layout::IndoorManifest, ZeroverseSceneType},
};

/// Exercise the first-party editor's startup contract without requiring a GPU or display.
/// This also runs in normal CI, where the rendering test below is ignored.
#[test]
fn editor_engine_types_and_picking_are_available() {
    use std::any::TypeId;

    assert!(DefaultPlugins
        .build()
        .enabled::<bevy::picking::PickingPlugin>());

    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Mesh>()
        .init_asset::<Image>();

    {
        let registry = app.world().resource::<AppTypeRegistry>().read();
        for (id, name) in [
            (
                TypeId::of::<bevy::gizmos::config::GizmoConfigStore>(),
                "GizmoConfigStore",
            ),
            (
                TypeId::of::<bevy::render::view::ColorGradingSection>(),
                "ColorGradingSection",
            ),
            (
                TypeId::of::<bevy::render::view::ColorGradingGlobal>(),
                "ColorGradingGlobal",
            ),
            (TypeId::of::<AmbientLight>(), "AmbientLight"),
            (TypeId::of::<PointLight>(), "PointLight"),
            (TypeId::of::<DirectionalLight>(), "DirectionalLight"),
            (
                TypeId::of::<bevy::light::cluster::ClusterConfig>(),
                "ClusterConfig",
            ),
            (
                TypeId::of::<bevy::camera::Camera3dDepthLoadOp>(),
                "Camera3dDepthLoadOp",
            ),
            (
                TypeId::of::<bevy::core_pipeline::oit::OrderIndependentTransparencySettings>(),
                "OrderIndependentTransparencySettings",
            ),
        ] {
            assert!(registry.get(id).is_some(), "inspector requires {name}");
        }
    }

    // This panicked before automatic engine type registration was enabled.
    app.add_plugins(
        bevy::camera_controller::pan_orbit_camera::controller::MinimalPanOrbitCameraPlugin,
    );
    assert!(app.is_plugin_added::<bevy::camera_controller::pan_orbit_camera::controller::MinimalPanOrbitCameraPlugin>());
}

#[test]
#[ignore = "requires a native graphics adapter and display server"]
fn editor_starts_at_a_generated_interior_viewpoint() {
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            editor: true,
            headless: false,
            gizmos: false,
            num_cameras: 0,
            width: 800.0,
            height: 600.0,
            keybinds: false,
            ..default()
        }),
    );
    app.insert_resource(bevy::winit::WinitSettings {
        focused_mode: bevy::winit::UpdateMode::Continuous,
        unfocused_mode: bevy::winit::UpdateMode::Continuous,
    });
    app.add_systems(PostUpdate, check_viewer);
    std::fs::create_dir_all("out/indoor_viewer").unwrap();
    let exit = app.run();
    assert!(matches!(exit, AppExit::Success));
    assert!(std::path::Path::new("out/indoor_viewer/viewer.png").exists());
}

#[allow(clippy::type_complexity)]
fn check_viewer(
    mut commands: Commands,
    scene: Option<Res<IndoorManifest>>,
    cameras: Query<
        (&GlobalTransform, &Projection),
        (With<EditorCameraMarker>, With<ProcessedEditorCameraMarker>),
    >,
    mut frame: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    *frame += 1;
    if *frame == 64 {
        let scene = scene.unwrap();
        let expected = Quat::from_rotation_y(scene.world_yaw) * scene.cameras[0].start;
        let (transform, projection) = cameras.single().unwrap();
        let Projection::Perspective(projection) = projection else {
            panic!("indoor viewer needs a perspective projection");
        };
        assert!((projection.fov - scene.cameras[0].fov_degrees.to_radians()).abs() < 1e-5);
        assert!(
            transform.translation().distance(expected) < 0.01,
            "editor at {:?}, expected {expected:?}",
            transform.translation()
        );
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(
                "out/indoor_viewer/viewer.png",
            ));
    }
    if *frame == 96 {
        exit.write(AppExit::Success);
    }
}
