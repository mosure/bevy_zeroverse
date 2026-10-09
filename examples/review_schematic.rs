//! Native viewport integration check and screenshots of all three display modes.
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use bevy_zeroverse::{
    app::{
        editor::{model::Page, Action, Actions},
        viewer_app, BevyZeroverseConfig,
    },
    scene::{procedural_indoor::layout::IndoorManifest, ZeroverseSceneType},
};
#[derive(Resource, Default)]
struct Review {
    frames: u32,
    stage: usize,
}
fn main() {
    std::fs::create_dir_all("out/schematic/ui").unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(47_586_113),
            num_cameras: 4,
            width: 512.,
            height: 512.,
            playback_mode: bevy_zeroverse::camera::PlaybackMode::Still,
            playback_speed: 0.,
            yaw_speed: 0.,
            room_schematic: true,
            ..default()
        }),
    );
    for mut window in app
        .world_mut()
        .query::<&mut Window>()
        .iter_mut(app.world_mut())
    {
        window.resolution.set(1360., 900.);
    }
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .init_resource::<Review>()
        .add_systems(Update, review)
        .run();
}
fn review(world: &mut World) {
    if !world.contains_resource::<IndoorManifest>() {
        return;
    }
    let mut review = world.remove_resource::<Review>().unwrap();
    review.frames += 1;
    if review.frames == 90 {
        world
            .resource_mut::<Actions>()
            .0
            .push(Action::Page(Page::View));
    }
    if review.frames == 180 {
        let config = world.resource::<BevyZeroverseConfig>();
        let hidden = config.room_schematic || config.camera_grid;
        for (camera, orbit) in world
            .query::<(
                &Camera,
                &bevy::camera_controller::pan_orbit_camera::prelude::PanOrbitCamera,
            )>()
            .iter(world)
        {
            assert_eq!(camera.is_active, !hidden);
            if hidden {
                assert!(
                    !orbit.enabled_motion.orbit,
                    "hidden editor must not respond to schematic gestures"
                );
            }
        }

        let names = ["schematic", "grid", "editor", "schematic-return"];
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(format!(
                "out/schematic/ui/{}.png",
                names[review.stage]
            )));
    }
    if review.frames == 200 {
        review.stage += 1;
        review.frames = 0;
        if review.stage == 4 {
            world.write_message(AppExit::Success);
        } else {
            let mode = ["schematic", "grid", "editor", "schematic"][review.stage];
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Set("@viewport".into(), serde_json::json!(mode)));
        }
    }
    world.insert_resource(review);
}
