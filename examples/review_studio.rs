//! Native UI and material-annotation shader qualification screenshots.
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
    frame: u32,
    stage: usize,
}
fn main() {
    std::fs::create_dir_all("out/release_0350/ui").unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            num_cameras: 3,
            width: 512.,
            height: 512.,
            gizmos: false,
            playback_speed: 0.,
            yaw_speed: 0.,
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
    app.init_resource::<Review>()
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .add_systems(Update, review)
        .run();
}
fn review(world: &mut World) {
    if !world.contains_resource::<IndoorManifest>() {
        return;
    }
    let mut review = world.remove_resource::<Review>().unwrap();
    review.frame += 1;
    let names = [
        "scene",
        "cameras",
        "materials",
        "people",
        "view",
        "advanced",
        "depth",
        "normal",
        "position",
        "semantic",
        "flow",
        "co-visibility",
        "glass-through",
    ];
    if review.frame == 90 {
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(format!(
                "out/release_0350/ui/{}.png",
                names[review.stage]
            )));
    }
    if review.frame == 110 {
        review.frame = 0;
        review.stage += 1;
        if review.stage == names.len() {
            world.write_message(AppExit::Success);
        } else if review.stage < 6 {
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Page(Page::ALL[review.stage]));
        } else {
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Page(Page::View));
            let mode = [
                "Depth",
                "Normal",
                "Position",
                "Semantic",
                "OpticalFlow",
                "CoVisibility",
                "Semantic",
            ][review.stage - 6];
            world
                .resource_mut::<Actions>()
                .0
                .push(Action::Set("/render_mode".into(), serde_json::json!(mode)));
            if review.stage == 11 {
                world
                    .resource_mut::<Actions>()
                    .0
                    .push(Action::Set("@viewport".into(), serde_json::json!("grid")));
            }
            if review.stage == 12 {
                world.resource_mut::<Actions>().0.push(Action::Set(
                    "/annotation_glass".into(),
                    serde_json::json!("through"),
                ));
            }
        }
    }
    world.insert_resource(review);
}
