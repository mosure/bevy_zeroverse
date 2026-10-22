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
    groups: bool,
    warmup: u32,
    output: std::path::PathBuf,
}
fn main() {
    std::env::set_var("BEVY_ASSET_ROOT", env!("CARGO_MANIFEST_DIR"));
    let groups = std::env::args().any(|arg| arg == "--groups");
    let output = std::path::PathBuf::from(
        std::env::args()
            .skip(1)
            .find(|arg| arg != "--groups")
            .unwrap_or_else(|| "out/studio_review".into()),
    );
    std::fs::create_dir_all(&output).unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(6),
            num_cameras: if groups { 1 } else { 3 },
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
    app.insert_resource(Review {
        output,
        groups,
        ..default()
    })
    .insert_resource(bevy::winit::WinitSettings::continuous())
    .add_systems(Update, review)
    .run();
}
fn review(world: &mut World) {
    if !world.contains_resource::<IndoorManifest>() {
        return;
    }
    let mut review = world.remove_resource::<Review>().unwrap();
    if review.groups && review.warmup < 80 {
        if !bevy_zeroverse::scene::procedural_indoor::indoor_generation_pending(world) {
            review.warmup += 1;
        }
        world.insert_resource(review);
        return;
    }
    review.frame += 1;
    if review.groups {
        review_groups(world, &mut review);
        world.insert_resource(review);
        return;
    }
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
    if review.frame == 24 {
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(
                review.output.join(format!("{}.png", names[review.stage])),
            ));
    }
    if review.frame == 32 {
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

/// Small visual review: one scene, no model loading or annotation capture sweep.
fn review_groups(world: &mut World, review: &mut Review) {
    use serde_json::json;
    let names = [
        "scene",
        "camera-groups",
        "fisheye",
        "furnishing-motion",
        "overlays",
        "capture",
        "compact",
    ];
    if review.frame == 24 {
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(
                review.output.join(format!("{}.png", names[review.stage])),
            ));
    }
    if review.frame != 32 {
        return;
    }
    review.frame = 0;
    review.stage += 1;
    let actions = match review.stage {
        1 => vec![
            Action::Page(Page::Cameras),
            Action::Section("cameras.rig".into()),
        ],
        2 => vec![Action::JumpToSection("cameras.fisheye".into())],
        3 => vec![
            Action::Page(Page::People),
            Action::Set("@indoor_motion".into(), json!(true)),
            Action::JumpToSection("people.motion-selection".into()),
        ],
        4 => vec![
            Action::Page(Page::View),
            Action::JumpToSection("view.overlays".into()),
        ],
        5 => vec![Action::Page(Page::Advanced)],
        6 => {
            for mut window in world.query::<&mut Window>().iter_mut(world) {
                window.resolution.set(960., 640.);
            }
            vec![
                Action::Page(Page::Cameras),
                Action::Section("cameras.fisheye".into()),
                Action::JumpToSection("cameras.rig".into()),
            ]
        }
        _ => {
            world.write_message(AppExit::Success);
            return;
        }
    };
    world.resource_mut::<Actions>().0.extend(actions);
}
