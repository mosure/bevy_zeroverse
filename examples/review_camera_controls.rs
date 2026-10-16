//! Actual Feathers screenshots and fixed-camera/runtime regression probe.
//! cargo run --example review_camera_controls
use bevy::{
    camera::primitives::Aabb,
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
};
use bevy_zeroverse::{
    app::{
        editor::{
            model::{EditorState, Page},
            Action, Actions,
        },
        viewer_app, BevyZeroverseConfig,
    },
    camera::{CaptureCameraIndex, PlaybackMode},
    scene::{procedural_indoor::layout::IndoorManifest, ZeroverseSceneType},
};

#[derive(Resource, Default)]
struct Review {
    frames: u32,
    poses: Vec<(usize, Transform)>,
    meshes: std::collections::HashMap<Entity, (Transform, Option<Aabb>)>,
    changed: std::collections::HashSet<Entity>,
}
fn main() {
    std::fs::create_dir_all("out/camera_controls/native").unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_seed: Some(61458299),
            indoor_camera: Some(r#"{"motion_fraction":0}"#.into()),
            indoor_motion: Some(r#"{"chair_fraction":1,"door_fraction":1,"cabinet_fraction":1,"covering_fraction":1}"#.into()),
            num_cameras: 4,
            width: 512.,
            height: 512.,
            camera_grid: true,
            gizmos: false,
            playback_mode: PlaybackMode::Loop,
            playback_speed: 0.2,
            ..default()
        }),
    );
    for mut window in app
        .world_mut()
        .query::<&mut Window>()
        .iter_mut(app.world_mut())
    {
        window.resolution.set(1440., 1080.);
    }
    app.insert_resource(bevy::winit::WinitSettings::continuous())
        .init_resource::<Review>()
        .add_systems(Update, review)
        .run();
}
fn review(world: &mut World) {
    if !world.contains_resource::<IndoorManifest>()
        || world.resource::<EditorState>().active_seed != Some(61458299)
    {
        return;
    }
    let mut review = world.remove_resource::<Review>().unwrap();
    review.frames += 1;
    if review.frames == 60 {
        world
            .resource_mut::<Actions>()
            .0
            .push(Action::Page(Page::Cameras));
        review.poses = world
            .query::<(&CaptureCameraIndex, &Transform)>()
            .iter(world)
            .map(|(index, pose)| (index.0, *pose))
            .collect();
        assert_eq!(review.poses.len(), 4);
        review.meshes = world
            .query_filtered::<(Entity, &Transform, Option<&Aabb>), With<Mesh3d>>()
            .iter(world)
            .map(|(entity, transform, bounds)| (entity, (*transform, bounds.copied())))
            .collect();
    }
    if review.frames > 60 {
        for (index, pose) in world
            .query::<(&CaptureCameraIndex, &Transform)>()
            .iter(world)
        {
            assert_eq!(
                *pose,
                review.poses.iter().find(|(i, _)| *i == index.0).unwrap().1
            );
        }
        assert_eq!(
            world
                .resource::<IndoorManifest>()
                .camera_settings
                .motion_fraction,
            0.
        );
    }
    if review.frames > 60 {
        for (entity, transform, bounds) in world
            .query_filtered::<(Entity, &Transform, Option<&Aabb>), With<Mesh3d>>()
            .iter(world)
        {
            if review
                .meshes
                .get(&entity)
                .is_some_and(|old| *old != (*transform, bounds.copied()))
            {
                review.changed.insert(entity);
            }
        }
    }
    if review.frames == 130 {
        for mut offset in world.query::<&mut ScrollPosition>().iter_mut(world) {
            offset.0.y = 520.;
        }
    }
    if review.frames == 190 {
        let actions = &mut world.resource_mut::<Actions>().0;
        actions.push(Action::Set("@progress".into(), serde_json::json!(0.4)));
        actions.push(Action::Set("@lens_bias".into(), serde_json::json!(0.55)));
        actions.push(Action::Set("@lens_spread".into(), serde_json::json!(0.4)));
        actions.push(Action::Set(
            "@camera_motion_percent".into(),
            serde_json::json!(50.),
        ));
    }
    if review.frames == 250 {
        for mut offset in world.query::<&mut ScrollPosition>().iter_mut(world) {
            offset.0.y = 0.;
        }
    }
    if [110, 170, 230, 290].contains(&review.frames) {
        world
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(format!(
                "out/camera_controls/native/{}.png",
                review.frames
            )));
    }
    if review.frames == 320 {
        let changed_meshes = review.changed.len();
        assert!(
            changed_meshes > 0,
            "scene furniture must keep moving while cameras are fixed"
        );
        let state = world.resource::<EditorState>();
        assert!(state.pending());
        assert_eq!(
            state.draft["indoor_camera"]["motion_fraction"],
            serde_json::json!(0.5)
        );
        std::fs::write("out/camera_controls/native/state.json", serde_json::to_vec_pretty(&serde_json::json!({
            "fixed_poses_unchanged":true, "changed_meshes":changed_meshes, "applied":state.applied, "pending":state.draft,
            "elapsed_frames":review.frames, "poses":review.poses.iter().map(|(i,t)| (i, t.to_matrix().to_cols_array())).collect::<Vec<_>>()
        })).unwrap()).unwrap();
        world.write_message(AppExit::Success);
    }
    world.insert_resource(review);
}
