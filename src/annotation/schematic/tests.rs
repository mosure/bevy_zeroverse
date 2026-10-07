use super::*;
use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
fn scene() -> IndoorManifest {
    IndoorManifest::generate_with_humans(47_586_113, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap()
}
#[test]
fn metric_projection_roundtrips_and_predictions_do_not_change_extent() {
    let mut manifest = scene();
    manifest.world_yaw = 0.8;
    let plan = Schematic::from_manifest(&manifest, 0.3).unwrap();
    let options = RenderOptions::default();
    let projection = plan.projection(&options).unwrap();
    for p in &plan.footprints[0].points {
        let recovered = projection.unproject(projection.project(*p), p[1]);
        assert!(Vec3::from_array(recovered).distance(Vec3::from_array(*p)) < 1e-5);
    }
    let mut overlay = Overlay::default();
    let mut camera = plan.cameras[0].clone();
    camera.world_from_view[3][0] += 100.;
    camera.label = "<prediction & camera>".into();
    overlay.cameras.push(camera);
    let svg = plan.svg(&options, &overlay).unwrap();
    assert!(svg.contains("&lt;prediction &amp; camera&gt;"));
    assert!(svg.contains("Dashed: predictions"));
    assert_eq!(projection, plan.projection(&options).unwrap());
    let rgba = plan.rgba(&options, &overlay).unwrap();
    assert_eq!(rgba.len(), 1024 * 1024 * 4);
    assert!(rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
}
#[test]
fn captured_cameras_and_pose_steps_override_planned_state() {
    let manifest = scene();
    let mut sample = crate::sample::Sample {
        indoor: Some(manifest.clone()),
        view_dim: 3,
        ..default()
    };
    for step in 0..2 {
        for camera in &manifest.cameras {
            sample.views.push(crate::sample::View {
                world_from_view: camera
                    .transform_at(step as f32)
                    .to_matrix()
                    .to_cols_array_2d(),
                fovy: camera.fov_degrees.to_radians(),
                trajectory_progress: Some(step as f32),
                time_seconds: Some(step as f32 * 5.),
                ..default()
            });
        }
    }
    sample.views[4].world_from_view[3][0] += 0.123;
    sample.human_pose_steps = vec![
        vec![],
        vec![crate::sample::HumanPoseSample {
            bone_positions: vec![[1., 2., 3.], [2., 3., 4.]],
            bone_rotations: vec![],
        }],
    ];
    sample.human_bone_parents = vec![-1, 0];
    let plan = sample.schematic(1).unwrap();
    assert_eq!(
        plan.cameras[1].world_from_view,
        sample.views[4].world_from_view
    );
    assert_eq!(plan.time_seconds, Some(5.));
    assert_eq!(
        plan.poses[0].joints,
        sample.human_pose_steps[1][0].bone_positions
    );
    plan.rgba(&RenderOptions::default(), &Overlay::default())
        .unwrap();
    assert!(sample.schematic(2).is_err());
}
#[test]
fn invalid_lenses_and_poses_fail_without_panics() {
    let mut plan = Schematic::from_manifest(&scene(), 0.).unwrap();
    let options = RenderOptions::default();
    plan.cameras[0].aspect = f32::NAN;
    assert!(plan.svg(&options, &Overlay::default()).is_err());
    plan.cameras.clear();
    plan.poses = vec![Pose {
        label: "broken".into(),
        joints: vec![[0.; 3]],
        parents: vec![5],
    }];
    assert!(plan.svg(&options, &Overlay::default()).is_err());
    assert!(plan
        .projection(&RenderOptions {
            width: 0,
            ..options
        })
        .is_err());
}

#[test]
fn recorded_root_transform_aligns_geometry_and_trajectories() {
    let mut manifest = scene();
    manifest.world_yaw = 0.4;
    let root = Mat4::from_rotation_translation(Quat::from_rotation_y(0.9), Vec3::new(2., 0., -1.));
    let mut sample = crate::sample::Sample {
        indoor: Some(manifest.clone()),
        view_dim: 3,
        indoor_render_metadata: Some(
            serde_json::json!({"world_from_scene":root.to_cols_array_2d()}),
        ),
        ..default()
    };
    for c in &manifest.cameras {
        sample.views.push(crate::sample::View {
            world_from_view: (root * c.transform_at(0.3).to_matrix()).to_cols_array_2d(),
            fovy: c.fov_degrees.to_radians(),
            trajectory_progress: Some(0.3),
            ..default()
        });
    }
    let plan = sample.schematic(0).unwrap();
    let floor = manifest.envelope.as_ref().unwrap().footprint[0];
    assert!(
        Vec3::from_array(plan.footprints[0].points[0])
            .distance(root.transform_point3(Vec3::new(floor.x, 0., floor.y)))
            < 1e-5
    );
    assert!(
        Vec3::from_array(plan.cameras[0].path[0])
            .distance(root.transform_point3(manifest.cameras[0].start))
            < 1e-5
    );
    for object in &manifest.objects {
        assert_eq!(
            plan.footprints
                .iter()
                .any(|f| f.instance_id == Some(object.id)),
            !object.neighbor
        );
    }
}
