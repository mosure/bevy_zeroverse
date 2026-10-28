use super::*;
use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
fn scene() -> IndoorManifest {
    IndoorManifest::generate_with_humans(47_586_113, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap()
}
#[test]
fn gizmo_scale_exports_without_changing_metric_annotation() {
    let plan = Schematic::from_manifest(&scene(), 0.3).unwrap();
    let before = plan.clone();
    let options = RenderOptions {
        width: 256,
        height: 256,
        ..default()
    };
    let baseline = plan.document(options, Overlay::default()).unwrap();
    for scale in [0.1, 0.5, 2., 4.] {
        let options = RenderOptions {
            camera_gizmo_scale: scale,
            ..options
        };
        let document = plan.document(options, Overlay::default()).unwrap();
        assert_eq!(document.schematic, baseline.schematic);
        assert_eq!(document.projection, baseline.projection);
        assert_eq!(document.options.camera_gizmo_scale, scale);
        let decoded: Document =
            serde_json::from_str(&serde_json::to_string(&document).unwrap()).unwrap();
        assert_eq!(decoded, document);
        assert_eq!(
            plan.rgba(&options, &Overlay::default()).unwrap().len(),
            256 * 256 * 4
        );
    }
    for invalid in [f32::NAN, f32::INFINITY, 0., 4.1] {
        assert!(plan
            .svg(
                &RenderOptions {
                    camera_gizmo_scale: invalid,
                    ..options
                },
                &Overlay::default()
            )
            .is_err());
    }
    assert_eq!(plan, before);
}
#[test]
fn planned_shifted_frustum_matches_full_capture_calibration() {
    let mut scene = scene();
    scene.cameras[0].principal_point_offset = Some(Vec2::new(0.15, -0.10));
    let plan = Schematic::from_manifest(&scene, 0.).unwrap();
    let mut c = plan.cameras[0].clone();
    let expected = super::camera::camera_footprint(&c, 1.5).unwrap();
    let projection = crate::camera::pinhole::projection(
        PerspectiveProjection {
            fov: c.fov_y,
            aspect_ratio: c.aspect,
            ..default()
        },
        c.principal_point_offset.unwrap(),
    );
    c.calibration = Some(crate::camera::pinhole::calibration(&projection, 640, 480).unwrap());
    c.principal_point_offset = None;
    let actual = super::camera::camera_footprint(&c, 1.5).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(a, b)| Vec3::from_array(*a).distance(Vec3::from_array(b)) < 1e-5));
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
    sample.views[4].calibration = Some(
        crate::calibration::Fisheye {
            coefficients: [0.; 4],
            max_angle: 95_f32.to_radians(),
        }
        .calibration(
            [640, 480],
            crate::calibration::FisheyeFit::Circular,
            [0.; 2],
            1.,
        )
        .unwrap(),
    );
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
    assert_eq!(plan.cameras[1].calibration, sample.views[4].calibration);
    let svg = plan
        .svg(&RenderOptions::default(), &Overlay::default())
        .unwrap();
    assert!(svg.contains("C1 · fisheye 190°"));
    assert_eq!(svg.matches("data-lens=\"fisheye\"").count(), 1);
    assert_eq!(svg.matches("data-lens=\"pinhole\"").count(), 2);
    assert_eq!(
        plan.poses[0].joints,
        sample.human_pose_steps[1][0].bone_positions
    );
    plan.rgba(&RenderOptions::default(), &Overlay::default())
        .unwrap();
    assert!(sample.schematic(2).is_err());
}

#[test]
fn planned_fisheye_and_predicted_lenses_survive_document_roundtrip() {
    let mut manifest = scene();
    let lens = crate::calibration::FisheyeSensor {
        lens: crate::calibration::Fisheye {
            coefficients: [0.; 4],
            max_angle: 100_f32.to_radians(),
        },
        fit: crate::calibration::FisheyeFit::Diagonal,
        anisotropy: 1.,
    };
    manifest.cameras[0].fisheye = Some(lens);
    let plan = Schematic::from_manifest(&manifest, 0.5).unwrap();
    let mut predicted = plan.cameras[0].clone();
    predicted.label = "Predicted <C0>".into();
    let overlay = Overlay {
        cameras: vec![predicted],
        ..default()
    };
    let options = RenderOptions::default();
    let doc = plan.document(options, overlay.clone()).unwrap();
    let restored: Document = serde_json::from_str(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(restored, doc);
    let svg = plan.svg(&options, &overlay).unwrap();
    assert_eq!(svg.matches("data-lens=\"fisheye\"").count(), 2);
    assert!(svg.contains("Predicted &lt;C0&gt; · fisheye 200°"));
    assert!(svg.contains("stroke-dasharray=\"6 4\""));
    manifest.cameras[0].fisheye.as_mut().unwrap().lens.max_angle = f32::NAN;
    assert!(Schematic::from_manifest(&manifest, 0.).is_err());
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
