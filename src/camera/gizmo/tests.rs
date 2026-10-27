use super::*;

#[test]
fn curved_prediction_outlines_keep_visible_gaps_instead_of_solid_microsegments() {
    for fit in [
        crate::calibration::FisheyeFit::Circular,
        crate::calibration::FisheyeFit::Diagonal,
        crate::calibration::FisheyeFit::Elliptical,
    ] {
        let c = crate::calibration::Fisheye {
            coefficients: [0.; 4],
            max_angle: 100_f32.to_radians(),
        }
        .calibration([768, 512], fit, [0.; 2], 1.)
        .unwrap();
        let gt = wireframe(&c).unwrap();
        let pred = prediction_wireframe(&c).unwrap();
        let length = |lines: &[[Vec3; 2]]| lines.iter().map(|[a, b]| a.distance(*b)).sum::<f32>();
        let ratio = length(&pred) / length(&gt);
        assert!(
            (0.55..0.8).contains(&ratio),
            "gaps must survive tessellation: {ratio}"
        );
        assert!(pred
            .iter()
            .flatten()
            .all(|p| p.is_finite() && p.length() <= 1.000001));
        assert!(pred.windows(2).any(|p| p[0][1].distance(p[1][0]) > 0.1));
    }
}

#[test]
fn wide_shifted_pinhole_glyphs_preserve_calibration_without_size_explosion() {
    for fov in [20_f32, 70., 120., 169.] {
        for [w, h] in [[512, 512], [1280, 720], [480, 960]] {
            for offset in [[0., 0.], [0.25, -0.2]] {
                let mut c = CameraCalibration::centered_pinhole(
                    w,
                    h,
                    fov.to_radians(),
                    w as f32 / h as f32,
                )
                .unwrap();
                c.k[0][2] += offset[0] * w as f32;
                c.k[1][2] += offset[1] * h as f32;
                let original = c.clone();
                let lines = wireframe(&c).unwrap();
                assert_eq!(c, original);
                assert_eq!(lines.len(), 8);
                let mut depth: Option<f32> = None;
                for (i, pixel) in [
                    [0., 0.],
                    [w as f32, 0.],
                    [w as f32, h as f32],
                    [0., h as f32],
                ]
                .iter()
                .enumerate()
                {
                    let corner = lines[i * 2][1];
                    assert!(corner.is_finite() && corner.length() <= 1.000001);
                    assert!(
                        corner
                            .normalize()
                            .distance(Vec3::from_array(c.ray(*pixel).unwrap()))
                            < 1e-6
                    );
                    if let Some(z) = depth {
                        assert!((corner.z - z).abs() < 1e-6);
                    }
                    depth = Some(corner.z);
                }
            }
        }
    }
}

#[test]
fn zoom_and_viewport_changes_keep_glyphs_below_pixel_and_metric_limits() {
    for size in [
        Vec2::new(1360., 900.),
        Vec2::splat(768.),
        Vec2::new(400., 900.),
    ] {
        for fov in [30_f32, 70., 140.] {
            let projection = Projection::Perspective(PerspectiveProjection {
                fov: fov.to_radians(),
                aspect_ratio: size.x / size.y,
                ..default()
            });
            let clip = projection.get_clip_from_view();
            let screen = |p: Vec3| clip.project_point3(p).xy() * size * 0.5;
            for depth in [0.02, 0.2, 1., 5., 20., 100.] {
                for offset in [Vec2::ZERO, Vec2::new(0.6, -0.4)] {
                    let origin = (offset * depth).extend(-depth);
                    let radius = display_radius(&projection, Mat4::IDENTITY, size, origin);
                    assert!(radius > 0. && radius <= MAX_RADIUS_METRES);
                    for y in -8..=8 {
                        for x in 0..32 {
                            let phi = x as f32 * std::f32::consts::TAU / 32.;
                            let latitude = y as f32 * std::f32::consts::FRAC_PI_2 / 8.;
                            let point = origin
                                + Vec3::new(
                                    phi.cos() * latitude.cos(),
                                    latitude.sin(),
                                    phi.sin() * latitude.cos(),
                                ) * radius;
                            let delta = (screen(point) - screen(origin)).abs();
                            assert!(delta.max_element() <= MAX_RADIUS_PIXELS + 0.01, "{delta:?}");
                        }
                    }
                }
            }
            assert_eq!(
                display_radius(&projection, Mat4::IDENTITY, size, Vec3::Z),
                0.
            );
        }
    }
    let p = Projection::Orthographic(OrthographicProjection {
        area: Rect::from_center_size(Vec2::ZERO, Vec2::splat(4.)),
        ..OrthographicProjection::default_3d()
    });
    let r = display_radius(&p, Mat4::IDENTITY, Vec2::splat(768.), Vec3::NEG_Z);
    assert!((r - 0.1).abs() < 1e-6);
}

#[test]
fn external_prediction_poses_reject_shear_scale_reflections_and_nonfinite_values() {
    let rigid = Transform::from_xyz(2., 3., 4.)
        .looking_to(Vec3::NEG_Y, Vec3::Z)
        .to_matrix();
    assert!(rigid_pose(rigid));
    let mut shear = Mat4::IDENTITY;
    shear.x_axis.y = 0.2;
    let mut projective = Mat4::IDENTITY;
    projective.x_axis.w = 0.1;
    for m in [
        shear,
        projective,
        Mat4::from_scale(Vec3::splat(2.)),
        Mat4::from_scale(Vec3::new(1., -1., 1.)),
        Mat4::from_translation(Vec3::NAN),
    ] {
        assert!(!rigid_pose(m));
    }
}

#[cfg(feature = "viewer")]
#[test]
fn editor_runtime_uses_capture_ids_for_sparse_predictions_and_excludes_source_views() {
    use crate::{
        annotation::schematic::{Camera as Prediction, Overlay},
        app::{editor::schematic::Predictions, BevyZeroverseConfig},
        camera::{
            CaptureCameraIndex, EditorCameraGizmoConfigGroup, EditorCameraMarker, ZeroverseCamera,
        },
    };
    use bevy::{asset::AssetPlugin, gizmos::GizmoAsset, transform::TransformPlugin};
    use std::any::TypeId;
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default(), TransformPlugin))
        .init_asset::<GizmoAsset>()
        .insert_resource(BevyZeroverseConfig {
            gizmos: true,
            gizmos_alpha: 1.,
            camera_grid: false,
            room_schematic: false,
            ..default()
        })
        .insert_gizmo_config(EditorCameraGizmoConfigGroup, config(false))
        .insert_gizmo_config(PredictionCameraGizmoConfigGroup, config(true))
        .add_systems(
            PostUpdate,
            draw_camera_gizmo.after(bevy::transform::TransformSystems::Propagate),
        );
    app.world_mut().spawn((
        Camera3d::default(),
        EditorCameraMarker::default(),
        Transform::default(),
    ));
    // Deliberately not index order; nonlinear source views have no logical camera.
    app.world_mut()
        .spawn((Camera3d::default(), Transform::from_xyz(50., 0., -5.)));
    for (id, x) in [(7, -1.), (2, 1.)] {
        app.world_mut().spawn((
            Camera3d::default(),
            ZeroverseCamera::default(),
            CaptureCameraIndex(id),
            Transform::from_xyz(x, 0., -4.),
        ));
    }
    let c = CameraCalibration::centered_pinhole(512, 512, 1., 1.).unwrap();
    let prediction = Prediction {
        label: "Pred C7 · pinhole".into(),
        world_from_view: Mat4::from_translation(Vec3::new(-1., 0.2, -4.)).to_cols_array_2d(),
        fov_y: 1.,
        aspect: 1.,
        principal_point_offset: None,
        calibration: Some(c),
        path: vec![],
    };
    app.insert_resource(Predictions(Overlay {
        cameras: vec![prediction],
        ..default()
    }));
    app.update();
    let assets = app.world().resource::<Assets<GizmoAsset>>();
    for group in [
        TypeId::of::<EditorCameraGizmoConfigGroup>(),
        TypeId::of::<PredictionCameraGizmoConfigGroup>(),
    ] {
        let buffer = assets
            .iter()
            .find(|(_, a)| a.config_typeid() == group)
            .unwrap()
            .1
            .buffer();
        assert!(buffer.list_colors.contains(&color(7).to_linear()));
        if group == TypeId::of::<EditorCameraGizmoConfigGroup>() {
            assert!(buffer.list_colors.contains(&color(2).to_linear()));
        } else {
            assert!(buffer
                .list_colors
                .iter()
                .all(|c| *c == color(7).to_linear()));
        }
        assert!(buffer
            .list_positions
            .iter()
            .all(|p| p.x.abs() < 2. && p.z < -3.));
    }
    // Pose changes must not alter calibration; cached outlines follow motion.
    app.world_mut().resource_mut::<Predictions>().0.cameras[0].world_from_view =
        Mat4::from_translation(Vec3::new(-1., 1., -4.)).to_cols_array_2d();
    app.update();
    let assets = app.world().resource::<Assets<GizmoAsset>>();
    let p = assets
        .iter()
        .find(|(_, a)| a.config_typeid() == TypeId::of::<PredictionCameraGizmoConfigGroup>())
        .unwrap()
        .1
        .buffer();
    assert!(p.list_positions.iter().all(|p| p.y > 0.65));
}
