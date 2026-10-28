use super::*;
use crate::calibration::{Fisheye, FisheyeFit};

#[test]
fn schematic_gizmo_scale_changes_only_display_geometry_for_all_lenses() {
    let mut cameras: Vec<_> = [
        FisheyeFit::Circular,
        FisheyeFit::Diagonal,
        FisheyeFit::Elliptical,
    ]
    .into_iter()
    .map(|fit| angular(210., fit))
    .collect();
    let mut pinhole = cameras[0].clone();
    pinhole.calibration = None;
    pinhole.fov_y = 150_f32.to_radians();
    cameras.push(pinhole);
    let projection = Projection {
        schema_version: 1,
        image_size: [768, 768],
        pixels_per_metre: 120.,
        offset: [384., 384.],
    };
    for mut c in cameras {
        for ceiling in [false, true] {
            let mut pose = Transform::from_xyz(2., 2.5, 3.);
            if ceiling {
                pose = pose.looking_to(Vec3::NEG_Y, Vec3::Z);
            }
            c.world_from_view = pose.to_matrix().to_cols_array_2d();
            c.path = vec![[2., 2.5, 3.], [3., 2.5, 3.]];
            let before = c.clone();
            for camera_frustums in [false, true] {
                let options = RenderOptions {
                    camera_frustums,
                    ..Default::default()
                };
                let base = camera_presentation(&c, &projection, &options).unwrap();
                let origin = Vec3::from_array(base.origin);
                for scale in [0.1, 0.5, 2., 4.] {
                    let options = RenderOptions {
                        camera_gizmo_scale: scale,
                        ..options
                    };
                    let g = camera_presentation(&c, &projection, &options).unwrap();
                    assert_eq!(g.origin, base.origin);
                    assert!((g.marker_radius() - base.marker_radius() * scale).abs() < 1e-6);
                    for (a, b) in base.footprint.iter().zip(&g.footprint) {
                        let expected =
                            origin.xz() + (Vec3::from_array(*a).xz() - origin.xz()) * scale;
                        assert!(expected.distance(Vec3::from_array(*b).xz()) < 1e-4);
                    }
                    let axis = origin + (Vec3::from_array(base.axis_end) - origin) * scale;
                    assert!(axis.distance(Vec3::from_array(g.axis_end)) < 1e-4);
                }
                assert_eq!(c, before);
            }
        }
    }
}

fn angular(degrees: f32, fit: FisheyeFit) -> Camera {
    Camera {
        label: "C0".into(),
        world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
        fov_y: 1.,
        aspect: 1.,
        principal_point_offset: None,
        calibration: Some(
            Fisheye {
                coefficients: [0.; 4],
                max_angle: degrees.to_radians() * 0.5,
            }
            .calibration([640, 480], fit, [0.; 2], 1.)
            .unwrap(),
        ),
        path: vec![],
    }
}

#[test]
fn calibration_selects_type_and_overrides_legacy_intrinsics() {
    for degrees in [80., 140., 180., 210.] {
        let mut c = angular(degrees, FisheyeFit::Circular);
        c.fov_y = 3.8;
        c.aspect = -1.;
        c.principal_point_offset = Some(Vec2::splat(2.));
        let gizmo = camera_gizmo(&c, 1.5).unwrap();
        assert!(gizmo.fisheye);
        assert_eq!(gizmo.lens_name(), "fisheye");
        assert_eq!(gizmo.label, format!("C0 · fisheye {degrees:.0}°"));
        assert!(gizmo.footprint.len() > 4);
        let mut nonfinite = c.clone();
        nonfinite.fov_y = f32::NAN;
        assert!(
            camera_gizmo(&nonfinite, 1.5).is_err(),
            "JSON metadata must remain finite"
        );
        c.calibration.as_mut().unwrap().k[0][0] = 0.;
        assert!(
            camera_gizmo(&c, 1.5).is_err(),
            "invalid calibration cannot fall back"
        );
        c.calibration = None;
        assert!(
            camera_gizmo(&c, 1.5).is_err(),
            "legacy lens must be valid without K"
        );
    }
}

#[test]
fn angular_footprints_keep_range_and_rearward_coverage() {
    let length = 1.5;
    let c = angular(210., FisheyeFit::Circular);
    let g = camera_gizmo(&c, length).unwrap();
    assert!(g
        .footprint
        .iter()
        .all(|p| Vec3::from_array(*p).length() <= length + 1e-5));
    assert!(
        g.footprint.iter().any(|p| p[2] > 0.35),
        "rearward rays belong to a >180 degree lens"
    );
    assert!(g.footprint.iter().any(|p| p[2] < -1.49));
    let mut pinhole = c;
    pinhole.calibration = None;
    let g = camera_gizmo(&pinhole, length).unwrap();
    assert!(!g.fisheye);
    assert_eq!(g.label, "C0 · pinhole");
    assert!(g.footprint.iter().all(|p| p[2] <= 0.));
}

#[test]
fn ceiling_fisheye_has_a_disk_and_no_invented_horizontal_heading() {
    let mut c = angular(210., FisheyeFit::Circular);
    c.world_from_view = Transform::from_xyz(0., 3., 0.)
        .looking_at(Vec3::ZERO, Vec3::Z)
        .to_matrix()
        .to_cols_array_2d();
    let g = camera_gizmo(&c, 1.5).unwrap();
    assert_eq!(g.label, "C0 · fisheye 210° · down");
    assert!(
        Vec3::from_array(g.origin)
            .xz()
            .distance(Vec3::from_array(g.axis_end).xz())
            < 1e-6
    );
    for axis in [0, 2] {
        assert!(g.footprint.iter().any(|p| p[axis] > 1.47));
        assert!(g.footprint.iter().any(|p| p[axis] < -1.47));
    }
}

#[test]
fn shifted_elliptical_and_diagonal_calibrations_remain_bounded() {
    for fit in [FisheyeFit::Elliptical, FisheyeFit::Diagonal] {
        let mut c = angular(200., fit);
        c.calibration = Some(
            Fisheye {
                coefficients: [-0.02, 0.003, 0., 0.],
                max_angle: 100_f32.to_radians(),
            }
            .calibration([1024, 512], fit, [0.15, -0.1], 1.12)
            .unwrap(),
        );
        let tf = Transform::from_xyz(2., 1.5, -3.).looking_at(Vec3::new(0., 1., 0.), Vec3::Y);
        c.world_from_view = tf.to_matrix().to_cols_array_2d();
        let g = camera_gizmo(&c, 2.).unwrap();
        assert!(g.footprint.len() > 4);
        assert!(g.footprint.iter().all(|p| {
            let p = Vec3::from_array(*p);
            p.is_finite() && p.xz().distance(tf.translation.xz()) <= 2.00001
        }));
    }
}

#[test]
fn compact_symbols_bound_wide_lenses_without_changing_calibration() {
    let mut cameras = vec![
        angular(210., FisheyeFit::Circular),
        angular(200., FisheyeFit::Diagonal),
    ];
    for aspect in [0.5, 1., 16. / 9.] {
        let mut c = angular(200., FisheyeFit::Circular);
        c.calibration = Some(
            crate::calibration::CameraCalibration::centered_pinhole(
                1024,
                1024,
                145_f32.to_radians(),
                aspect,
            )
            .unwrap(),
        );
        cameras.push(c);
    }
    let projection = Projection {
        schema_version: 1,
        image_size: [1024, 1024],
        pixels_per_metre: 120.,
        offset: [512., 512.],
    };
    for size in [256, 768, 1024] {
        let options = RenderOptions {
            width: size,
            height: size,
            ..Default::default()
        };
        for c in &cameras {
            let before = c.clone();
            let g = camera_presentation(c, &projection, &options).unwrap();
            let origin = Vec2::from_array(projection.project(g.origin));
            for p in g.footprint.iter().chain(std::iter::once(&g.axis_end)) {
                assert!(Vec2::from_array(projection.project(*p)).distance(origin) <= 24.001);
            }
            assert_eq!(g.caption, c.label);
            assert_eq!(*c, before);
            let full = camera_presentation(
                c,
                &projection,
                &RenderOptions {
                    camera_frustums: true,
                    ..options
                },
            )
            .unwrap();
            let raw = camera_gizmo(c, options.frustum_length).unwrap();
            assert_eq!(full.footprint, raw.footprint);
            assert_eq!(full.axis_end, raw.axis_end);
        }
    }
    let g = camera_presentation(&cameras[0], &projection, &RenderOptions::default()).unwrap();
    assert!(
        g.footprint.iter().any(|p| p[2] > 0.),
        "keep rearward fisheye coverage in the symbol"
    );
}

#[test]
fn old_render_options_default_to_compact_symbols() {
    let mut json = serde_json::to_value(RenderOptions::default()).unwrap();
    json.as_object_mut().unwrap().remove("camera_frustums");
    json.as_object_mut().unwrap().remove("camera_gizmo_scale");
    let restored: RenderOptions = serde_json::from_value(json).unwrap();
    assert!(!restored.camera_frustums);
    assert_eq!(restored.camera_gizmo_scale, 1.);
}
