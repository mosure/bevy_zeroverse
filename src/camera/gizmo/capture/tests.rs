use super::*;

fn scene() -> App {
    let mut app = App::new();
    app.init_resource::<BevyZeroverseConfig>()
        .init_resource::<Assets<GizmoAsset>>()
        .insert_resource(RenderMode::Color)
        .add_systems(Update, sync);
    app
}

fn receiver(world: &mut World, index: usize) -> Entity {
    world
        .spawn((
            Camera3d::default(),
            ZeroverseCamera::default(),
            CaptureCameraIndex(index),
            GlobalTransform::from_xyz(index as f32, 1., 0.),
        ))
        .id()
}

fn drawings(world: &mut World) -> Vec<(RenderLayers, Vec<Vec3>)> {
    let mut query = world.query::<(&Gizmo, &RenderLayers)>();
    let assets = world.resource::<Assets<GizmoAsset>>();
    query
        .iter(world)
        .map(|(gizmo, layers)| {
            let data = assets.get(&gizmo.handle).unwrap().buffer().buffer();
            (
                layers.clone(),
                data.list_positions
                    .iter()
                    .chain(
                        data.strip_positions
                            .iter()
                            .filter(|p| !(p.x.is_nan() && p.y.is_nan() && p.z.is_nan())),
                    )
                    .copied()
                    .collect(),
            )
        })
        .collect()
}

#[test]
fn calibrated_frustums_are_peer_only_and_restore_existing_layers() {
    let mut app = scene();
    let w = app.world_mut();
    let a = receiver(w, 3);
    let b = receiver(w, 9);
    let original = RenderLayers::from_layers(&[0, 22, 64]);
    let mut angular = lens::LensCamera::test_camera();
    angular.calibration = bevy_zeroverse_capture::calibration::Fisheye {
        coefficients: [0.; 4],
        max_angle: 1.65,
    }
    .calibration(
        [256, 256],
        bevy_zeroverse_capture::calibration::FisheyeFit::Circular,
        [0.; 2],
        1.,
    )
    .unwrap();
    w.entity_mut(b).insert((original.clone(), angular));
    let editor = w
        .spawn((
            Camera3d::default(),
            crate::camera::ProcessedEditorCameraMarker,
        ))
        .id();
    app.update();
    assert!(drawings(app.world_mut()).is_empty());
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .rgb_gizmos
        .camera_frustums = true;
    app.update();
    let w = app.world_mut();
    let visible = drawings(w);
    assert_eq!(visible.len(), 2);
    for (observer, subject, expected) in [(a, b, 9.0), (b, a, 3.0)] {
        let layers = w.get::<RenderLayers>(observer).unwrap();
        let subject_layers = w.get::<RenderLayers>(subject).unwrap();
        let matching: Vec<_> = visible
            .iter()
            .filter(|(glyph, _)| glyph.intersects(layers))
            .collect();
        assert_eq!(matching.len(), 1);
        let (glyph, points) = matching[0];
        assert!(!glyph.intersects(subject_layers));
        assert!(!glyph.intersects(&original));
        assert!(
            !points.is_empty()
                && points
                    .iter()
                    .all(|p| (p.x - expected).abs() <= MAX_RADIUS_METRES + 0.001)
        );
    }
    assert!(w.get::<RenderLayers>(editor).is_none());
    w.insert_resource(RenderMode::Depth);
    app.update();
    assert!(drawings(app.world_mut()).is_empty());
    assert!(app.world().get::<RenderLayers>(a).is_none());
    assert_eq!(app.world().get::<RenderLayers>(b), Some(&original));
}

#[test]
fn boxes_and_paths_switch_independently_and_single_camera_sees_its_path() {
    let mut app = scene();
    let w = app.world_mut();
    let owner = receiver(w, 0);
    let origin = w.spawn(GlobalTransform::from_xyz(2., 3., 4.)).id();
    w.entity_mut(owner).insert(ChildOf(origin));
    w.get_mut::<ZeroverseCamera>(owner).unwrap().trajectory = TrajectorySampler::WaypointPath {
        positions: vec![Vec3::ZERO, Vec3::X],
        rotations: vec![Quat::IDENTITY; 2],
    };
    w.spawn(ObjectObb {
        center: Vec3::new(-3., 1., 0.),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
        class_name: "chair".into(),
    });
    w.resource_mut::<BevyZeroverseConfig>().rgb_gizmos = RgbGizmos {
        camera_frustums: true,
        camera_trajectories: true,
        ..default()
    };
    app.update();
    let path = drawings(app.world_mut());
    assert_eq!(path.len(), 1);
    assert_eq!(path[0].1.first(), Some(&Vec3::new(2., 3., 4.)));
    assert_eq!(path[0].1.last(), Some(&Vec3::new(3., 3., 4.)));
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .rgb_gizmos = RgbGizmos {
        object_obbs: true,
        ..default()
    };
    app.update();
    let bounds = drawings(app.world_mut());
    assert_eq!(bounds.len(), 1);
    assert!(bounds[0].1.iter().any(|p| p.x < -3.));
    app.world_mut()
        .entity_mut(owner)
        .remove::<ZeroverseCamera>();
    app.update();
    assert!(app.world().get::<RenderLayers>(owner).is_none());
    assert!(drawings(app.world_mut()).is_empty());
}
