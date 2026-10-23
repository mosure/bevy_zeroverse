use super::*;
use crate::scene::procedural_indoor::materials::Surface;

#[test]
fn batching_preserves_vertices_tracks_changes_and_restores_original_casters() {
    let mut app = App::new();
    app.add_plugins(TaskPoolPlugin::default());
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.add_message::<AssetEvent<Mesh>>();
    force_for_test(&mut app, true);
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(1., 2., 3.));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let parent = app.world_mut().spawn(Transform::from_xyz(2., 0., 4.)).id();
    let light = app.world_mut().spawn(PointLight::default()).id();
    let layered_light = app
        .world_mut()
        .spawn((SpotLight::default(), RenderLayers::from_layers(&[0, 4])))
        .id();
    let spawn = |world: &mut World| {
        world
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                IndoorSurface(Surface::Wood),
                Transform::IDENTITY,
                GlobalTransform::from_xyz(2., 0., 4.),
                InheritedVisibility::VISIBLE,
                ChildOf(parent),
            ))
            .id()
    };
    let first = spawn(app.world_mut());
    let second = spawn(app.world_mut());
    let excluded = spawn(app.world_mut());
    app.world_mut().entity_mut(excluded).insert(NotShadowCaster);
    app.update();
    let batches: Vec<_> = app
        .world_mut()
        .query_filtered::<(Entity, &Mesh3d), With<ShadowBatch>>()
        .iter(app.world())
        .map(|(e, m)| (e, m.0.clone()))
        .collect();
    assert_eq!(batches.len(), 1);
    let batch = batches[0].0;
    assert_eq!(app.world().resource::<Assets<StandardMaterial>>().len(), 1);
    assert_eq!(
        app.world()
            .get::<MeshMaterial3d<StandardMaterial>>(batch)
            .unwrap()
            .0,
        material
    );
    let mesh_assets = app.world().resource::<Assets<Mesh>>();
    let source = mesh_assets.get(&mesh).unwrap();
    let combined = mesh_assets.get(&batches[0].1).unwrap();
    let VertexAttributeValues::Float32x3(vertices) =
        source.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
    else {
        panic!()
    };
    let VertexAttributeValues::Float32x3(merged) =
        combined.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        merged,
        &vertices.iter().chain(vertices).copied().collect::<Vec<_>>()
    );
    assert_eq!(
        combined.indices().unwrap().len(),
        source.indices().unwrap().len() * 2
    );
    assert!(app
        .world()
        .get::<super::super::RenderOnlyOverlay>(batch)
        .is_some());
    assert!(app
        .world()
        .get::<crate::ovoxel::OvoxelTracked>(batch)
        .is_none());
    assert_eq!(app.world().get::<ChildOf>(batch).unwrap().parent(), parent);
    assert_eq!(
        app.world().get::<GlobalTransform>(batch),
        app.world().get::<GlobalTransform>(first)
    );
    assert!(app.world().get::<NotShadowCaster>(first).is_some());
    assert!(app.world().get::<NotShadowCaster>(second).is_some());
    assert_eq!(
        app.world().get::<RenderLayers>(layered_light),
        Some(&RenderLayers::from_layers(&[0, 4]))
    );
    assert!(app
        .world()
        .get::<VisibilityClass>(batch)
        .unwrap()
        .is_empty());
    // Exercise Bevy's actual CPU camera and light visibility systems. Proxies
    // must be present in shadow queues and absent from camera RGB/prepass queues.
    {
        use bevy::{
            camera::{
                primitives::CascadesFrusta,
                visibility::{CascadesVisibleEntities, VisibleEntities},
                CameraProjection,
            },
            ecs::system::RunSystemOnce,
        };
        use std::any::TypeId;
        let frustum = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_2,
            ..default()
        }
        .compute_frustum(&GlobalTransform::from_xyz(2., 0., 10.));
        for entity in [first, second, excluded] {
            let mut class = VisibilityClass::default();
            class.push(TypeId::of::<Mesh3d>());
            app.world_mut().entity_mut(entity).insert((
                class,
                ViewVisibility::VISIBLE,
                Aabb::from_min_max(Vec3::new(-0.5, -1., -1.5), Vec3::new(0.5, 1., 1.5)),
            ));
        }
        let camera = app
            .world_mut()
            .spawn((Camera::default(), VisibleEntities::default(), frustum))
            .id();
        let directional = app
            .world_mut()
            .spawn((
                DirectionalLight {
                    shadow_maps_enabled: true,
                    ..default()
                },
                CascadesFrusta {
                    frusta: [(camera, vec![frustum])].into_iter().collect(),
                },
                CascadesVisibleEntities::default(),
                ViewVisibility::VISIBLE,
                RenderLayers::from_layers(&[0, 31]),
            ))
            .id();
        app.world_mut()
            .run_system_once(bevy::camera::visibility::check_visibility_cpu_culling)
            .unwrap();
        app.world_mut()
            .run_system_once(bevy::light::check_dir_light_mesh_visibility)
            .unwrap();
        let camera_meshes = app
            .world()
            .get::<VisibleEntities>(camera)
            .unwrap()
            .get(TypeId::of::<Mesh3d>());
        assert!(camera_meshes.contains(&first) && camera_meshes.contains(&second));
        assert!(!camera_meshes.contains(&batch));
        let shadows = &app
            .world()
            .get::<CascadesVisibleEntities>(directional)
            .unwrap()
            .entities[&camera][0]
            .entities;
        assert_eq!(shadows, &[batch]);
        assert_eq!(
            app.world().get::<RenderLayers>(directional),
            Some(&RenderLayers::from_layers(&[0, 31]))
        );
    }
    app.update();
    assert!(
        app.world().get_entity(batch).is_ok(),
        "stable frames must reuse the batch"
    );
    // User edits keep their unrelated layers and update caster participation.
    app.world_mut()
        .entity_mut(layered_light)
        .insert(RenderLayers::layer(4));
    app.update();
    assert_eq!(
        app.world().get::<RenderLayers>(layered_light),
        Some(&RenderLayers::layer(4))
    );
    app.world_mut()
        .entity_mut(layered_light)
        .insert(RenderLayers::from_layers(&[0, 5]));
    app.update();
    assert_eq!(
        app.world().get::<RenderLayers>(layered_light),
        Some(&RenderLayers::from_layers(&[0, 5]))
    );
    // A part acquiring a local pose must leave the rigid batch immediately.
    app.world_mut()
        .get_mut::<Transform>(second)
        .unwrap()
        .translation
        .x = 1.;
    app.update();
    assert!(app.world().get_entity(batch).is_err());
    assert!(app.world().get::<NotShadowCaster>(first).is_none());
    assert!(app.world().get::<NotShadowCaster>(second).is_none());
    assert!(app.world().get::<NotShadowCaster>(excluded).is_some());
    clear(app.world_mut());
    assert!(app.world().get::<RenderLayers>(light).is_none());
    assert_eq!(
        app.world().get::<RenderLayers>(layered_light),
        Some(&RenderLayers::from_layers(&[0, 5]))
    );
}

#[test]
fn capture_policy_requires_active_angular_rgb_and_ignores_readback_transport() {
    let mut app = App::new();
    app.insert_resource(crate::app::BevyZeroverseConfig {
        headless: true,
        editor: false,
        image_copiers: true,
        ..default()
    });
    app.insert_resource(crate::render::RenderMode::Color);
    assert!(!requested(app.world_mut()));
    let camera = app
        .world_mut()
        .spawn((
            Camera::default(),
            crate::camera::lens::LensCamera::test_camera(),
        ))
        .id();
    assert!(requested(app.world_mut()));
    app.init_resource::<crate::sample::CaptureReadiness>();
    assert!(!requested(app.world_mut()));
    app.world_mut()
        .resource_mut::<crate::sample::CaptureReadiness>()
        .blocker = None;
    assert!(requested(app.world_mut()));
    app.world_mut()
        .spawn(crate::io::image_copy::ImageCopier::test_idle());
    assert!(requested(app.world_mut()));
    app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = false;
    assert!(!requested(app.world_mut()));
    app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = true;
    app.insert_resource(crate::render::RenderMode::Depth);
    assert!(!requested(app.world_mut()));
    app.insert_resource(crate::render::RenderMode::Color);
    app.world_mut()
        .resource_mut::<crate::app::BevyZeroverseConfig>()
        .room_schematic = true;
    assert!(!requested(app.world_mut()));
}
