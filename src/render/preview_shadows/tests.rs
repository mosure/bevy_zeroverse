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
    // A preview can enter capture mode in the same app. Existing proxies must
    // disappear and original shadow casters must be restored before capture.
    app.world_mut()
        .get_mut::<Transform>(second)
        .unwrap()
        .translation
        .x = 0.;
    app.update();
    assert_eq!(active_batches(app.world_mut()), 1);
    app.insert_resource(crate::app::BevyZeroverseConfig {
        headless: false,
        image_copiers: false,
        ..default()
    });
    app.insert_resource(crate::render::RenderMode::Color);
    app.world_mut().spawn((
        Camera::default(),
        crate::camera::lens::LensCamera::test_camera(),
    ));
    app.world_mut().resource_mut::<State>().force = None;
    assert!(requested(app.world_mut()));
    app.update();
    assert_eq!(active_batches(app.world_mut()), 1);
    let retained = app
        .world_mut()
        .query_filtered::<Entity, With<ShadowBatch>>()
        .single(app.world())
        .unwrap();
    // Camera idling must not rebuild/reallocate shadow meshes. A subsequent
    // capture request must still restore originals, even while the rig is idle.
    for mut camera in app
        .world_mut()
        .query::<&mut Camera>()
        .iter_mut(app.world_mut())
    {
        camera.is_active = false;
    }
    app.update();
    assert!(app.world().get::<ShadowBatch>(retained).is_some());
    app.world_mut()
        .resource_mut::<crate::app::BevyZeroverseConfig>()
        .image_copiers = true;
    app.update();
    assert_eq!(active_batches(app.world_mut()), 0);
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
fn replay_order_preserves_batch_triangles_material_and_budget_admission() {
    // More geometry than the global budget: recycling parent IDs must not
    // switch which group receives a batch, nor which material represents it.
    fn build(reverse: bool) -> (Vec<[f32; 3]>, Vec<usize>, Color, Vec<u64>) {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default());
        app.init_resource::<Assets<Mesh>>();
        app.init_resource::<Assets<StandardMaterial>>();
        app.add_message::<AssetEvent<Mesh>>();
        force_for_test(&mut app, true);
        let group_order = if reverse { [1, 0] } else { [0, 1] };
        for group in group_order {
            let parent = app.world_mut().spawn(Transform::IDENTITY).id();
            let part_order = if reverse { [1, 0] } else { [0, 1] };
            for part in part_order {
                let order = group * 2 + part;
                let triangle = [
                    [order as f32, 0., 0.],
                    [order as f32, 1., 0.],
                    [order as f32, 0., 1.],
                ];
                let positions = triangle
                    .into_iter()
                    .cycle()
                    .take(150_000)
                    .collect::<Vec<_>>();
                let mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[1., 0., 0.]; 150_000]);
                let mesh = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
                let material = app
                    .world_mut()
                    .resource_mut::<Assets<StandardMaterial>>()
                    .add(StandardMaterial {
                        base_color: Color::srgb(order as f32 / 4., 0., 0.),
                        ..default()
                    });
                app.world_mut().spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    IndoorSurface(Surface::Wood),
                    crate::render::ground_truth::AnnotationOrder(order),
                    Transform::IDENTITY,
                    GlobalTransform::IDENTITY,
                    InheritedVisibility::VISIBLE,
                    ChildOf(parent),
                ));
            }
        }
        app.update();
        let (mesh, material) = app
            .world_mut()
            .query_filtered::<(&Mesh3d, &MeshMaterial3d<StandardMaterial>), With<ShadowBatch>>()
            .single(app.world())
            .unwrap();
        let mesh = app.world().resource::<Assets<Mesh>>().get(&mesh.0).unwrap();
        let VertexAttributeValues::Float32x3(positions) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!()
        };
        let positions = positions.clone();
        let indices = mesh.indices().unwrap().iter().collect();
        let color = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(&material.0)
            .unwrap()
            .base_color;
        let mut sources = app
            .world_mut()
            .query_filtered::<&crate::render::ground_truth::AnnotationOrder, With<ShadowSource>>()
            .iter(app.world())
            .map(|o| o.0)
            .collect::<Vec<_>>();
        sources.sort_unstable();
        (positions, indices, color, sources)
    }
    let forward = build(false);
    let reversed = build(true);
    assert_eq!(forward.3, [0, 1]);
    assert_eq!(forward, reversed);
}

#[test]
fn batching_requires_ready_angular_rgb_and_qualified_direct_capture() {
    let mut app = App::new();
    app.insert_resource(crate::app::BevyZeroverseConfig {
        headless: false,
        image_copiers: false,
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
    let copier = app
        .world_mut()
        .spawn(crate::io::image_copy::ImageCopier::test_idle())
        .id();
    assert!(!requested(app.world_mut()));
    app.world_mut()
        .entity_mut(copier)
        .insert((Camera::default(), bevy::render::view::NoIndirectDrawing));
    assert_eq!(requested(app.world_mut()), !cfg!(target_arch = "wasm32"));
    // One indirect capture camera requires restoring all original casters.
    let indirect = app
        .world_mut()
        .spawn((
            Camera::default(),
            crate::io::image_copy::ImageCopier::test_idle(),
        ))
        .id();
    assert!(!requested(app.world_mut()));
    app.world_mut().despawn(indirect);
    app.world_mut().despawn(copier);
    assert!(requested(app.world_mut()));
    // Headless warm-up without an installed qualified copier retains originals.
    for (headless, image_copiers) in [(true, false), (false, true), (true, true)] {
        let mut config = app
            .world_mut()
            .resource_mut::<crate::app::BevyZeroverseConfig>();
        config.headless = headless;
        config.image_copiers = image_copiers;
        assert!(!requested(app.world_mut()));
    }
    let mut config = app
        .world_mut()
        .resource_mut::<crate::app::BevyZeroverseConfig>();
    config.headless = false;
    config.image_copiers = false;
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
