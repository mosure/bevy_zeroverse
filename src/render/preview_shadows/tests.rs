use super::*;
use crate::scene::procedural_indoor::materials::Surface;

#[test]
fn batching_preserves_vertices_tracks_changes_and_restores_original_casters() {
    let mut app = App::new();
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
        Some(&RenderLayers::from_layers(&[0, 4, LAYER]))
    );
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
        Some(&RenderLayers::from_layers(&[0, 5, LAYER]))
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
    assert!(app.world().get::<ShadowLight>(light).is_none());
    assert_eq!(
        app.world().get::<RenderLayers>(layered_light),
        Some(&RenderLayers::from_layers(&[0, 5]))
    );
}

#[test]
fn private_layer_collision_restores_unbatched_rendering() {
    let mut app = App::new();
    app.init_resource::<Assets<Mesh>>();
    app.init_resource::<Assets<StandardMaterial>>();
    app.add_message::<AssetEvent<Mesh>>();
    force_for_test(&mut app, true);
    let original = app.world_mut().spawn((ShadowSource, NotShadowCaster)).id();
    app.world_mut().resource_mut::<State>().inputs.push(Input {
        entity: original,
        mesh: Handle::<Mesh>::default().id(),
        key: Key {
            parent: original,
            cull: 0,
            double_sided: false,
            depth_bias: 0,
        },
    });
    app.world_mut()
        .spawn((Camera3d::default(), RenderLayers::layer(LAYER)));
    app.update();
    assert!(app.world().get::<NotShadowCaster>(original).is_none());
    assert!(app.world().resource::<State>().inputs.is_empty());
}
