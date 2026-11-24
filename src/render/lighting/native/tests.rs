use super::*;
use crate::{camera::ProcessedEditorCameraMarker, scene::SceneLoadedEvent};
use bevy::core_pipeline::prepass::{DeferredPrepass, DepthPrepass};
use bevy_solari::realtime::SolariLighting;

#[test]
fn preview_releases_only_its_camera_components() {
    let mut app = App::new();
    app.init_resource::<SolariPreviewStatus>()
        .add_message::<SceneLoadedEvent>()
        .add_systems(Update, (views::release, views::synchronize).chain());
    let entity = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            ProcessedEditorCameraMarker,
            DepthPrepass,
            Msaa::Off,
        ))
        .id();
    app.world_mut().resource_mut::<SolariPreviewStatus>().active = true;
    app.update();
    assert!(app.world().get::<SolariLighting>(entity).is_some());
    assert!(app.world().get::<DeferredPrepass>(entity).is_some());
    app.world_mut()
        .get_mut::<Camera>(entity)
        .unwrap()
        .computed
        .target_info = Some(bevy::camera::RenderTargetInfo {
        physical_size: UVec2::splat(16),
        ..default()
    });
    let mut status = app.world_mut().resource_mut::<SolariPreviewStatus>();
    status.max_reservoir_pixels = 256;
    status.temporal_reuse = true;
    app.update();
    assert!(app.world().get::<SolariLighting>(entity).unwrap().restir);
    app.world_mut()
        .resource_mut::<SolariPreviewStatus>()
        .max_reservoir_pixels = 255;
    app.update();
    assert!(!app.world().get::<SolariLighting>(entity).unwrap().restir);
    app.world_mut().resource_mut::<SolariPreviewStatus>().active = false;
    app.update();
    assert!(app.world().get::<DepthPrepass>(entity).is_some());
    assert!(app.world().get::<DeferredPrepass>(entity).is_none());
    assert!(app.world().get::<SolariLighting>(entity).is_none());
}

#[test]
fn only_current_opaque_rigid_geometry_enters_the_traced_scene() {
    use super::meshes::{filter_new_blas, synchronize, TracedAssets};
    use crate::render::semantic::SemanticLabel;
    use bevy::{
        asset::RenderAssetUsages,
        material::OpaqueRendererMethod,
        mesh::{Indices, PrimitiveTopology},
    };
    use bevy_solari::scene::RaytracingMesh3d;
    fn triangle() -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0., 0., 1.]; 3])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0., 0.], [1., 0.], [0., 1.]])
        .with_inserted_attribute(Mesh::ATTRIBUTE_TANGENT, vec![[1., 0., 0., 1.]; 3])
        .with_inserted_indices(Indices::U32(vec![0, 1, 2]))
    }
    let mut app = App::new();
    app.init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<SolariPreviewStatus>()
        .init_resource::<TracedAssets>()
        .add_message::<AssetEvent<Mesh>>()
        .add_message::<AssetEvent<StandardMaterial>>()
        .add_systems(Update, (synchronize, filter_new_blas).chain());
    let rigid = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(triangle());
    let staged = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(triangle());
    app.world_mut()
        .write_message(AssetEvent::Added { id: staged.id() });
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let glass = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            specular_transmission: 0.7,
            ..default()
        });
    let mut part = |mesh: &Handle<Mesh>, mat: &Handle<StandardMaterial>, vis| {
        app.world_mut()
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(mat.clone()),
                SemanticLabel::Wall,
                vis,
            ))
            .id()
    };
    let wall = part(&rigid, &material, InheritedVisibility::VISIBLE);
    let window = part(&rigid, &glass, InheritedVisibility::VISIBLE);
    let pending = part(&staged, &material, InheritedVisibility::HIDDEN);
    app.world_mut().resource_mut::<SolariPreviewStatus>().active = true;
    app.update();
    assert!(app.world().get::<RaytracingMesh3d>(wall).is_some());
    assert!(app.world().get::<RaytracingMesh3d>(window).is_none());
    assert!(app.world().get::<RaytracingMesh3d>(pending).is_none());
    assert!(
        !app.world()
            .resource::<Assets<Mesh>>()
            .get(&staged)
            .unwrap()
            .enable_raytracing
    );
    assert_eq!(
        app.world()
            .resource::<Assets<StandardMaterial>>()
            .get(&material)
            .unwrap()
            .opaque_render_method,
        OpaqueRendererMethod::Deferred
    );
    app.world_mut().entity_mut(wall).remove::<SemanticLabel>();
    app.update();
    assert!(app.world().get::<RaytracingMesh3d>(wall).is_none());
    assert_eq!(
        app.world()
            .resource::<SolariPreviewStatus>()
            .traced_instances,
        0
    );
    app.world_mut()
        .entity_mut(pending)
        .insert(InheritedVisibility::VISIBLE);
    app.update();
    assert!(app.world().get::<RaytracingMesh3d>(pending).is_some());
    app.world_mut().resource_mut::<SolariPreviewStatus>().active = false;
    app.update();
    assert!(app.world().get::<RaytracingMesh3d>(pending).is_none());
    assert!(
        !app.world()
            .resource::<Assets<Mesh>>()
            .get(&staged)
            .unwrap()
            .enable_raytracing
    );
    assert_eq!(
        app.world()
            .resource::<Assets<StandardMaterial>>()
            .get(&material)
            .unwrap()
            .opaque_render_method,
        OpaqueRendererMethod::Auto
    );
}

#[test]
fn scene_and_annotation_switches_select_lighting_in_the_present_frame() {
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        preview_lighting: PreviewLighting::SolariRestir,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    })
    .insert_resource(RenderMode::Color)
    .init_resource::<SolariPreviewStatus>()
    .add_systems(PostUpdate, select);
    app.update();
    assert!(app.world().resource::<SolariPreviewStatus>().temporal_reuse);
    for mode in [
        RenderMode::Depth,
        RenderMode::Normal,
        RenderMode::Semantic,
        RenderMode::CoVisibility,
        RenderMode::OpticalFlow,
    ] {
        app.insert_resource(mode);
        app.update();
        assert!(!app.world().resource::<SolariPreviewStatus>().active);
    }
    app.insert_resource(RenderMode::Color);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .room_schematic = true;
    app.update();
    assert!(!app.world().resource::<SolariPreviewStatus>().active);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .room_schematic = false;
    app.update();
    assert!(app.world().resource::<SolariPreviewStatus>().active);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .preview_lighting = PreviewLighting::Raster;
    app.update();
    assert!(!app.world().resource::<SolariPreviewStatus>().active);
}
