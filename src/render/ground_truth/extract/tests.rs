use super::*;
use bevy::{
    ecs::message::Messages,
    mesh::skinning::SkinnedMeshInverseBindposes,
    render::{ExtractSchedule, MainWorld},
};

#[path = "oracle.rs"]
mod oracle;

struct Extraction {
    main: World,
    current: World,
    original: World,
    current_schedule: Schedule,
    original_schedule: Schedule,
    scratch: Option<MainWorld>,
    camera: Entity,
}

impl Extraction {
    fn new() -> Self {
        let mut main = World::new();
        main.init_resource::<Assets<Mesh>>();
        main.init_resource::<Assets<StandardMaterial>>();
        main.init_resource::<Assets<SkinnedMeshInverseBindposes>>();
        main.init_resource::<Messages<AssetEvent<Mesh>>>();
        let mut images = Assets::default();
        let camera = main
            .spawn(GroundTruthCamera::new(&mut images, UVec2::splat(8)))
            .id();
        let mut current = World::new();
        current.init_resource::<GeometryCache>();
        current.init_resource::<ExtractedGeometry>();
        let mut original = World::new();
        original.init_resource::<GeometryCache>();
        original.init_resource::<ExtractedGeometry>();
        let mut current_schedule = Schedule::new(ExtractSchedule);
        current_schedule.add_systems(extract_geometry);
        let mut original_schedule = Schedule::new(ExtractSchedule);
        original_schedule.add_systems(oracle::extract_geometry_original);
        Self {
            main,
            current,
            original,
            current_schedule,
            original_schedule,
            scratch: Some(MainWorld::default()),
            camera,
        }
    }

    fn run_current(&mut self) {
        let mut main_world = self.scratch.take().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.current.insert_resource(main_world);
        self.current_schedule.run(&mut self.current);
        let mut main_world = self.current.remove_resource::<MainWorld>().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.scratch = Some(main_world);
    }

    fn run_original(&mut self) {
        let mut main_world = self.scratch.take().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.original.insert_resource(main_world);
        self.original_schedule.run(&mut self.original);
        let mut main_world = self.original.remove_resource::<MainWorld>().unwrap();
        std::mem::swap(&mut self.main, &mut *main_world);
        self.scratch = Some(main_world);
    }

    fn check(&mut self) {
        self.run_current();
        self.run_original();
        same_geometry(
            self.current.resource::<ExtractedGeometry>(),
            self.original.resource::<ExtractedGeometry>(),
        );
        assert_eq!(
            self.main
                .get::<GroundTruthCamera>(self.camera)
                .map(GroundTruthCamera::failure),
            self.current
                .get_resource::<ExtractedGeometry>()
                .map(|g| g.failure.clone()),
        );
        self.main.clear_trackers();
    }

    fn geometry(&self) -> &ExtractedGeometry {
        self.current.resource()
    }

    fn event(&mut self, event: AssetEvent<Mesh>) {
        self.main
            .resource_mut::<Messages<AssetEvent<Mesh>>>()
            .write(event);
    }

    fn object(
        &mut self,
        mesh: Mesh,
        cull: Option<Face>,
        layers: RenderLayers,
        transform: Mat4,
    ) -> (Entity, Handle<Mesh>, Handle<StandardMaterial>) {
        let mesh = self.main.resource_mut::<Assets<Mesh>>().add(mesh);
        let material = self
            .main
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                cull_mode: cull,
                ..default()
            });
        let entity = self
            .main
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material.clone()),
                GlobalTransform::from(transform),
                InheritedVisibility::VISIBLE,
                SemanticLabel::Chair,
                layers,
            ))
            .id();
        (entity, mesh, material)
    }
}

fn triangle(indices: Option<Indices>) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[-0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
    if let Some(indices) = indices {
        mesh.insert_indices(indices);
    }
    mesh
}

fn same_geometry(current: &ExtractedGeometry, original: &ExtractedGeometry) {
    assert_eq!(current.generation, original.generation, "generation");
    assert_eq!(current.failure, original.failure, "failure");
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&current.vertices),
        bytemuck::cast_slice::<_, u8>(&original.vertices),
        "vertex bits/order"
    );
    assert_eq!(current.indices, original.indices, "index bits/order");
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&current.instances),
        bytemuck::cast_slice::<_, u8>(&original.instances),
        "instance bits/order"
    );
    let batches = |g: &ExtractedGeometry| {
        g.batches
            .iter()
            .map(|b| (b.indices.clone(), b.cull, b.layers.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        batches(current),
        batches(original),
        "batch order/layers/cull"
    );
    let topology = |g: &ExtractedGeometry| {
        g.topology
            .iter()
            .map(|t| (t.entity, t.mesh, t.vertices.clone(), t.indices_hash))
            .collect::<Vec<_>>()
    };
    assert_eq!(topology(current), topology(original), "flow topology");
}

#[test]
fn ground_truth_span_assembly_matches_original_indices_batches_and_flow_topology() {
    let mut x = Extraction::new();
    for i in 0..18 {
        let indices = match i % 3 {
            0 => None,
            1 => Some(Indices::U16(vec![2, 1, 0])),
            _ => Some(Indices::U32(vec![1, 2, 0, 2, 1, 0])),
        };
        x.object(
            triangle(indices),
            [None, Some(Face::Front), Some(Face::Back)][i % 3],
            RenderLayers::layer((i / 3) % 3),
            Mat4::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
        );
    }
    let (hidden, _, _) = x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    x.main
        .entity_mut(hidden)
        .insert(InheritedVisibility::HIDDEN);
    let (overlay, _, _) = x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    x.main
        .entity_mut(overlay)
        .insert(crate::render::RenderOnlyOverlay);
    x.main.get_mut::<GroundTruthCamera>(x.camera).unwrap().flow = Some(Handle::default());
    x.check();
    assert_eq!(x.geometry().instances.len(), 18);
    assert_eq!(x.geometry().batches.len(), 9);
    assert_eq!(x.geometry().topology.len(), 18);
    let generation = x.geometry().generation;
    x.check();
    assert_eq!(x.geometry().generation, generation);
    x.main.get_mut::<GroundTruthCamera>(x.camera).unwrap().flow = None;
    x.check();
    assert_eq!(x.geometry().generation, generation + 1);
    assert!(x.geometry().topology.is_empty());
}

#[test]
fn ground_truth_exact_matrix_reuse_matches_original_after_live_component_changes() {
    let mut x = Extraction::new();
    let matrix = Mat4::from_scale_rotation_translation(
        Vec3::new(0.7, 1.8, -2.3),
        Quat::from_rotation_y(0.71),
        Vec3::new(2.0, -0.0, 3.0),
    );
    let (entity, _, material) = x.object(
        triangle(None),
        Some(Face::Back),
        RenderLayers::default(),
        matrix,
    );
    x.check();
    let generation = x.geometry().generation;
    for i in 0..12 {
        // Both unchanged and changing transforms exercise the existing-buffer path.
        let transform = if i % 3 == 0 {
            matrix
        } else {
            Mat4::from_translation(Vec3::new(i as f32, 0.0, 0.0)) * matrix
        };
        x.main
            .entity_mut(entity)
            .insert((GlobalTransform::from(transform), SemanticLabel::Table));
        x.check();
        assert_eq!(x.geometry().generation, generation);
        assert_eq!(
            x.geometry().instances[0].semantic,
            semantic_id(&SemanticLabel::Table)
        );
    }
    x.main
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&material)
        .unwrap()
        .cull_mode = Some(Face::Front);
    x.check();
    assert_eq!(x.geometry().generation, generation + 1);
    x.main.entity_mut(entity).insert(RenderLayers::layer(5));
    x.check();
    assert_eq!(x.geometry().generation, generation + 2);
    x.main
        .entity_mut(entity)
        .remove::<MeshMaterial3d<StandardMaterial>>();
    x.main.entity_mut(entity).insert(DisabledPbrMaterial {
        cull_mode: Some(Face::Back),
        ..default()
    });
    x.check();
    assert_eq!(x.geometry().batches[0].cull, 1);
    x.main
        .entity_mut(entity)
        .insert(InheritedVisibility::HIDDEN);
    x.check();
    assert!(x.geometry().vertices.is_empty());
    x.main
        .entity_mut(entity)
        .insert(InheritedVisibility::VISIBLE);
    x.check();
    assert_eq!(x.geometry().instances.len(), 1);
    x.main.entity_mut(entity).despawn();
    x.object(triangle(None), None, RenderLayers::default(), matrix);
    x.check();
}

#[test]
fn ground_truth_matrix_bits_preserve_signed_zero_and_nonfinite_failure_recovery() {
    let mut positive = Mat4::IDENTITY.to_cols_array_2d();
    let mut negative = positive;
    negative[3][0] = -0.0;
    assert!(!same_matrix_bits(&positive, &negative));
    positive[0][0] = f32::from_bits(0x7fc0_0123);
    negative = positive;
    assert!(same_matrix_bits(&positive, &negative));
    negative[0][0] = f32::from_bits(0x7fc0_0456);
    assert!(!same_matrix_bits(&positive, &negative));
    let mut x = Extraction::new();
    let (entity, _, _) = x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    x.check();
    for zero in [-0.0, 0.0, -0.0] {
        x.main
            .entity_mut(entity)
            .insert(GlobalTransform::from_translation(Vec3::new(zero, 0.0, 0.0)));
        x.check();
    }
    for matrix in [
        Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0)),
        Mat4::from_translation(Vec3::new(f32::NAN, 0.0, 0.0)),
        Mat4::from_translation(Vec3::new(f32::INFINITY, 0.0, 0.0)),
    ] {
        x.main
            .entity_mut(entity)
            .insert(GlobalTransform::from(matrix));
        x.check();
        assert!(x
            .geometry()
            .failure
            .as_ref()
            .unwrap()
            .contains("singular/nonfinite transform"));
        assert!(x.geometry().instances.is_empty());
        x.main
            .entity_mut(entity)
            .insert(GlobalTransform::from(Mat4::IDENTITY));
        x.check();
        assert!(x.geometry().failure.is_none());
        assert_eq!(x.geometry().instances.len(), 1);
    }
}

#[test]
fn ground_truth_mesh_events_invalidate_live_assets_and_retry_failed_geometry() {
    let mut x = Extraction::new();
    let (_, mesh, _) = x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    let unrelated = x.main.resource_mut::<Assets<Mesh>>().add(triangle(None));
    x.check();
    let generation = x.geometry().generation;
    x.event(AssetEvent::Modified { id: unrelated.id() });
    x.check();
    assert_eq!(x.geometry().generation, generation);
    x.main
        .resource_mut::<Assets<Mesh>>()
        .get_mut(&mesh)
        .unwrap()
        .insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
        );
    x.event(AssetEvent::Modified { id: mesh.id() });
    x.check();
    assert_eq!(x.geometry().generation, generation + 1);
    assert_eq!(x.geometry().vertices[0].position[2], 2.0);
    let removed = x.main.resource_mut::<Assets<Mesh>>().remove(&mesh).unwrap();
    x.event(AssetEvent::Removed { id: mesh.id() });
    x.check();
    assert!(x
        .geometry()
        .failure
        .as_ref()
        .unwrap()
        .contains("unavailable"));
    x.main
        .resource_mut::<Assets<Mesh>>()
        .insert(mesh.id(), removed)
        .unwrap();
    x.event(AssetEvent::Added { id: mesh.id() });
    x.check();
    assert!(x.geometry().failure.is_none());
    assert_eq!(x.geometry().vertices[0].position[2], 2.0);
}

#[test]
fn ground_truth_invalid_meshes_preserve_original_partial_geometry_and_errors() {
    let bad = vec![
        triangle(Some(Indices::U32(vec![0, 1, 3]))),
        triangle(Some(Indices::U16(vec![0, 1]))),
        triangle(None).with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0, 0.0, 0.0]; 2]),
        triangle(None).with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 0.0]; 3]),
        triangle(None)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[f32::INFINITY, 0.0, 1.0]; 3]),
        triangle(None)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[f32::NAN, 0.0, 0.0]; 3]),
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default()),
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        ),
        triangle(None).with_morph_targets(Vec::new()),
    ];
    for mesh in bad {
        let mut x = Extraction::new();
        let mut objects = Vec::new();
        for _ in 0..3 {
            objects.push(x.object(
                triangle(None),
                None,
                RenderLayers::default(),
                Mat4::IDENTITY,
            ));
        }
        objects.sort_by_key(|(entity, ..)| entity.to_bits());
        let handle = &objects[1].1;
        x.main
            .resource_mut::<Assets<Mesh>>()
            .insert(handle.id(), mesh)
            .unwrap();
        x.main.get_mut::<GroundTruthCamera>(x.camera).unwrap().flow = Some(Handle::default());
        x.check();
        assert!(x.geometry().failure.is_some());
        assert_eq!(
            x.geometry().vertices.len(),
            3,
            "only the valid prefix is retained"
        );
        assert_eq!(x.geometry().indices.len(), 3);
        assert_eq!(x.geometry().topology.len(), 1);
        x.main
            .resource_mut::<Assets<Mesh>>()
            .insert(handle.id(), triangle(None))
            .unwrap();
        x.check();
        assert!(x.geometry().failure.is_none());
        assert_eq!(x.geometry().vertices.len(), 9);
    }
}

#[test]
fn ground_truth_skin_capture_joint_and_bind_changes_match_original_baked_bits() {
    let mut x = Extraction::new();
    let joint = x
        .main
        .spawn(GlobalTransform::from_translation(Vec3::new(0.3, 0.7, -0.1)))
        .id();
    let poses = x
        .main
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .add(vec![Mat4::IDENTITY]);
    let mesh = triangle(Some(Indices::U16(vec![0, 1, 2])))
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(vec![[0; 4]; 3]),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, vec![[1.0, 0.0, 0.0, 0.0]; 3]);
    let (entity, _, _) = x.object(
        mesh,
        None,
        RenderLayers::default(),
        Mat4::from_scale(Vec3::splat(3.0)),
    );
    x.main.entity_mut(entity).insert(SkinnedMesh {
        inverse_bindposes: poses.clone(),
        joints: vec![joint],
    });
    x.main.get_mut::<GroundTruthCamera>(x.camera).unwrap().flow = Some(Handle::default());
    x.check();
    assert_eq!(
        x.geometry().instances[0].world_from_local,
        Mat4::IDENTITY.to_cols_array_2d()
    );
    x.check();
    let generation = x.geometry().generation;
    x.main
        .entity_mut(joint)
        .insert(GlobalTransform::from_translation(Vec3::new(0.8, 0.2, 0.1)));
    x.check();
    assert_eq!(x.geometry().generation, generation + 1);
    x.main
        .resource_mut::<Assets<SkinnedMeshInverseBindposes>>()
        .insert(poses.id(), vec![Mat4::from_translation(Vec3::X)].into())
        .unwrap();
    x.main
        .get_mut::<GroundTruthCamera>(x.camera)
        .unwrap()
        .frame_id = 7;
    x.check();
    assert_eq!(x.geometry().generation, generation + 2);
    x.main
        .entity_mut(joint)
        .insert(GlobalTransform::from_scale(Vec3::ZERO));
    x.check();
    assert!(x.geometry().failure.as_ref().unwrap().contains("skin"));
    x.main.entity_mut(joint).insert(GlobalTransform::IDENTITY);
    x.check();
    assert!(x.geometry().failure.is_none());
    x.main.entity_mut(entity).remove::<SkinnedMesh>();
    x.check();
    assert_eq!(
        x.geometry().instances[0].world_from_local,
        Mat4::from_scale(Vec3::splat(3.0)).to_cols_array_2d()
    );
}

#[test]
fn ground_truth_camera_teardown_and_empty_scene_match_original_cache_reset() {
    let mut x = Extraction::new();
    x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    x.check();
    x.main.entity_mut(x.camera).despawn();
    x.run_current();
    x.run_original();
    same_geometry(x.current.resource(), x.original.resource());
    assert!(x.geometry().vertices.is_empty());
    assert!(x.current.resource::<GeometryCache>().keys.is_empty());
    let mut images = Assets::default();
    x.camera = x
        .main
        .spawn(GroundTruthCamera::new(&mut images, UVec2::splat(8)))
        .id();
    x.check();
    assert_eq!(x.geometry().vertices.len(), 3);
}

#[test]
#[ignore = "allocation-inclusive counterbalanced ground-truth CPU diagnostic; root runs alone"]
fn ground_truth_extraction_allocation_inclusive_cpu_diagnostic() {
    use crate::scene::procedural_indoor::{
        architecture,
        layout::{IndoorLayout, IndoorManifest},
        objects,
    };
    use std::{hint::black_box, time::Instant};

    fn room(seed: u64) -> Extraction {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
        let mut x = Extraction::new();
        let yaw = Mat4::from_rotation_y(scene.world_yaw);
        let mut assemblies = vec![(architecture::architecture(&scene), yaw)];
        assemblies.extend(scene.objects.iter().map(|object| {
            (
                objects::build_object(object),
                yaw * object.transform().to_matrix(),
            )
        }));
        for (assembly, transform) in assemblies {
            for ((surface, label), geometry) in assembly.parts {
                if geometry.indices.is_empty() {
                    continue;
                }
                let mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, geometry.positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, geometry.normals)
                .with_inserted_indices(Indices::U32(geometry.indices));
                let cull = if matches!(
                    surface,
                    crate::scene::procedural_indoor::materials::Surface::Glass
                        | crate::scene::procedural_indoor::materials::Surface::GlassInterior
                ) {
                    None
                } else {
                    Some(Face::Back)
                };
                let (entity, _, _) = x.object(mesh, cull, RenderLayers::default(), transform);
                x.main
                    .entity_mut(entity)
                    .insert(SemanticLabel::from_label(label.split('#').next().unwrap()).unwrap());
            }
        }
        x.check();
        x
    }

    fn run(x: &mut Extraction, original: bool, mode: &str, repeats: usize) -> f64 {
        let entities = x
            .main
            .query_filtered::<Entity, With<Mesh3d>>()
            .iter(&x.main)
            .collect::<Vec<_>>();
        let start = Instant::now();
        for step in 0..repeats {
            let render = if original {
                &mut x.original
            } else {
                &mut x.current
            };
            if mode == "fresh_rebuild" {
                render.insert_resource(ExtractedGeometry::default());
                render.insert_resource(GeometryCache::default());
            } else if mode == "reused_rebuild" {
                render.resource_mut::<GeometryCache>().keys.clear();
            } else if mode == "changing_transforms" {
                for entity in &entities {
                    x.main
                        .entity_mut(*entity)
                        .insert(GlobalTransform::from_translation(Vec3::new(
                            step as f32 * 0.01,
                            0.2,
                            -0.1,
                        )));
                }
            }
            if original {
                x.run_original();
            } else {
                x.run_current();
            }
            black_box(if original {
                x.original.resource::<ExtractedGeometry>()
            } else {
                x.current.resource::<ExtractedGeometry>()
            });
        }
        start.elapsed().as_secs_f64()
    }

    for seed in [200, 207, 239] {
        let mut x = room(seed);
        let counts = (
            x.geometry().instances.len(),
            x.geometry().vertices.len(),
            x.geometry().indices.len(),
        );
        for (mode, repeats) in [
            ("fresh_rebuild", 12),
            ("reused_rebuild", 12),
            ("unchanged_transforms", 192),
            ("changing_transforms", 96),
        ] {
            let mut original = Vec::new();
            let mut current = Vec::new();
            for pair in 0..8 {
                if pair % 2 == 0 {
                    original.push(run(&mut x, true, mode, repeats));
                    current.push(run(&mut x, false, mode, repeats));
                } else {
                    current.push(run(&mut x, false, mode, repeats));
                    original.push(run(&mut x, true, mode, repeats));
                }
                same_geometry(x.current.resource(), x.original.resource());
            }
            let mut sorted_original = original.clone();
            let mut sorted_current = current.clone();
            sorted_original.sort_by(f64::total_cmp);
            sorted_current.sort_by(f64::total_cmp);
            let median = |values: &[f64]| (values[3] + values[4]) * 0.5;
            println!(
                "{}",
                serde_json::json!({
                    "seed":seed, "mode":mode, "repetitions_per_pair":repeats,
                    "instances":counts.0, "vertices":counts.1, "indices":counts.2,
                    "original_seconds":original, "candidate_seconds":current,
                    "original_median_seconds":median(&sorted_original), "candidate_median_seconds":median(&sorted_current),
                    "speedup":median(&sorted_original) / median(&sorted_current),
                    "scope":"CPU full extraction including batch/index/instance allocation, setup and teardown where fresh; room synthesis and GPU/capture excluded; copied original oracle checked after each counterbalanced pair"
                })
            );
        }
    }
}

#[test]
fn glass_policy_filters_shared_geometry_and_invalidates_cached_topology() {
    use crate::render::glass::AnnotationGlass;
    let mut x = Extraction::new();
    let (glass, _, material) = x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::IDENTITY,
    );
    x.main
        .resource_mut::<Assets<StandardMaterial>>()
        .get_mut(&material)
        .unwrap()
        .specular_transmission = 0.8;
    x.object(
        triangle(None),
        None,
        RenderLayers::default(),
        Mat4::from_translation(Vec3::Z),
    );
    x.main.get_mut::<GroundTruthCamera>(x.camera).unwrap().flow = Some(Handle::default());
    x.main
        .insert_resource(crate::app::BevyZeroverseConfig::default());
    x.run_current();
    assert_eq!(x.geometry().instances.len(), 2);
    let generation = x.geometry().generation;
    x.main
        .resource_mut::<crate::app::BevyZeroverseConfig>()
        .annotation_glass = AnnotationGlass::Through;
    x.run_current();
    assert_eq!(x.geometry().instances.len(), 1);
    assert_eq!(x.geometry().topology.len(), 1);
    assert_ne!(x.geometry().topology[0].entity, glass);
    assert!(x.geometry().generation > generation);
    // Annotation preview disables the live PBR material: classification still uses its saved source.
    x.main
        .entity_mut(glass)
        .remove::<MeshMaterial3d<StandardMaterial>>()
        .insert(DisabledPbrMaterial {
            material: material.clone(),
            ..default()
        });
    x.run_current();
    assert_eq!(x.geometry().instances.len(), 1);
    x.main
        .resource_mut::<crate::app::BevyZeroverseConfig>()
        .annotation_glass = AnnotationGlass::Surface;
    x.run_current();
    assert_eq!(x.geometry().instances.len(), 2);
    assert_eq!(x.geometry().topology.len(), 2);
}
