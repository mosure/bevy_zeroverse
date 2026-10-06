//! Reserve immutable mesh identities independently of material-map completion.
//! These plans never reach ECS/render extraction until all bindings are resolved.
use super::*;

enum Binding {
    Surface(Surface, String),
    Human(usize, humans::HumanSurface),
}

struct PartPlan {
    mesh: Handle<Mesh>,
    binding: Binding,
    label: SemanticLabel,
    surface: Option<Surface>,
    human_surface: Option<humans::HumanSurface>,
    ovoxel_excluded: bool,
}

pub(super) struct GroupPlan {
    name: String,
    transform: Transform,
    bounds: (Vec3, Vec3),
    object: Option<(usize, super::super::layout::ObjectKind)>,
    human: Option<(usize, Vec<Vec3>)>,
    annotation: Option<SemanticLabel>,
    parts: Vec<PartPlan>,
}

impl GroupPlan {
    pub(super) fn bind(
        self,
        manifest: &IndoorManifest,
        material_set: &IndoorMaterials,
        materials: &mut StagedAssets<StandardMaterial>,
    ) -> Group {
        let parts = self
            .parts
            .into_iter()
            .map(|part| {
                let material = match part.binding {
                    Binding::Surface(surface, label) => material_set.for_part(surface, &label),
                    Binding::Human(index, surface) => {
                        let material = humans::person_material(
                            &manifest.humans[index],
                            surface,
                            material_set,
                            materials,
                        );
                        materials.add(material)
                    }
                };
                Part {
                    mesh: part.mesh,
                    material,
                    label: part.label,
                    surface: part.surface,
                    human_surface: part.human_surface,
                    ovoxel_excluded: part.ovoxel_excluded,
                }
            })
            .collect();
        Group {
            name: self.name,
            transform: self.transform,
            bounds: self.bounds,
            object: self.object,
            human: self.human,
            annotation: self.annotation,
            parts,
        }
    }
}

pub(super) async fn plan(
    manifest: &IndoorManifest,
    geometry: SceneGeometry,
    meshes: &StagedAssets<Mesh>,
) -> (
    Vec<GroupPlan>,
    Vec<(Handle<Mesh>, super::super::geometry::Geometry)>,
) {
    let mut groups = Vec::new();
    let mut mesh_jobs = Vec::new();
    let mut add =
        |assembly: objects::Assembly, name: String, transform: Transform, object, annotation| {
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|((surface, label), geometry)| PartPlan {
                    mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                    label: SemanticLabel::from_label(objects::part_label(&label))
                        .expect("indoor semantic vocabulary"),
                    surface: Some(surface),
                    human_surface: None,
                    ovoxel_excluded: label.ends_with("#exterior"),
                    binding: Binding::Surface(surface, label),
                })
                .collect();
            groups.push(GroupPlan {
                name,
                transform,
                bounds,
                object,
                human: None,
                annotation,
                parts,
            });
        };
    let mut shell = objects::Assembly::default();
    let mut fixtures = std::collections::BTreeMap::<String, objects::Assembly>::new();
    for ((surface, label), geometry) in geometry.architecture.parts {
        if label.starts_with("lamp#") {
            fixtures
                .entry(label.clone())
                .or_default()
                .parts
                .insert((surface, label), geometry);
        } else {
            shell.parts.insert((surface, label), geometry);
        }
    }
    for (name, fixture) in fixtures {
        add(
            fixture,
            format!("ceiling_{name}"),
            Transform::IDENTITY,
            None,
            Some(SemanticLabel::Lamp),
        );
    }
    add(
        shell,
        "architecture".into(),
        Transform::IDENTITY,
        None,
        None,
    );
    for (object, assembly) in manifest.objects.iter().zip(geometry.objects) {
        cooperate().await;
        add(
            assembly,
            format!("{:?}_{}", object.kind, object.id),
            object.transform(),
            Some((object.id, object.kind)),
            None,
        );
    }
    for (index, (person, assembly)) in manifest.humans.iter().zip(geometry.humans).enumerate() {
        cooperate().await;
        let bounds = assembly.bounds();
        let parts = assembly
            .parts
            .into_iter()
            .filter(|(_, g)| !g.indices.is_empty())
            .map(|(surface, geometry)| PartPlan {
                mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                binding: Binding::Human(index, surface),
                label: SemanticLabel::Person,
                surface: None,
                human_surface: Some(surface),
                ovoxel_excluded: false,
            })
            .collect();
        groups.push(GroupPlan {
            name: format!("person_{}", person.id),
            transform: person.transform(),
            bounds,
            object: None,
            human: Some((person.id, assembly.local_joints)),
            annotation: None,
            parts,
        });
    }
    (groups, mesh_jobs)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::super::super::geometry::Geometry;
    use super::*;

    // Keep the previous serial construction as an independent ordering/binding
    // oracle. The production planner defers material binding, not geometry math.
    async fn original_groups(
        manifest: &IndoorManifest,
        geometry: SceneGeometry,
        material_set: &IndoorMaterials,
        materials: &mut StagedAssets<StandardMaterial>,
        meshes: &StagedAssets<Mesh>,
    ) -> (Vec<Group>, Vec<(Handle<Mesh>, Geometry)>) {
        let mut groups = Vec::new();
        let mut mesh_jobs = Vec::new();
        let mut add = |assembly: objects::Assembly,
                       name: String,
                       transform: Transform,
                       object,
                       annotation| {
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|((surface, label), geometry)| Part {
                    mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                    material: material_set.for_part(surface, &label),
                    label: SemanticLabel::from_label(objects::part_label(&label))
                        .expect("indoor semantic vocabulary"),
                    surface: Some(surface),
                    human_surface: None,
                    ovoxel_excluded: label.ends_with("#exterior"),
                })
                .collect();
            groups.push(Group {
                name,
                transform,
                bounds,
                object,
                human: None,
                annotation,
                parts,
            });
        };
        let mut shell = objects::Assembly::default();
        let mut fixtures = std::collections::BTreeMap::<String, objects::Assembly>::new();
        for ((surface, label), geometry) in geometry.architecture.parts {
            if label.starts_with("lamp#") {
                fixtures
                    .entry(label.clone())
                    .or_default()
                    .parts
                    .insert((surface, label), geometry);
            } else {
                shell.parts.insert((surface, label), geometry);
            }
        }
        for (name, fixture) in fixtures {
            add(
                fixture,
                format!("ceiling_{name}"),
                Transform::IDENTITY,
                None,
                Some(SemanticLabel::Lamp),
            );
        }
        add(
            shell,
            "architecture".into(),
            Transform::IDENTITY,
            None,
            None,
        );
        for (object, assembly) in manifest.objects.iter().zip(geometry.objects) {
            cooperate().await;
            add(
                assembly,
                format!("{:?}_{}", object.kind, object.id),
                object.transform(),
                Some((object.id, object.kind)),
                None,
            );
        }
        for (person, assembly) in manifest.humans.iter().zip(geometry.humans) {
            cooperate().await;
            let bounds = assembly.bounds();
            let parts = assembly
                .parts
                .into_iter()
                .filter(|(_, g)| !g.indices.is_empty())
                .map(|(surface, geometry)| {
                    let material =
                        humans::person_material(person, surface, material_set, materials);
                    Part {
                        mesh: meshes.defer_geometry(geometry, &mut mesh_jobs),
                        material: materials.add(material),
                        label: SemanticLabel::Person,
                        surface: None,
                        human_surface: Some(surface),
                        ovoxel_excluded: false,
                    }
                })
                .collect();
            groups.push(Group {
                name: format!("person_{}", person.id),
                transform: person.transform(),
                bounds,
                object: None,
                human: Some((person.id, assembly.local_joints)),
                annotation: None,
                parts,
            });
        }
        (groups, mesh_jobs)
    }

    fn clone_geometry(scene: &SceneGeometry) -> SceneGeometry {
        fn assembly(source: &objects::Assembly) -> objects::Assembly {
            objects::Assembly {
                parts: source.parts.clone(),
            }
        }
        SceneGeometry {
            architecture: assembly(&scene.architecture),
            objects: scene.objects.iter().map(assembly).collect(),
            humans: scene.humans.clone(),
        }
    }

    fn fixture(manifest: &IndoorManifest) -> SceneGeometry {
        let shape = |offset: f32| {
            let mut shape = Geometry::default();
            shape.cuboid(
                Vec3::new(0.71, 0.83, 0.39),
                0.034,
                Transform::from_translation(Vec3::new(offset, 0.17, -0.31))
                    .with_rotation(Quat::from_rotation_y(0.37)),
            );
            shape
        };
        let mut architecture = objects::Assembly::default();
        // BTree order, fixture extraction, a finish, exterior exclusion and
        // empty-index filtering all have different metadata/material behavior.
        for (surface, label, offset) in [
            (Surface::Light, "lamp#1", 1.),
            (Surface::Light, "lamp#0", 2.),
            (Surface::Plastic, "wall#finish7", 3.),
            (Surface::Glass, "window#exterior", 4.),
        ] {
            architecture
                .parts
                .insert((surface, label.into()), shape(offset));
        }
        let mut empty = shape(7.);
        empty.indices.clear();
        architecture
            .parts
            .insert((Surface::Paint, "wall".into()), empty);
        let objects = manifest
            .objects
            .iter()
            .enumerate()
            .map(|(index, object)| {
                let mut assembly = objects::Assembly::default();
                assembly.parts.insert(
                    (
                        Surface::Plastic,
                        format!("{}#finish7", object.kind.class_name()),
                    ),
                    shape(index as f32),
                );
                assembly.parts.insert(
                    (Surface::Metal, object.kind.class_name().into()),
                    shape(index as f32 + 0.5),
                );
                assembly
            })
            .collect();
        let humans = manifest
            .humans
            .iter()
            .map(|_| {
                let mut assembly = humans::HumanAssembly::default();
                for (index, surface) in [
                    humans::HumanSurface::Skin,
                    humans::HumanSurface::Top,
                    humans::HumanSurface::Hair,
                    humans::HumanSurface::Shoes,
                ]
                .into_iter()
                .enumerate()
                {
                    assembly.parts.insert(surface, shape(index as f32));
                }
                assembly
                    .parts
                    .insert(humans::HumanSurface::Lip, Geometry::default());
                assembly.local_joints = vec![Vec3::new(0., 0.5, -0.2), Vec3::new(0.3, 0.7, 0.1)];
                assembly
            })
            .collect();
        SceneGeometry {
            architecture,
            objects,
            humans,
        }
    }

    #[test]
    fn deferred_material_binding_preserves_groups_mesh_bytes_and_tangents() {
        let mut manifest = (0..32)
            .find_map(|seed| {
                let scene =
                    IndoorManifest::generate_with_humans(seed, default(), 0.65, 0, 1.).unwrap();
                (!scene.humans.is_empty() && scene.objects.len() >= 2).then_some(scene)
            })
            .expect("fixture needs two objects and one human");
        manifest.objects.truncate(2);
        manifest.humans.truncate(1);
        manifest.program = None;
        let geometry = fixture(&manifest);
        let original_geometry = clone_geometry(&geometry);
        let image_assets = Assets::<Image>::default();
        let material_assets = Assets::<StandardMaterial>::default();
        let mesh_assets = Assets::<Mesh>::default();
        let mut images = StagedAssets::new(&image_assets);
        let mut materials = StagedAssets::new(&material_assets);
        let mut meshes = StagedAssets::new(&mesh_assets);
        // Minimal map selection keeps this a bounded CPU oracle; one finish
        // still exercises the actual material lookup and human template logic.
        let selection = MaterialSelection {
            surfaces: Default::default(),
            finishes: [(Surface::Plastic, 7)].into_iter().collect(),
            direct_surfaces: Default::default(),
        };
        let material_set = bevy::tasks::block_on(IndoorMaterials::build_async_with_selection(
            &manifest,
            IndoorQuality::Auto,
            &mut images,
            &mut materials,
            Some(&selection),
        ));
        let (plans, jobs) = bevy::tasks::block_on(plan(&manifest, geometry, &meshes));
        assert!(
            meshes.entries.is_empty(),
            "reserving plans must not publish meshes"
        );
        let groups: Vec<_> = plans
            .into_iter()
            .map(|plan| plan.bind(&manifest, &material_set, &mut materials))
            .collect();
        let (original, original_jobs) = bevy::tasks::block_on(original_groups(
            &manifest,
            original_geometry,
            &material_set,
            &mut materials,
            &meshes,
        ));
        assert_eq!(groups.len(), 2 + 1 + 2 + 1);
        assert_eq!(groups.len(), original.len());
        assert_eq!(jobs.len(), original_jobs.len());
        // Realization uses the exact old into_mesh() including MikkTSpace and
        // per-vertex normal/tangent hardening; compare packed attribute bytes.
        let expected: Vec<_> = original_jobs
            .into_iter()
            .map(|(_, geometry)| geometry.into_mesh())
            .collect();
        let ordered_handles: Vec<_> = jobs.iter().map(|(handle, _)| handle.clone()).collect();
        bevy::tasks::block_on(meshes.realize_geometry(jobs));
        assert_eq!(meshes.entries.len(), expected.len());
        for (handle, expected) in ordered_handles.iter().zip(&expected) {
            let actual = meshes.get(handle).unwrap();
            assert_eq!(
                actual.create_packed_vertex_buffer_data(),
                expected.create_packed_vertex_buffer_data()
            );
            assert_eq!(
                actual
                    .indices()
                    .map(|indices| indices.iter().collect::<Vec<_>>()),
                expected
                    .indices()
                    .map(|indices| indices.iter().collect::<Vec<_>>())
            );
            assert!(actual.attribute(Mesh::ATTRIBUTE_TANGENT).is_some());
        }
        let mut part_offset = 0;
        for (actual, expected) in groups.iter().zip(&original) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.transform, expected.transform);
            assert_eq!(actual.bounds, expected.bounds);
            assert_eq!(actual.object, expected.object);
            assert_eq!(actual.human, expected.human);
            assert_eq!(actual.annotation, expected.annotation);
            assert_eq!(actual.parts.len(), expected.parts.len());
            for (actual, expected) in actual.parts.iter().zip(&expected.parts) {
                assert_eq!(actual.mesh.id(), ordered_handles[part_offset].id());
                part_offset += 1;
                assert_eq!(actual.label, expected.label);
                assert_eq!(actual.surface, expected.surface);
                assert_eq!(actual.human_surface, expected.human_surface);
                assert_eq!(actual.ovoxel_excluded, expected.ovoxel_excluded);
                if actual.human_surface.is_none() {
                    assert_eq!(actual.material.id(), expected.material.id());
                }
                assert_eq!(
                    format!("{:?}", materials.get(&actual.material).unwrap()),
                    format!("{:?}", materials.get(&expected.material).unwrap())
                );
            }
        }
        assert_eq!(part_offset, ordered_handles.len());
    }
}
