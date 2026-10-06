//! Frozen pre-optimization extraction, intentionally independent of the span
//! assembly and matrix reuse paths. Keep its original validation/order intact.
use super::*;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn extract_geometry_original(
    cameras: Extract<Query<&GroundTruthCamera>>,
    objects: Extract<
        Query<
            (
                Entity,
                &Mesh3d,
                &GlobalTransform,
                Option<&InheritedVisibility>,
                Option<&SemanticLabel>,
                Option<&RenderLayers>,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Option<&DisabledPbrMaterial>,
                Option<&SkinnedMesh>,
            ),
            Without<crate::render::RenderOnlyOverlay>,
        >,
    >,
    meshes: Extract<Res<Assets<Mesh>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    inverse_bindposes: Extract<Res<Assets<bevy::mesh::skinning::SkinnedMeshInverseBindposes>>>,
    joints: Extract<Query<Ref<GlobalTransform>>>,
    mut mesh_events: Extract<MessageReader<AssetEvent<Mesh>>>,
    mut cache: ResMut<GeometryCache>,
    mut geometry: ResMut<ExtractedGeometry>,
) {
    let events: Vec<_> = mesh_events.read().copied().collect();
    if cameras.is_empty() {
        if !geometry.vertices.is_empty() {
            *geometry = ExtractedGeometry {
                generation: geometry.generation.wrapping_add(1),
                ..default()
            };
        }
        cache.keys.clear();
        return;
    }
    let flow_enabled = cameras.iter().any(|c| c.flow.is_some());
    let retry_failed_geometry = geometry.failure.is_some();
    geometry.failure = None;
    let mut objects: Vec<_> = objects
        .iter()
        .filter(|(_, _, _, visible, ..)| visible.is_none_or(|v| v.get()))
        .collect();
    objects.sort_by_key(|(entity, ..)| entity.to_bits());
    let keys: Vec<_> = objects
        .iter()
        .map(|(entity, mesh, _, _, _, layers, material, disabled, _)| {
            let cull = material
                .and_then(|m| materials.get(&m.0))
                .map(|m| m.cull_mode)
                .or_else(|| disabled.map(|m| m.cull_mode))
                .flatten();
            let cull = match cull {
                None => 0,
                Some(Face::Back) => 1,
                Some(Face::Front) => 2,
            };
            (
                *entity,
                mesh.id(),
                cull,
                layers.cloned().unwrap_or_default(),
            )
        })
        .collect();
    let skin_stamp = cameras.iter().map(|c| c.frame_id).max().unwrap_or(0);
    let skinned_entities: Vec<_> = objects
        .iter()
        .filter(|o| o.8.is_some())
        .map(|o| o.0)
        .collect();
    // A new capture refreshes skin data even when only inverse binds or the
    // SkinnedMesh component changed. Warm-up joint updates remain inexpensive.
    let skin_changed = skinned_entities != cache.skinned_entities
        || (!skinned_entities.is_empty() && skin_stamp != cache.skin_stamp)
        || objects.iter().any(|(_, _, _, _, _, _, _, _, skin)| {
            skin.is_some_and(|s| {
                s.joints
                    .iter()
                    .any(|e| joints.get(*e).is_ok_and(|t| t.is_changed()))
            })
        });
    cache.skin_stamp = skin_stamp;
    cache.skinned_entities = skinned_entities;
    let changed = skin_changed || retry_failed_geometry || keys != cache.keys || flow_enabled != cache.flow_enabled || events.iter().any(|event| matches!(event, AssetEvent::Modified { id } | AssetEvent::Removed { id } if keys.iter().any(|(_, mesh, ..)| mesh == id)));
    if changed {
        geometry.generation = geometry.generation.wrapping_add(1);
        geometry.vertices.clear();
        geometry.indices.clear();
        geometry.batches.clear();
        geometry.topology.clear();
        let mut batch_indices = BTreeMap::<(usize, RenderLayers), Vec<u32>>::new();
        for (instance, ((entity, mesh_handle, _, _, _, _, _, _, skin), (_, _, cull, layers))) in
            objects.iter().zip(&keys).enumerate()
        {
            let Some(mesh) = meshes.get(mesh_handle.id()) else {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} is unavailable in the CPU asset world"
                ));
                break;
            };
            if mesh.morph_targets().is_some() {
                geometry.failure = Some(format!(
                    "ground truth requires baked geometry; morph deformation on {entity:?} is unsupported"
                ));
                break;
            }
            if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} is not a triangle list"
                ));
                break;
            }
            let (
                Some(VertexAttributeValues::Float32x3(positions)),
                Some(VertexAttributeValues::Float32x3(normals)),
            ) = (
                mesh.attribute(Mesh::ATTRIBUTE_POSITION),
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL),
            )
            else {
                geometry.failure = Some(format!(
                    "ground truth mesh {entity:?} lacks Float32 positions/normals"
                ));
                break;
            };
            if positions.len() != normals.len()
                || positions.len() > u32::MAX as usize - geometry.vertices.len()
            {
                geometry.failure = Some(
                    "ground truth mesh has inconsistent or excessive vertex attributes".into(),
                );
                break;
            }
            if positions.iter().any(|p| !Vec3::from_array(*p).is_finite())
                || normals.iter().any(|n| {
                    !Vec3::from_array(*n).is_finite()
                        || Vec3::from_array(*n).length_squared() < 1e-12
                })
                || mesh.indices().is_some_and(|indices| {
                    indices.len() % 3 != 0 || indices.iter().any(|index| index >= positions.len())
                })
                || (mesh.indices().is_none() && positions.len() % 3 != 0)
            {
                geometry.failure =
                    Some(format!("ground truth mesh {entity:?} has invalid geometry: positions={}, normals={}, bad_position={:?}, bad_normal={:?}, indices={:?}",
                        positions.len(), normals.len(),
                        positions.iter().position(|p| !Vec3::from_array(*p).is_finite()),
                        normals.iter().position(|n| !Vec3::from_array(*n).is_finite() || Vec3::from_array(*n).length_squared() < 1e-12),
                        mesh.indices().map(|i| i.len())));
                break;
            }
            let baked;
            let (positions, normals) = if let Some(skin) = skin {
                match skin::bake(mesh, skin, &inverse_bindposes, &joints) {
                    Ok(value) => baked = value,
                    Err(error) => {
                        geometry.failure = Some(format!("ground truth skin {entity:?}: {error}"));
                        break;
                    }
                }
                (&baked.0, &baked.1)
            } else {
                (positions, normals)
            };
            let offset = geometry.vertices.len() as u32;
            if flow_enabled {
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                if let Some(indices) = mesh.indices() {
                    for index in indices.iter() {
                        index.hash(&mut hash);
                    }
                }
                geometry.topology.push(ObjectTopology {
                    entity: *entity,
                    mesh: mesh_handle.id(),
                    vertices: offset as usize..offset as usize + positions.len(),
                    indices_hash: hash.finish(),
                });
            }
            geometry
                .vertices
                .extend(
                    positions
                        .iter()
                        .zip(normals)
                        .map(|(position, normal)| Vertex {
                            position: *position,
                            normal: *normal,
                            instance: instance as u32,
                        }),
                );
            let indices = batch_indices.entry((*cull, layers.clone())).or_default();
            if let Some(source) = mesh.indices() {
                indices.extend(source.iter().map(|i| offset + i as u32));
            } else {
                indices.extend(offset..offset + positions.len() as u32);
            }
        }
        for ((cull, layers), indices) in batch_indices {
            let start = geometry.indices.len() as u32;
            geometry.indices.extend(indices);
            let end = geometry.indices.len() as u32;
            geometry.batches.push(Batch {
                indices: start..end,
                cull,
                layers,
            });
        }
        cache.keys = keys;
        cache.flow_enabled = flow_enabled;
    }
    geometry.instances.clear();
    for (entity, _, transform, _, semantic, _, _, _, skin) in objects {
        // Bevy's skin matrices already transform rest vertices to world space.
        let world_from_local = if skin.is_some() {
            Mat4::IDENTITY
        } else {
            transform.to_matrix()
        };
        let normal_from_local = world_from_local.inverse().transpose();
        if !world_from_local.is_finite() || !normal_from_local.is_finite() {
            geometry.failure = Some(format!(
                "ground truth entity {entity:?} has a singular/nonfinite transform"
            ));
            break;
        }
        geometry.instances.push(Instance {
            world_from_local: world_from_local.to_cols_array_2d(),
            normal_from_local: normal_from_local.to_cols_array_2d(),
            semantic: semantic.map_or(0, semantic_id),
            padding: [0; 3],
        });
    }
    for camera in &cameras {
        *camera.status.failure.lock().unwrap() = geometry.failure.clone();
        if geometry.failure.is_some() {
            camera.status.valid.store(false, Ordering::Release);
        }
    }
}
