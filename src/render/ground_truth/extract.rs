//! Exact CPU geometry extraction. Index spans borrow the live mesh assets only
//! for this extraction; matrix reuse lives in the existing instance buffer.
use super::*;
use bevy::{asset::AssetEvent, mesh::Indices, render::Extract};
use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
};

#[derive(Resource, Default)]
pub(super) struct GeometryCache {
    flow_enabled: bool,
    skin_stamp: u64,
    skinned_entities: Vec<Entity>,
    keys: Vec<(Entity, AssetId<Mesh>, usize, RenderLayers)>,
}

/// One span per live mesh instance, rather than a temporary copy of its indices.
struct IndexSpan<'a> {
    source: Option<&'a Indices>,
    vertices: Range<u32>,
}

impl IndexSpan<'_> {
    fn len(&self) -> usize {
        self.source.map_or(self.vertices.len(), Indices::len)
    }

    fn append_to(self, indices: &mut Vec<u32>) {
        if let Some(source) = self.source {
            indices.extend(
                source
                    .iter()
                    .map(|index| self.vertices.start + index as u32),
            );
        } else {
            indices.extend(self.vertices);
        }
    }
}

/// Match bits, including signed zero, before reusing the inverse-transpose.
fn same_matrix_bits(left: &[[f32; 4]; 4], right: &[[f32; 4]; 4]) -> bool {
    bytemuck::bytes_of(left) == bytemuck::bytes_of(right)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn extract_geometry(
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
        let mut batch_spans = BTreeMap::<(usize, RenderLayers), Vec<IndexSpan>>::new();
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
            batch_spans
                .entry((*cull, layers.clone()))
                .or_default()
                .push(IndexSpan {
                    source: mesh.indices(),
                    vertices: offset..offset + positions.len() as u32,
                });
        }
        // Reserve once, then write each index directly in the existing sorted
        // batch order and entity order within each batch. Spans remain bounded
        // by the visible instances and are dropped before extraction finishes.
        let index_count = batch_spans.values().flatten().map(IndexSpan::len).sum();
        geometry.indices.reserve(index_count);
        for ((cull, layers), spans) in batch_spans {
            let start = geometry.indices.len() as u32;
            for span in spans {
                span.append_to(&mut geometry.indices);
            }
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
    let mut instance_count = 0;
    for (instance, (entity, _, transform, _, semantic, _, _, _, skin)) in
        objects.into_iter().enumerate()
    {
        // Bevy's skin matrices already transform rest vertices to world space.
        let world_from_local = if skin.is_some() {
            Mat4::IDENTITY
        } else {
            transform.to_matrix()
        };
        let world_columns = world_from_local.to_cols_array_2d();
        let normal_from_local = geometry
            .instances
            .get(instance)
            .filter(|previous| same_matrix_bits(&previous.world_from_local, &world_columns))
            .map_or_else(
                || world_from_local.inverse().transpose(),
                |previous| Mat4::from_cols_array_2d(&previous.normal_from_local),
            );
        if !world_from_local.is_finite() || !normal_from_local.is_finite() {
            geometry.failure = Some(format!(
                "ground truth entity {entity:?} has a singular/nonfinite transform"
            ));
            break;
        }
        let value = Instance {
            world_from_local: world_columns,
            normal_from_local: normal_from_local.to_cols_array_2d(),
            semantic: semantic.map_or(0, semantic_id),
            padding: [0; 3],
        };
        if let Some(previous) = geometry.instances.get_mut(instance) {
            *previous = value;
        } else {
            geometry.instances.push(value);
        }
        instance_count += 1;
    }
    geometry.instances.truncate(instance_count);
    for camera in &cameras {
        *camera.status.failure.lock().unwrap() = geometry.failure.clone();
        if geometry.failure.is_some() {
            camera.status.valid.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
#[path = "extract/tests.rs"]
mod tests;
