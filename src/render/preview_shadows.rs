//! Batch compatible, rigid, opaque procedural parts for multi-view shadows.
//! Original meshes still render RGB and all annotations. Shadow-only batches
//! are excluded from ground truth and ovoxels. Parts keep their exact local
//! vertices and parent transform; no geometry simplification or rebaking occurs.
//! Direct capture uses stable color/prepass ordering to resolve depth ties at
//! contact edges independently of proxy allocation. Indirect capture retains
//! original casters until its separate multidraw ordering is qualified.
use crate::scene::procedural_indoor::{motion::AnimatedPart, objects::IndoorSurface};
use bevy::{
    asset::RenderAssetUsages,
    camera::{
        primitives::Aabb,
        visibility::{RenderLayers, VisibilityClass, VisibilitySystems},
    },
    light::{NotShadowCaster, SimulationLightSystems},
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::{Face, PrimitiveTopology},
};
use std::collections::BTreeMap;

const MAX_VERTICES: usize = 500_000;
const MAX_INDICES: usize = 3_000_000;
#[derive(Component)]
struct ShadowBatch;
#[derive(Component)]
struct ShadowSource;
#[derive(Resource, Default)]
struct State {
    inputs: Vec<Input>,
    #[cfg(test)]
    force: Option<bool>,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    parent: Entity,
    cull: u8,
    double_sided: bool,
    depth_bias: u32,
}
#[derive(Clone, PartialEq, Eq)]
struct Input {
    entity: Entity,
    order: Option<u64>,
    mesh: AssetId<Mesh>,
    material: Handle<StandardMaterial>,
    key: Key,
}

#[cfg(test)]
pub(super) fn force_for_test(app: &mut App, enabled: bool) {
    if !app.world().contains_resource::<State>() {
        install(app);
    }
    app.world_mut().resource_mut::<State>().force = Some(enabled);
}
#[cfg(test)]
pub(super) fn active_batches(world: &mut World) -> usize {
    world
        .query_filtered::<Entity, With<ShadowBatch>>()
        .iter(world)
        .count()
}
pub(super) fn install(app: &mut App) {
    app.init_resource::<State>();
    app.add_systems(
        PostUpdate,
        update
            .after(crate::sample::CaptureReadinessUpdate)
            .after(crate::sample::camera_activity::CaptureCameraGate)
            .after(bevy::transform::TransformSystems::Propagate)
            .after(VisibilitySystems::VisibilityPropagate)
            .before(VisibilitySystems::CheckVisibility)
            .before(SimulationLightSystems::CheckLightVisibility),
    );
}

fn requested(world: &mut World) -> bool {
    let Some(config) = world.get_resource::<crate::app::BevyZeroverseConfig>() else {
        return false;
    };
    if config.room_schematic
        || world.get_resource::<super::RenderMode>() != Some(&super::RenderMode::Color)
        // Source geometry, material baking and lighting publication must finish
        // before introducing additional render meshes that reference them.
        || world
            .get_resource::<crate::sample::CaptureReadiness>()
            .is_some_and(|ready| !ready.scene_ready())
    {
        return false;
    }
    let configured_capture = config.headless || config.image_copiers;
    // Bevy's indirect bins have a separate, allocator-dependent ordering.
    // Only direct camera bins currently have our stable scene-part ordering.
    // Include copiers attached through the library API, not only CLI settings.
    let mut copiers = world.query_filtered::<
        (Has<Camera>, Has<bevy::render::view::NoIndirectDrawing>),
        With<crate::io::image_copy::ImageCopier>,
    >();
    let mut has_copier = false;
    let mut direct = true;
    for (camera, direct_draw) in copiers.iter(world) {
        has_copier = true;
        direct &= camera && direct_draw;
    }
    if (configured_capture || has_copier)
        && (!has_copier || !direct || cfg!(target_arch = "wasm32"))
    {
        return false;
    }
    let mut cameras = world.query_filtered::<&Camera, With<crate::camera::lens::LensCamera>>();
    let mut has_rig = false;
    let mut active = false;
    for camera in cameras.iter(world) {
        has_rig = true;
        active |= camera.is_active;
    }
    // Inactivity is not a geometry change. Keep an already built rig's batches
    // while its cameras are hidden/idle, instead of reallocating meshes and
    // moving every original caster between ECS archetypes on reactivation.
    active
        || (has_rig
            && world
                .get_resource::<State>()
                .is_some_and(|state| !state.inputs.is_empty()))
}

fn clear(world: &mut World) {
    let entities: Vec<_> = world
        .query_filtered::<Entity, With<ShadowSource>>()
        .iter(world)
        .collect();
    for entity in entities {
        world
            .entity_mut(entity)
            .remove::<(ShadowSource, NotShadowCaster)>();
    }
    let batches: Vec<_> = world
        .query_filtered::<Entity, With<ShadowBatch>>()
        .iter(world)
        .collect();
    for entity in batches {
        world.despawn(entity);
    }
    world.resource_mut::<State>().inputs.clear();
}

#[allow(clippy::type_complexity)]
fn update(
    world: &mut World,
    mut events: Local<bevy::ecs::message::MessageCursor<AssetEvent<Mesh>>>,
) {
    let enabled = requested(world);
    #[cfg(test)]
    let enabled = world.resource::<State>().force.unwrap_or(enabled);
    // Non-angular and annotation-only views do not scan geometry or assets.
    if !enabled {
        if !world.resource::<State>().inputs.is_empty() {
            clear(world);
        }
        return;
    }
    let changed: Vec<_> = events
        .read(world.resource::<Messages<AssetEvent<Mesh>>>())
        .filter_map(|e| {
            if let AssetEvent::Modified { id } = e {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    let mut inputs = Vec::new();
    let mut groups = BTreeMap::<Key, (GlobalTransform, Vec<Input>)>::new();
    let mut query = world.query_filtered::<(
        Entity,
        &Mesh3d,
        &MeshMaterial3d<StandardMaterial>,
        &ChildOf,
        &Transform,
        &GlobalTransform,
        &InheritedVisibility,
        Has<NotShadowCaster>,
        Has<ShadowSource>,
        Option<&RenderLayers>,
        Option<&crate::render::ground_truth::AnnotationOrder>,
    ), (
        With<IndoorSurface>,
        Without<AnimatedPart>,
        Without<bevy::mesh::skinning::SkinnedMesh>,
    )>();
    let materials = world.resource::<Assets<StandardMaterial>>();
    for (entity, mesh, material, parent, local, global, visible, skipped, owned, layers, order) in
        query.iter(world)
    {
        if (skipped && !owned)
            || !visible.get()
            || *local != Transform::IDENTITY
            || layers.is_some_and(|l| *l != RenderLayers::default())
        {
            continue;
        }
        let material_handle = material.0.clone();
        let Some(material) = materials.get(&material_handle) else {
            continue;
        };
        if material.alpha_mode != AlphaMode::Opaque {
            continue;
        }
        let key = Key {
            parent: parent.parent(),
            cull: match material.cull_mode {
                None => 0,
                Some(Face::Front) => 1,
                Some(Face::Back) => 2,
            },
            double_sided: material.double_sided,
            depth_bias: material.depth_bias.to_bits(),
        };
        let input = Input {
            entity,
            order: order.map(|order| order.0),
            mesh: mesh.id(),
            material: material_handle,
            key: key.clone(),
        };
        inputs.push(input.clone());
        groups
            .entry(key)
            .or_insert_with(|| (*global, Vec::new()))
            .1
            .push(input);
    }
    inputs.sort_by_key(|input| input.entity);
    let old = &world.resource::<State>().inputs;
    let dirty = *old != inputs
        || changed
            .iter()
            .any(|id| inputs.iter().any(|input| input.mesh == *id));
    if dirty {
        clear(world);
        let mut vertices = 0;
        let mut index_count = 0;
        // Entity IDs and query/archetype order change when a room is replayed.
        // Stable scene-part order fixes both triangle/material selection within
        // a batch and admission under the total mesh budget across batches.
        let mut groups: Vec<_> = groups.into_iter().collect();
        for (_, (_, parts)) in &mut groups {
            parts.sort_by_key(Input::sort_key);
        }
        groups.sort_by_key(|(_, (_, parts))| parts[0].sort_key());
        for (key, (global, parts)) in groups {
            if parts.len() < 2 {
                continue;
            }
            let Some((mesh, bounds, count)) = combine(
                world.resource::<Assets<Mesh>>(),
                &parts,
                MAX_VERTICES - vertices,
                MAX_INDICES - index_count,
            ) else {
                continue;
            };
            if vertices + count > MAX_VERTICES {
                continue;
            }
            vertices += count;
            index_count += mesh.indices().unwrap().len();
            let mesh = world.resource_mut::<Assets<Mesh>>().add(mesh);
            // Reuse a compatible source material. Opaque shadow passes depend
            // on the cull/bias key, not albedo, and need no duplicate bindless
            // material allocations merely to combine vertices.
            let material = parts[0].material.clone();
            let batch = world
                .spawn((
                    ShadowBatch,
                    super::RenderOnlyOverlay,
                    Mesh3d(mesh),
                    MeshMaterial3d(material),
                    Transform::IDENTITY,
                    global,
                    Visibility::Inherited,
                    InheritedVisibility::VISIBLE,
                    ViewVisibility::default(),
                    bounds,
                    ChildOf(key.parent),
                    Name::new("procedural_shadow_batch"),
                ))
                .id();
            // Camera draw queues use visibility classes; Bevy's shadow queues
            // collect Mesh3d casters independently. Clear this AFTER Mesh3d's
            // add hook, so proxies never enter camera RGB/prepass queues. Keep
            // original light layers and source material bindings untouched.
            world.entity_mut(batch).insert(VisibilityClass::default());
            for part in parts {
                world
                    .entity_mut(part.entity)
                    .insert((ShadowSource, NotShadowCaster));
            }
        }
        world.resource_mut::<State>().inputs = inputs;
    }
}

impl Input {
    fn sort_key(&self) -> (bool, u64, Entity) {
        // Legacy/custom parts without an annotation order retain an entity
        // fallback. Generated scene parts always carry a unique stable order.
        (self.order.is_none(), self.order.unwrap_or(0), self.entity)
    }
}

fn combine(
    meshes: &Assets<Mesh>,
    parts: &[Input],
    vertex_budget: usize,
    index_budget: usize,
) -> Option<(Mesh, Aabb, usize)> {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for part in parts {
        let mesh = meshes.get(part.mesh)?;
        if mesh.primitive_topology() != PrimitiveTopology::TriangleList || mesh.has_morph_targets()
        {
            return None;
        }
        let VertexAttributeValues::Float32x3(p) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)? else {
            return None;
        };
        let VertexAttributeValues::Float32x3(n) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)? else {
            return None;
        };
        if p.len() != n.len() || positions.len() + p.len() > vertex_budget {
            return None;
        }
        let offset = positions.len() as u32;
        let count = mesh.indices().map_or(p.len(), |i| i.len());
        if count > index_budget.saturating_sub(indices.len()) {
            return None;
        }
        let local: Vec<_> = mesh
            .indices()
            .map(|i| i.iter().collect())
            .unwrap_or_else(|| (0..p.len()).collect());
        if local.len() % 3 != 0 || local.iter().any(|&i| i >= p.len()) {
            return None;
        }
        indices.extend(local.into_iter().map(|i| offset + i as u32));
        for p in p {
            let v = Vec3::from_array(*p);
            if !v.is_finite() {
                return None;
            }
            min = min.min(v);
            max = max.max(v);
        }
        positions.extend_from_slice(p);
        normals.extend_from_slice(n);
    }
    if positions.is_empty() {
        return None;
    }
    let count = positions.len();
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.; 2]; count])
    .with_inserted_indices(Indices::U32(indices));
    Some((mesh, Aabb::from_min_max(min, max), count))
}

#[cfg(test)]
mod tests;
