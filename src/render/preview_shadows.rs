//! Batch only compatible, rigid, opaque procedural parts for interactive shadows.
//! Original meshes still render RGB and all annotations. Capture restores the
//! original casters before visibility extraction. Parts keep their exact local
//! vertices and parent transform; no geometry simplification or rebaking occurs.
use crate::scene::procedural_indoor::{motion::AnimatedPart, objects::IndoorSurface};
use bevy::{
    asset::RenderAssetUsages,
    camera::{
        primitives::Aabb,
        visibility::{RenderLayers, VisibilitySystems},
    },
    light::{NotShadowCaster, SimulationLightSystems},
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::{Face, PrimitiveTopology},
};
use std::collections::BTreeMap;

// Reserved only while unused by the application. A conflict disables batching.
const LAYER: usize = 31;
const MAX_VERTICES: usize = 500_000;
const MAX_INDICES: usize = 3_000_000;
#[derive(Component)]
struct ShadowBatch;
#[derive(Component)]
struct ShadowSource;
#[derive(Component)]
struct ShadowLight {
    had_layers: bool,
}
#[derive(Resource, Default)]
struct State {
    inputs: Vec<Input>,
    #[cfg(test)]
    force: bool,
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
    mesh: AssetId<Mesh>,
    key: Key,
}

#[cfg(test)]
pub(super) fn force_for_test(app: &mut App, enabled: bool) {
    if !app.world().contains_resource::<State>() {
        install(app);
    }
    app.world_mut().resource_mut::<State>().force = enabled;
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
    if config.headless
        || !config.editor
        || !config.camera_grid
        || config.room_schematic
        || config.image_copiers
        || world.get_resource::<super::RenderMode>() != Some(&super::RenderMode::Color)
        || world
            .get_resource::<crate::sample::SamplerState>()
            .is_some_and(|s| s.enabled)
        || world
            .query::<&crate::io::image_copy::ImageCopier>()
            .iter(world)
            .next()
            .is_some()
    {
        return false;
    }
    world
        .query_filtered::<&Camera, With<crate::camera::lens::LensCamera>>()
        .iter(world)
        .any(|c| c.is_active)
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
    let lights: Vec<_> = world
        .query::<(Entity, &ShadowLight, Option<&RenderLayers>)>()
        .iter(world)
        .map(|(e, owned, layers)| {
            (
                e,
                owned.had_layers,
                layers.cloned().unwrap_or_default().without(LAYER),
            )
        })
        .collect();
    for (entity, had, layers) in lights {
        let mut entity = world.entity_mut(entity);
        entity.remove::<ShadowLight>();
        if !had && layers == RenderLayers::default() {
            entity.remove::<RenderLayers>();
        } else {
            entity.insert(layers);
        }
    }
    world.resource_mut::<State>().inputs.clear();
}

#[allow(clippy::type_complexity)]
fn update(
    world: &mut World,
    mut events: Local<bevy::ecs::message::MessageCursor<AssetEvent<Mesh>>>,
) {
    #[cfg(test)]
    let force = world.resource::<State>().force;
    #[cfg(not(test))]
    let force = false;
    // Headless generation and non-grid views do not scan geometry or assets.
    if !force && !requested(world) {
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
    let conflict = world
        .query_filtered::<&RenderLayers, (Without<ShadowBatch>, Without<ShadowLight>)>()
        .iter(world)
        .any(|layers| layers.intersects(&RenderLayers::layer(LAYER)));
    if conflict {
        if !world.resource::<State>().inputs.is_empty() {
            clear(world);
        }
        return;
    }
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
    ), (
        With<IndoorSurface>,
        Without<AnimatedPart>,
        Without<bevy::mesh::skinning::SkinnedMesh>,
    )>();
    let materials = world.resource::<Assets<StandardMaterial>>();
    for (entity, mesh, material, parent, local, global, visible, skipped, owned, layers) in
        query.iter(world)
    {
        if (skipped && !owned)
            || !visible.get()
            || *local != Transform::IDENTITY
            || layers.is_some_and(|l| *l != RenderLayers::default())
        {
            continue;
        }
        let Some(material) = materials.get(&material.0) else {
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
            mesh: mesh.id(),
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
            let material = world
                .resource_mut::<Assets<StandardMaterial>>()
                .add(StandardMaterial {
                    cull_mode: match key.cull {
                        1 => Some(Face::Front),
                        2 => Some(Face::Back),
                        _ => None,
                    },
                    double_sided: key.double_sided,
                    depth_bias: f32::from_bits(key.depth_bias),
                    ..default()
                });
            world.spawn((
                ShadowBatch,
                super::RenderOnlyOverlay,
                Mesh3d(mesh),
                MeshMaterial3d(material),
                RenderLayers::layer(LAYER),
                Transform::IDENTITY,
                global,
                Visibility::Inherited,
                InheritedVisibility::VISIBLE,
                ViewVisibility::default(),
                bounds,
                ChildOf(key.parent),
                Name::new("interactive_shadow_batch"),
            ));
            for part in parts {
                world
                    .entity_mut(part.entity)
                    .insert((ShadowSource, NotShadowCaster));
            }
        }
        world.resource_mut::<State>().inputs = inputs;
    }
    if world
        .query_filtered::<Entity, With<ShadowBatch>>()
        .iter(world)
        .next()
        .is_none()
    {
        return;
    }
    // New lights also need the private caster layer, without rebuilding meshes.
    let lights: Vec<_> = world
        .query_filtered::<(Entity, Option<&RenderLayers>, Has<ShadowLight>),
            Or<(With<PointLight>, With<SpotLight>, With<DirectionalLight>)>>()
        .iter(world)
        .map(|(e, l, owned)| (e, l.cloned(), owned))
        .collect();
    for (entity, layers, owned) in lights {
        let mask = layers.clone().unwrap_or_default();
        let receives = mask.intersects(&RenderLayers::default());
        if receives {
            if !owned {
                world.entity_mut(entity).insert(ShadowLight {
                    had_layers: layers.is_some(),
                });
            }
            if !mask.intersects(&RenderLayers::layer(LAYER)) {
                world.entity_mut(entity).insert(mask.with(LAYER));
            }
        } else if owned {
            world
                .entity_mut(entity)
                .remove::<ShadowLight>()
                .insert(mask.without(LAYER));
        }
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
