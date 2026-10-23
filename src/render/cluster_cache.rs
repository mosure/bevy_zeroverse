//! Retain and update Bevy's CPU cluster buffers without repeated mapped uploads.
//!
//! Bevy 0.20 otherwise allocates and uploads them every frame. In particular,
//! mapped-at-creation zero fills can stall discrete Vulkan devices. Keep Bevy's
//! assignment, ordering and packing. Changed storage data uses bounded queue
//! copies, and a bounded pool retains allocations across room regeneration.
//! No cluster resolution or illumination is changed.
use bevy::{
    light::{
        cluster::{ClusterableObjects, Clusters, GlobalClusterSettings},
        ClusteredDecal, IrradianceVolume, RectLight,
    },
    pbr::{
        extract_clusters_for_cpu_clustering, prepare_clusters_for_cpu_clustering,
        ExtractedClusterableObjects, GlobalClusterableObjectMeta, MeshPipeline,
        RenderViewLightProbes, ViewClusterBindings,
    },
    prelude::*,
    render::{
        render_resource::BufferBindingType,
        renderer::{RenderAdapterInfo, RenderDevice, RenderQueue},
        sync_world::{MainEntity, MainEntityHashMap, RenderEntity},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

mod storage;
#[cfg(test)]
mod tests;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct ClusterPreparation;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct ClusterBufferUpload;

pub(super) fn install(app: &mut App) {
    // WebGPU also pays for fresh allocations and mapped uploads per view per
    // frame. Reuse the same bounded, order-preserving storage there. Native
    // adapters retain the existing NVIDIA/Vulkan qualification boundary.
    let Some(adapter) = app.world().get_resource::<RenderAdapterInfo>() else {
        return;
    };
    if !cfg!(target_arch = "wasm32")
        && (adapter.vendor != 0x10de || adapter.backend != wgpu::Backend::Vulkan)
    {
        return;
    }
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.init_resource::<storage::Pool>();
    render.init_resource::<storage::Uploads>();
    // A component hook does not allocate an observer entity. Adding a render
    // entity here can change equal-depth draw/light ordering after regeneration.
    render
        .world_mut()
        .register_component_hooks::<ViewClusterBindings>()
        .on_discard(storage::recycle);
    render.add_systems(
        ExtractSchedule,
        extract_key.after(extract_clusters_for_cpu_clustering),
    );
    render.add_systems(
        Render,
        reuse_unchanged
            .in_set(RenderSystems::PrepareResources)
            .in_set(ClusterPreparation)
            .before(prepare_clusters_for_cpu_clustering),
    );
    render.add_systems(
        Render,
        storage::flush
            .in_set(RenderSystems::PrepareResources)
            .in_set(ClusterBufferUpload)
            .after(ClusterPreparation),
    );
}

/// One palette entry per distinct object, rather than repeating ECS lookups and
/// resolving the same GPU index in every cluster in every view.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Object {
    main: Entity,
    render: Option<Entity>,
    flags: [bool; 6],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Cluster {
    counts: [u32; 6],
    end: usize,
}

#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
struct InputKey {
    clusters: Vec<Cluster>,
    members: Vec<u32>,
    objects: Vec<Object>,
}

#[derive(Component, Debug, PartialEq, Eq)]
struct UploadedKey {
    input: InputKey,
    indices: Vec<(Entity, usize)>,
    reflection: Vec<(MainEntity, u32)>,
    irradiance: Vec<(MainEntity, u32)>,
    binding: BufferBindingType,
}

type ObjectFlags = (
    Has<PointLight>,
    Has<SpotLight>,
    Has<RectLight>,
    Has<EnvironmentMapLight>,
    Has<IrradianceVolume>,
    Has<ClusteredDecal>,
);

fn extract_key(
    mut commands: Commands,
    settings: Extract<Res<GlobalClusterSettings>>,
    views: Extract<Query<(RenderEntity, &Clusters, &Camera)>>,
    objects: Extract<Query<(Option<&RenderEntity>, ObjectFlags)>>,
) {
    let mut extracted = bevy::ecs::entity::EntityHashMap::default();
    for (view, clusters, camera) in &views {
        let ClusterableObjects::Cpu(clusters) = &clusters.clusterable_objects else {
            commands.entity(view).remove::<(InputKey, UploadedKey)>();
            continue;
        };
        if !camera.is_active || settings.gpu_clustering.is_some() {
            commands.entity(view).remove::<(InputKey, UploadedKey)>();
            continue;
        }
        let mut input = InputKey {
            clusters: Vec::with_capacity(clusters.len()),
            ..default()
        };
        let mut slots = bevy::ecs::entity::EntityHashMap::default();
        let mut valid = true;
        for cluster in clusters {
            for &entity in cluster.iter() {
                let slot = if let Some(&slot) = slots.get(&entity) {
                    slot
                } else {
                    let object = extracted.entry(entity).or_insert_with(|| {
                        let (render, (a, b, c, d, e, f)) = objects.get(entity).ok()?;
                        (a || b || c || d || e || f).then_some(Object {
                            main: entity,
                            render: render.map(|e| **e),
                            flags: [a, b, c, d, e, f],
                        })
                    });
                    let Some(object) = object else {
                        valid = false;
                        break;
                    };
                    let Ok(slot) = u32::try_from(input.objects.len()) else {
                        valid = false;
                        break;
                    };
                    input.objects.push(object.clone());
                    slots.insert(entity, slot);
                    slot
                };
                input.members.push(slot);
            }
            if !valid {
                break;
            }
            let c = cluster.counts;
            input.clusters.push(Cluster {
                counts: [
                    c.point_lights,
                    c.spot_lights,
                    c.rect_lights,
                    c.reflection_probes,
                    c.irradiance_volumes,
                    c.decals,
                ],
                end: input.members.len(),
            });
        }
        if valid {
            commands.entity(view).insert(input);
        } else {
            commands.entity(view).remove::<(InputKey, UploadedKey)>();
        }
    }
}

fn probe_indices(probes: Option<&MainEntityHashMap<u32>>) -> Vec<(MainEntity, u32)> {
    let mut indices = probes
        .map(|p| p.iter().map(|(&e, &i)| (e, i)).collect::<Vec<_>>())
        .unwrap_or_default();
    indices.sort_unstable_by_key(|(e, _)| e.to_bits());
    indices
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn reuse_unchanged(
    mut commands: Commands,
    settings: Res<GlobalClusterSettings>,
    pipeline: Res<MeshPipeline>,
    objects: Res<GlobalClusterableObjectMeta>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut pool: ResMut<storage::Pool>,
    mut uploads: ResMut<storage::Uploads>,
    views: Query<
        (
            Entity,
            &InputKey,
            Option<&UploadedKey>,
            Option<&ViewClusterBindings>,
            Option<&RenderViewLightProbes<EnvironmentMapLight>>,
            Option<&RenderViewLightProbes<IrradianceVolume>>,
            Has<storage::Managed>,
        ),
        With<ExtractedClusterableObjects>,
    >,
) {
    if settings.gpu_clustering.is_some() {
        return;
    }
    let mut indices: Vec<_> = objects
        .entity_to_index
        .iter()
        .map(|(&e, &i)| (e, i))
        .collect();
    indices.sort_unstable_by_key(|(e, _)| e.to_bits());
    for (entity, input, previous, bindings, reflection, irradiance, managed) in &views {
        let key = UploadedKey {
            input: input.clone(),
            indices: indices.clone(),
            reflection: probe_indices(
                reflection.map(|p| &p.main_entity_to_render_light_probe_index),
            ),
            irradiance: probe_indices(
                irradiance.map(|p| &p.main_entity_to_render_light_probe_index),
            ),
            binding: pipeline.clustered_forward_buffer_binding_type,
        };
        let unchanged = bindings.is_some() && previous == Some(&key);
        let updated = !unchanged
            && matches!(key.binding, BufferBindingType::Storage { .. })
            && storage::update(
                entity,
                input,
                &objects,
                reflection,
                irradiance,
                bindings.filter(|_| managed),
                &device,
                &queue,
                &mut pool,
                &mut uploads,
                &mut commands,
            );
        if unchanged || updated {
            // Bevy extraction reinserts this next frame. All other view/light
            // resources still update, including shadows and probe textures.
            commands
                .entity(entity)
                .remove::<ExtractedClusterableObjects>()
                .insert(key);
        } else {
            // A fallback is not a successfully uploaded key. In particular, a
            // missing render index must be retried when Bevy extracts it later.
            commands.entity(entity).remove::<UploadedKey>();
        }
    }
}
