//! Retain Bevy's CPU cluster buffers when their exact inputs have not changed.
//!
//! Bevy 0.20 otherwise allocates and uploads them every frame. In particular,
//! mapped-at-creation zero fills can stall discrete Vulkan devices. Keep Bevy's
//! assignment, ordering, packing and upload implementation; only suppress a
//! redundant preparation. No cluster resolution or illumination is changed.
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
        renderer::RenderAdapterInfo,
        sync_world::{MainEntity, MainEntityHashMap, RenderEntity},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

#[cfg(test)]
mod tests;

pub(super) fn install(app: &mut App) {
    // Qualify the extra CPU comparison only where mapped allocation stalls were
    // measured. Other adapters (including browsers) retain Bevy's preparation.
    let Some(adapter) = app.world().get_resource::<RenderAdapterInfo>() else {
        return;
    };
    if adapter.vendor != 0x10de || adapter.backend != wgpu::Backend::Vulkan {
        return;
    }
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.add_systems(
        ExtractSchedule,
        extract_key.after(extract_clusters_for_cpu_clustering),
    );
    render.add_systems(
        Render,
        reuse_unchanged
            .in_set(RenderSystems::PrepareResources)
            .before(prepare_clusters_for_cpu_clustering),
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Record {
    Counts([u32; 6]),
    Object(Entity, Option<Entity>, [bool; 6]),
}

#[derive(Component)]
struct InputKey(Vec<Record>);

#[derive(Component, Debug, PartialEq, Eq)]
struct UploadedKey {
    records: Vec<Record>,
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
    for (view, clusters, camera) in &views {
        let ClusterableObjects::Cpu(clusters) = &clusters.clusterable_objects else {
            commands.entity(view).remove::<(InputKey, UploadedKey)>();
            continue;
        };
        if !camera.is_active || settings.gpu_clustering.is_some() {
            commands.entity(view).remove::<(InputKey, UploadedKey)>();
            continue;
        }
        let mut records = Vec::with_capacity(clusters.len() * 2);
        let mut valid = true;
        for cluster in clusters {
            let c = cluster.counts;
            records.push(Record::Counts([
                c.point_lights,
                c.spot_lights,
                c.rect_lights,
                c.reflection_probes,
                c.irradiance_volumes,
                c.decals,
            ]));
            for &entity in cluster.iter() {
                let Ok((render, (a, b, c, d, e, f))) = objects.get(entity) else {
                    valid = false;
                    break;
                };
                // Match every classification read by Bevy's extractor. Unknown
                // objects fall back to Bevy, including its own diagnostics.
                if !(a || b || c || d || e || f) {
                    valid = false;
                    break;
                }
                records.push(Record::Object(
                    entity,
                    render.map(|e| **e),
                    [a, b, c, d, e, f],
                ));
            }
            if !valid {
                break;
            }
        }
        if valid {
            commands.entity(view).insert(InputKey(records));
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

#[allow(clippy::type_complexity)]
fn reuse_unchanged(
    mut commands: Commands,
    settings: Res<GlobalClusterSettings>,
    pipeline: Res<MeshPipeline>,
    objects: Res<GlobalClusterableObjectMeta>,
    views: Query<
        (
            Entity,
            &InputKey,
            Option<&UploadedKey>,
            Option<&ViewClusterBindings>,
            Option<&RenderViewLightProbes<EnvironmentMapLight>>,
            Option<&RenderViewLightProbes<IrradianceVolume>>,
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
    for (entity, input, previous, bindings, reflection, irradiance) in &views {
        let key = UploadedKey {
            records: input.0.clone(),
            indices: indices.clone(),
            reflection: probe_indices(
                reflection.map(|p| &p.main_entity_to_render_light_probe_index),
            ),
            irradiance: probe_indices(
                irradiance.map(|p| &p.main_entity_to_render_light_probe_index),
            ),
            binding: pipeline.clustered_forward_buffer_binding_type,
        };
        if bindings.is_some() && previous == Some(&key) {
            // Bevy extraction reinserts this next frame. All other view/light
            // resources still update, including shadows and probe textures.
            commands
                .entity(entity)
                .remove::<ExtractedClusterableObjects>();
        } else {
            commands.entity(entity).insert(key);
        }
    }
}
