use super::SolariPreviewStatus;
use crate::render::semantic::SemanticLabel;
use bevy::{
    material::OpaqueRendererMethod,
    mesh::{
        morph::{MeshMorphWeights, MorphWeights},
        skinning::SkinnedMesh,
        Indices, PrimitiveTopology,
    },
    prelude::*,
};
use bevy_solari::scene::RaytracingMesh3d;
use std::collections::{HashMap, HashSet};

#[derive(Resource, Default)]
pub(super) struct TracedAssets {
    active: bool,
    meshes: HashSet<AssetId<Mesh>>,
    materials: HashMap<AssetId<StandardMaterial>, OpaqueRendererMethod>,
}

fn usable_mesh(mesh: &Mesh) -> bool {
    mesh.primitive_topology() == PrimitiveTopology::TriangleList
        && matches!(mesh.indices(), Some(Indices::U32(_)))
        && mesh.attributes().map(|(a, _)| (a.id, a.format)).eq([
            (Mesh::ATTRIBUTE_POSITION.id, Mesh::ATTRIBUTE_POSITION.format),
            (Mesh::ATTRIBUTE_NORMAL.id, Mesh::ATTRIBUTE_NORMAL.format),
            (Mesh::ATTRIBUTE_UV_0.id, Mesh::ATTRIBUTE_UV_0.format),
            (Mesh::ATTRIBUTE_TANGENT.id, Mesh::ATTRIBUTE_TANGENT.format),
        ])
}

fn usable_material(m: &StandardMaterial) -> bool {
    m.alpha_mode == AlphaMode::Opaque
        && m.specular_transmission == 0.
        && m.diffuse_transmission == 0.
        && (!m.unlit || m.emissive.red.max(m.emissive.green).max(m.emissive.blue) > 0.)
}

#[derive(Component)]
pub(super) struct ManagedInstance;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn synchronize(
    mut commands: Commands,
    mut status: ResMut<SolariPreviewStatus>,
    mut state: ResMut<TracedAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut material_events: MessageReader<AssetEvent<StandardMaterial>>,
    mut mesh_events: MessageReader<AssetEvent<Mesh>>,
    changes: Query<
        (),
        (
            With<SemanticLabel>,
            Or<(
                Changed<Mesh3d>,
                Changed<MeshMaterial3d<StandardMaterial>>,
                Changed<InheritedVisibility>,
                Changed<SemanticLabel>,
                Added<SkinnedMesh>,
                Added<MeshMorphWeights>,
                Added<MorphWeights>,
            )>,
        ),
    >,
    mut removed: RemovedComponents<ManagedInstance>,
    mut removed_labels: RemovedComponents<SemanticLabel>,
    mut removed_materials: RemovedComponents<MeshMaterial3d<StandardMaterial>>,
    mut removed_meshes: RemovedComponents<Mesh3d>,
    instances: Query<
        (
            Entity,
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &InheritedVisibility,
            &SemanticLabel,
            Has<SkinnedMesh>,
            Has<MeshMorphWeights>,
            Has<MorphWeights>,
            Option<&RaytracingMesh3d>,
        ),
        With<SemanticLabel>,
    >,
    owned: Query<Entity, With<ManagedInstance>>,
    #[cfg(feature = "human_motion")] moving_people: Query<
        (),
        With<crate::human_motion::MotionPart>,
    >,
) {
    let asset_change = material_events.read().count() + mesh_events.read().count() != 0;
    let removed = removed.read().count()
        + removed_labels.read().count()
        + removed_materials.read().count()
        + removed_meshes.read().count()
        != 0;
    if status.active == state.active
        && (!status.active || (!asset_change && !removed && changes.is_empty()))
    {
        return;
    }
    state.active = status.active;
    let mut live_meshes = HashSet::new();
    let mut supported = HashSet::new();
    let mut forward = HashSet::new();
    let mut count = 0;
    if status.active {
        if removed {
            for entity in &owned {
                if !instances.contains(entity) {
                    commands
                        .entity(entity)
                        .remove::<(RaytracingMesh3d, ManagedInstance)>();
                }
            }
        }
        for (entity, mesh, material, visibility, label, skinned, morph, mesh_morph, existing) in
            &instances
        {
            #[cfg(feature = "human_motion")]
            let skinned = skinned || moving_people.contains(entity);
            let visible = visibility.get();
            let trace = visible
                && !skinned
                && !morph
                && !mesh_morph
                && *label != SemanticLabel::Sky
                && materials.get(material.id()).is_some_and(usable_material)
                && meshes.get(mesh.id()).is_some_and(usable_mesh);
            if trace {
                count += 1;
                live_meshes.insert(mesh.id());
                if materials.get(material.id()).is_some_and(|m| m.unlit) {
                    forward.insert(material.id());
                } else {
                    supported.insert(material.id());
                }
                if existing.is_none_or(|r| r.id() != mesh.id()) {
                    commands
                        .entity(entity)
                        .insert((RaytracingMesh3d(mesh.0.clone()), ManagedInstance));
                }
            } else {
                if visible {
                    forward.insert(material.id());
                }
                if existing.is_some() && owned.contains(entity) {
                    commands
                        .entity(entity)
                        .remove::<(RaytracingMesh3d, ManagedInstance)>();
                }
            }
        }
    } else {
        for entity in &owned {
            commands
                .entity(entity)
                .remove::<(RaytracingMesh3d, ManagedInstance)>();
        }
    }
    // A material shared with skinned/forward-only geometry must remain forward
    // for both raster views. Its rigid instances still participate in traced rays.
    supported.retain(|id| !forward.contains(id));
    state.materials.retain(|&id, original| {
        if supported.contains(&id) {
            return true;
        }
        if let Some(mut m) = materials.get_mut(id) {
            if m.opaque_render_method == OpaqueRendererMethod::Deferred {
                m.opaque_render_method = *original;
            }
        }
        false
    });
    for id in supported {
        if materials
            .get(id)
            .is_some_and(|m| m.opaque_render_method != OpaqueRendererMethod::Deferred)
        {
            if let Some(mut m) = materials.get_mut(id) {
                state.materials.insert(id, m.opaque_render_method);
                m.opaque_render_method = OpaqueRendererMethod::Deferred;
            }
        }
    }
    for id in state.meshes.symmetric_difference(&live_meshes) {
        let trace = live_meshes.contains(id);
        if let Some(mut mesh) = meshes.get_mut(*id) {
            if mesh.enable_raytracing != trace {
                mesh.enable_raytracing = trace;
            }
        }
    }
    status.traced_instances = count;
    status.traced_meshes = live_meshes.len();
    state.meshes = live_meshes;
}

/// Mesh defaults to ray-traceable. Amend this frame's asset event before extraction
/// without advertising another mesh mutation or re-uploading it on the next frame.
pub(super) fn filter_new_blas(
    state: Res<TracedAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut events: MessageReader<AssetEvent<Mesh>>,
) {
    for event in events.read() {
        if let AssetEvent::Added { id } | AssetEvent::Modified { id } = event {
            if let Some(mesh) = meshes.get_mut_untracked(*id) {
                mesh.enable_raytracing = state.meshes.contains(id);
            }
        }
    }
}
