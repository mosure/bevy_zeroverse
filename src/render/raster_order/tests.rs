use super::*;
use bevy::{
    core_pipeline::prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey},
    render::{
        batching::gpu_preprocessing::GpuPreprocessingMode,
        render_phase::{BinnedRenderPhaseType, DrawFunctionId},
        render_resource::CachedRenderPipelineId,
        view::RetainedViewEntity,
    },
};

#[test]
fn direct_bins_follow_scene_parts_after_mesh_and_material_allocation_changes() {
    fn replay(reverse: bool) -> (Vec<u64>, Vec<u64>) {
        let mut assets = Assets::<Mesh>::default();
        let mut handles = Vec::new();
        let mut world = World::new();
        let mut order = MeshOrder::default();
        let mut phases = ViewBinnedRenderPhases::<Opaque3dPrepass>::default();
        let view = RetainedViewEntity::new(Entity::PLACEHOLDER.into(), None, 0);
        phases.prepare_for_new_frame(view, GpuPreprocessingMode::PreprocessingOnly);
        for part in if reverse { [3, 2, 1, 0] } else { [0, 1, 2, 3] } {
            let handle = assets.add(Cuboid::default());
            let id = handle.id().untyped();
            handles.push(handle);
            order.insert(id, part);
            let key = (
                OpaqueNoLightmap3dBatchSetKey {
                    pipeline: CachedRenderPipelineId::INVALID,
                    draw_function: DrawFunctionId(0),
                    material_bind_group_index: Some(if reverse { 3 - part } else { part } as u32),
                    slabs: default(),
                },
                OpaqueNoLightmap3dBinKey { asset_id: id },
            );
            let phase = phases.get_mut(&view).unwrap();
            phase.batchable_meshes.insert(key.clone(), default());
            let entity = world.spawn_empty().id();
            phase.add(
                key.0,
                key.1,
                (entity, entity.into()),
                default(),
                BinnedRenderPhaseType::UnbatchableMesh,
            );
        }
        let phase = phases.get_mut(&view).unwrap();
        phase.batchable_meshes.sort_unstable_keys();
        phase.unbatchable_meshes.sort_unstable_keys();
        order_phase(&mut phases, &order, |bin| bin.asset_id);
        let phase = &phases[&view];
        assert_eq!(phase.batchable_meshes.len(), 4);
        assert_eq!(phase.unbatchable_meshes.len(), 4);
        (
            phase
                .batchable_meshes
                .keys()
                .map(|(_, b)| order.0[&b.asset_id])
                .collect(),
            phase
                .unbatchable_meshes
                .keys()
                .map(|(_, b)| order.0[&b.asset_id])
                .collect(),
        )
    }
    assert_eq!(replay(false), (vec![0, 1, 2, 3], vec![0, 1, 2, 3]));
    assert_eq!(replay(false), replay(true));
}

#[test]
fn shared_mesh_order_does_not_depend_on_query_order_or_retain_old_rooms() {
    let mut assets = Assets::<Mesh>::default();
    let shared = assets.add(Cuboid::default());
    let other = assets.add(Cuboid::default());
    let mut order = MeshOrder::default();
    for indexes in [[9, 3, 5], [5, 3, 9]] {
        order.0.clear();
        for index in indexes {
            order.insert(shared.id().untyped(), index);
        }
        assert_eq!(order.key(shared.id().untyped()).1, 3);
        assert!(order.key(shared.id().untyped()) < order.key(other.id().untyped()));
    }
    order.0.clear();
    order.insert(other.id().untyped(), 0);
    assert!(
        order.key(shared.id().untyped()).0,
        "retired meshes must lose their scene order"
    );
}
