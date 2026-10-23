//! Stable ordering for generated meshes in Bevy's direct-draw color/prepass bins.
//!
//! At a contact edge, rasterization can quantize two different surface depths
//! to the same value. Bevy breaks that tie by draw order, which normally depends
//! on recycled mesh/slab/pipeline IDs. Shadow proxy allocation exposed this at
//! a window gasket: identical depth but different normals then affected SSAO.
//! Use the scene's stable part order without changing depth, geometry, materials,
//! draw counts or batching. Indirect multidraw bins retain Bevy's ordering;
//! capture shadow batching is restricted to the qualified direct path.
use bevy::{
    asset::UntypedAssetId,
    core_pipeline::{
        core_3d::{AlphaMask3d, Opaque3d},
        prepass::{AlphaMask3dPrepass, Opaque3dPrepass},
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        batching::sort_binned_render_phase,
        render_phase::{BinnedPhaseItem, ViewBinnedRenderPhases},
        Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
    },
};

use super::ground_truth::AnnotationOrder;

#[derive(Resource, Default)]
struct MeshOrder(HashMap<UntypedAssetId, u64>);

impl MeshOrder {
    fn insert(&mut self, mesh: UntypedAssetId, order: u64) {
        self.0
            .entry(mesh)
            .and_modify(|previous| *previous = (*previous).min(order))
            .or_insert(order);
    }

    fn key(&self, mesh: UntypedAssetId) -> (bool, u64, UntypedAssetId) {
        let order = self.0.get(&mesh).copied();
        (order.is_none(), order.unwrap_or(0), mesh)
    }
}

pub(super) fn install(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render.init_resource::<MeshOrder>();
    render.add_systems(ExtractSchedule, extract);
    // The normal/depth and color passes must resolve contact edges in the same
    // order. Run after Bevy sorts, but before it builds instance/batch buffers.
    render.add_systems(
        Render,
        sort.in_set(RenderSystems::PhaseSort)
            .after(sort_binned_render_phase::<Opaque3d>)
            .after(sort_binned_render_phase::<AlphaMask3d>)
            .after(sort_binned_render_phase::<Opaque3dPrepass>)
            .after(sort_binned_render_phase::<AlphaMask3dPrepass>),
    );
}

fn extract(mut order: ResMut<MeshOrder>, meshes: Extract<Query<(&Mesh3d, &AnnotationOrder)>>) {
    // Reuse allocation, but never retain a retired scene's asset identities.
    order.0.clear();
    for (mesh, index) in &meshes {
        order.insert(mesh.id().untyped(), index.0);
    }
}

fn order_phase<P: BinnedPhaseItem>(
    phases: &mut ViewBinnedRenderPhases<P>,
    order: &MeshOrder,
    mesh: impl Fn(&P::BinKey) -> UntypedAssetId,
) {
    for phase in phases.values_mut() {
        phase
            .batchable_meshes
            .sort_by(|(_, a), _, (_, b), _| order.key(mesh(a)).cmp(&order.key(mesh(b))));
        phase
            .unbatchable_meshes
            .sort_by(|(_, a), _, (_, b), _| order.key(mesh(a)).cmp(&order.key(mesh(b))));
    }
}

fn sort(
    order: Res<MeshOrder>,
    mut opaque: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    mut mask: ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    mut prepass: ResMut<ViewBinnedRenderPhases<Opaque3dPrepass>>,
    mut mask_prepass: ResMut<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
) {
    if order.0.is_empty() {
        return;
    }
    order_phase(&mut opaque, &order, |bin| bin.asset_id);
    order_phase(&mut mask, &order, |bin| bin.asset_id);
    order_phase(&mut prepass, &order, |bin| bin.asset_id);
    order_phase(&mut mask_prepass, &order, |bin| bin.asset_id);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests;
#[cfg(test)]
mod tests;
