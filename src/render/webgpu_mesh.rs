//! Avoid querying every WebGPU device limit for every ordinary mesh draw.
//!
//! Bevy 0.20's SetMeshBindGroup asks `device.limits()` whether skin uniforms
//! require dynamic offsets, even on unskinned meshes. On WebGPU this crosses
//! the JS boundary for dozens of immutable properties per draw. Fisheye shadow
//! cascades magnify it. Bind ordinary meshes using the existing Bevy bind group;
//! delegate skinning, morph targets and lightmaps to Bevy unchanged. No shaders,
//! geometry, draw ordering, shadow coverage or dependency patches are changed.
use std::any::TypeId;

use bevy::{
    core_pipeline::{core_3d::*, deferred::*, prepass::*},
    ecs::{
        query::ROQueryItem,
        system::{ReadOnlySystemParam, SystemParamItem},
    },
    pbr::*,
    prelude::*,
    render::{render_phase::*, RenderApp},
};

type FastMaterial = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    OrdinaryMeshBindings<2>,
    SetMaterialBindGroup<3>,
    DrawMesh,
);
type FastPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    OrdinaryMeshBindings<2>,
    SetMaterialBindGroup<3>,
    DrawMesh,
);
type FastDepth = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    OrdinaryMeshBindings<2>,
    SetPrepassEmptyMaterialBindGroup<3>,
    DrawMesh,
);

pub(super) fn install(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    let world = render.world_mut();
    replace::<Opaque3d, DrawMaterial, FastMaterial>(world);
    replace::<AlphaMask3d, DrawMaterial, FastMaterial>(world);
    replace::<Transmissive3d, DrawMaterial, FastMaterial>(world);
    replace::<Transparent3d, DrawMaterial, FastMaterial>(world);
    replace::<Opaque3dPrepass, DrawPrepass, FastPrepass>(world);
    replace::<AlphaMask3dPrepass, DrawPrepass, FastPrepass>(world);
    replace::<Opaque3dDeferred, DrawPrepass, FastPrepass>(world);
    replace::<AlphaMask3dDeferred, DrawPrepass, FastPrepass>(world);
    replace::<Opaque3dPrepass, DrawDepthOnlyPrepass, FastDepth>(world);
    replace::<Shadow, DrawPrepass, FastPrepass>(world);
    replace::<Shadow, DrawDepthOnlyPrepass, FastDepth>(world);
}

fn replace<P, Old, New>(world: &mut World)
where
    P: PhaseItem + 'static,
    Old: 'static,
    New: RenderCommand<P> + Send + Sync + 'static,
    New::Param: ReadOnlySystemParam,
{
    let Some(functions) = world.get_resource::<DrawFunctions<P>>() else {
        return;
    };
    let Some(id) = functions.read().get_id::<Old>() else {
        return;
    };
    let command = RenderCommandState::<P, New>::new(world);
    // Retain the original id, including references already held by materials.
    world.resource::<DrawFunctions<P>>().write().draw_functions[id.0 as usize] = Box::new(command);
}

struct OrdinaryMeshBindings<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for OrdinaryMeshBindings<I> {
    type Param = <SetMeshBindGroup<I> as RenderCommand<P>>::Param;
    type ViewQuery = <SetMeshBindGroup<I> as RenderCommand<P>>::ViewQuery;
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        view: ROQueryItem<'w, '_, Self::ViewQuery>,
        entity: Option<()>,
        param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let main = item.main_entity();
        let Some(mesh) = param.2.render_mesh_queue_data(main) else {
            return RenderCommandResult::Success;
        };
        let asset = mesh.mesh_asset_id();
        let slabs = param.5.mesh_slabs(&asset);
        let uniform_morph = matches!(&*param.4, MorphIndices::Uniform { current, .. } if current.contains_key(&main));
        let storage_morph = slabs.is_some_and(|s| s.morph_target_slab_id.is_some());
        if param.3.skin_byte_offset(main).is_some()
            || uniform_morph
            || storage_morph
            || mesh.lightmap_slab_index().is_some()
        {
            return <SetMeshBindGroup<I> as RenderCommand<P>>::render(
                item, view, entity, param, pass,
            );
        }
        let (_, groups, _, _, _, _, _, fallback) = param;
        let metadata = slabs
            .and_then(|s| s.metadata_slab_id)
            .unwrap_or(fallback.slab_id);
        let phase = match groups.into_inner() {
            MeshBindGroups::CpuPreprocessing(phase) => Some(phase),
            MeshBindGroups::GpuPreprocessing(phases) => phases.get(&TypeId::of::<P>()),
        };
        let Some(phase) = phase else {
            return RenderCommandResult::Success;
        };
        let Some(binding) = phase.get(
            metadata,
            None,
            false,
            MeshMorphBindGroupKey::NoMorphTargets,
            view,
        ) else {
            // Preserve Bevy's diagnostics for an unexpectedly missing binding.
            return RenderCommandResult::Failure("ordinary mesh bind group missing");
        };
        let offset;
        let offsets = if let PhaseItemExtraIndex::DynamicOffset(value) = item.extra_index() {
            offset = [value];
            &offset[..]
        } else {
            &[]
        };
        pass.set_bind_group(I, binding, offsets);
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests;
