//! Filter the opaque HDR background once for each view containing rough glass.
//! Bevy runs Core3d to completion for each camera. Equal-sized views can reuse
//! the pyramid because both its construction and consumption are in that scope.
pub(super) mod kernel;
mod pyramid;

use super::GlassFilter;
use crate::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
use bevy::{
    asset::UntypedAssetId,
    camera::MainPassResolutionOverride,
    core_pipeline::{
        core_3d::main_opaque_pass_3d, prepass::ViewPrepassTextures, Core3d, Core3dSystems,
    },
    ecs::schedule::ScheduleLabel,
    pbr::{
        main_transmissive_pass_3d, RenderMaterialInstances, ScreenSpaceTransmission,
        Transmissive3d, ViewTransmissionTexture,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_phase::ViewSortedRenderPhases,
        renderer::{RenderContext, RenderDevice, ViewQuery},
        view::{ExtractedView, ViewTarget},
        Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use pyramid::{Filters, Pyramid, Source};
use std::collections::{HashMap, HashSet};

pub(super) fn build(app: &mut App) {
    if !app.world().contains_resource::<Assets<StandardMaterial>>() {
        return;
    }
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render
        .init_resource::<FrostedMaterials>()
        .init_resource::<Work>()
        .add_systems(RenderStartup, |mut work: ResMut<Work>| {
            *work = Work::default()
        })
        .add_systems(ExtractSchedule, extract)
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareResourcesFlush))
        .add_systems(
            Core3d,
            filter_background
                .after(main_opaque_pass_3d)
                .before(main_transmissive_pass_3d)
                .in_set(Core3dSystems::MainPass),
        );
}

#[derive(Resource, Default)]
struct FrostedMaterials {
    enabled: bool,
    rough: HashSet<UntypedAssetId>,
}

fn extract(
    config: Extract<Option<Res<BevyZeroverseConfig>>>,
    filter: Extract<Option<Res<GlassFilter>>>,
    mut events: Extract<MessageReader<AssetEvent<StandardMaterial>>>,
    materials: Extract<Res<Assets<StandardMaterial>>>,
    mut state: ResMut<FrostedMaterials>,
) {
    let enabled = config.as_ref().is_some_and(|c| {
        c.scene_type == ZeroverseSceneType::ProceduralIndoor
            && c.indoor_quality.specular_transmission()
    }) && filter
        .as_deref()
        .is_some_and(|v| matches!(v, GlassFilter::Default));
    if !enabled || !state.enabled {
        state.enabled = enabled;
        state.rough.clear();
        if enabled {
            state.rough.extend(
                materials
                    .iter()
                    .filter(|(_, m)| needs_filter(m))
                    .map(|(id, _)| id.untyped()),
            );
        }
        events.clear();
        return;
    }
    for event in events.read() {
        match *event {
            AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                if materials.get(id).is_some_and(needs_filter) {
                    state.rough.insert(id.untyped());
                } else {
                    state.rough.remove(&id.untyped());
                }
            }
            AssetEvent::Removed { id } => {
                state.rough.remove(&id.untyped());
            }
            _ => {}
        }
    }
}

fn needs_filter(material: &StandardMaterial) -> bool {
    material.specular_transmission > 0.0 && material.perceptual_roughness >= 0.10
}

#[derive(Default)]
struct ViewWork {
    size: UVec2,
    source: Option<Source>,
    live: bool,
}
#[derive(Resource, Default)]
struct Work {
    filters: Option<Filters>,
    pyramids: HashMap<UVec2, Pyramid>,
    views: HashMap<Entity, ViewWork>,
}

#[allow(clippy::type_complexity)]
fn prepare(
    materials: Res<FrostedMaterials>,
    instances: Option<Res<RenderMaterialInstances>>,
    phases: Option<Res<ViewSortedRenderPhases<Transmissive3d>>>,
    device: Res<RenderDevice>,
    mut work: ResMut<Work>,
    mut views: Query<
        (
            Entity,
            &ExtractedCamera,
            &ExtractedView,
            &ScreenSpaceTransmission,
            &Msaa,
            &mut ViewTransmissionTexture,
        ),
        (
            Without<MainPassResolutionOverride>,
            Without<bevy::pbr::ExtractedAtmosphere>,
        ),
    >,
) {
    let Work {
        filters,
        pyramids,
        views: active,
    } = &mut *work;
    for entry in active.values_mut() {
        entry.live = false;
    }
    if let (true, Some(instances), Some(phases)) = (
        materials.enabled && !materials.rough.is_empty(),
        instances,
        phases,
    ) {
        for (entity, camera, view, transmission, msaa, mut texture) in &mut views {
            let Some(size) = camera.physical_target_size else {
                continue;
            };
            // Bevy copies a new background between multi-step draws. Each step
            // must see its own background, so keep the original path there.
            if transmission.steps != 1
                || *msaa != Msaa::Off
                || view.target_format != wgpu::TextureFormat::Rgba16Float
                || camera.schedule != Core3d.intern()
                || size.min_element() == 0
                || size.max_element() < 16
                || view.viewport != UVec4::new(0, 0, size.x, size.y)
            {
                continue;
            }
            let Some(phase) = phases.get(&view.retained_view_entity) else {
                continue;
            };
            if !phase.items.values().any(|item| {
                instances
                    .instances
                    .get(&item.entity.1)
                    .is_some_and(|m| materials.rough.contains(&m.asset_id))
            }) {
                continue;
            }
            let raw = device.wgpu_device();
            let filters = filters.get_or_insert_with(|| Filters::new(raw));
            let pyr = pyramids
                .entry(size)
                .or_insert_with(|| Pyramid::new(raw, filters, size.x, size.y));
            texture.texture = pyr.texture.clone();
            texture.view = pyr.all.clone();
            let entry = active.entry(entity).or_default();
            entry.size = size;
            entry.live = true;
        }
    }
    active.retain(|_, v| v.live);
    pyramids.retain(|size, _| active.values().any(|v| v.size == *size));
}

fn filter_background(
    view: ViewQuery<(&ViewTarget, Option<&ViewPrepassTextures>)>,
    device: Res<RenderDevice>,
    mut work: ResMut<Work>,
    mut context: RenderContext,
) {
    let entity = view.entity();
    let Work {
        filters,
        pyramids,
        views,
    } = &mut *work;
    let Some(entry) = views.get_mut(&entity) else {
        return;
    };
    let (target, prepass) = view.into_inner();
    let depth = prepass.and_then(ViewPrepassTextures::depth_only_view);
    pyramids[&entry.size].encode(
        device.wgpu_device(),
        filters.as_ref().unwrap(),
        context.command_encoder(),
        target.main_texture_view(),
        depth,
        &mut entry.source,
    );
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
