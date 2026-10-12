//! Retry initial 2D specialization after asynchronous mesh preparation. Bevy
//! 0.20's Material2d queue retries missing materials, but can skip a mesh that
//! was not yet prepared without retaining it in the pending queue.
use bevy::{
    prelude::*,
    render::{
        camera::DirtySpecializations,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        sync_world::MainEntity,
        Render, RenderApp, RenderSystems,
    },
    sprite_render::{specialize_material2d_meshes, SpecializedMaterial2dPipelineCache},
};
#[derive(Component, Clone, ExtractComponent)]
#[extract_app(RenderApp)]
pub(super) struct SchematicBatch;
pub(in crate::app::editor) fn install(app: &mut App) {
    app.add_plugins(ExtractComponentPlugin::<SchematicBatch>::default());
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(
            Render,
            retry
                .in_set(RenderSystems::Specialize)
                .before(specialize_material2d_meshes),
        );
    }
}
fn retry(
    batches: Query<&MainEntity, With<SchematicBatch>>,
    cache: Res<SpecializedMaterial2dPipelineCache>,
    mut dirty: ResMut<DirtySpecializations>,
) {
    for entity in &batches {
        if !cache.values().any(|view| view.contains_key(entity)) {
            dirty.changed_renderables.insert(*entity);
        }
    }
}
