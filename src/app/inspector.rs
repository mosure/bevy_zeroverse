//! Optional ECS diagnostics; the everyday editor is built from Bevy Feathers.
use super::*;
use bevy_egui::{egui, EguiContext, PrimaryEguiContext};
pub(super) fn panel(world: &mut World) {
    if !world
        .get_resource::<editor::model::EditorState>()
        .is_some_and(|s| s.debug)
    {
        return;
    }
    let Ok(context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single(world)
    else {
        return;
    };
    let mut context = context.clone();
    let mut open = true;
    egui::Window::new("Scene resource inspector").open(&mut open).default_pos([400.,70.]).default_width(420.).show(context.get_mut(),|ui|{
        ui.label("Resource edits apply on regeneration. These low-level ECS edits are not included in share links.");
        egui::ScrollArea::vertical().max_height(650.).show(ui,|ui|{
            use bevy_inspector_egui::bevy_inspector::ui_for_resource;
            match world.resource::<BevyZeroverseConfig>().scene_type {
                ZeroverseSceneType::Object=>ui_for_resource::<crate::scene::object::ZeroverseObjectSceneSettings>(world,ui),
                ZeroverseSceneType::Room=>ui_for_resource::<ZeroverseRoomSettings>(world,ui),
                ZeroverseSceneType::SemanticRoom=>ui_for_resource::<ZeroverseSemanticRoomSettings>(world,ui),
                _=>{ui.label("Generated objects and components:");}
            }
            ui.collapsing("World entities",|ui|bevy_inspector_egui::bevy_inspector::ui_for_world(world,ui));
        });
    });
    if !open {
        world.resource_mut::<editor::model::EditorState>().debug = false;
    }
}
