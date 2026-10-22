//! The preceding pipelined render can submit after the main-world activity
//! gate ran. Recheck at extraction, after that render returned, before preparing
//! another identical set of views. Never activate a camera late.
use super::*;
use bevy::camera::visibility::RenderLayers;
use bevy::render::{
    camera::{extract_cameras, ExtractedCamera, MipBias, TemporalJitter},
    sync_world::RenderEntity,
    view::{
        ExtractedView, NoIndirectDrawing, RenderVisibleEntities, ResolvedCompositingSpace,
        ViewUniformOffset,
    },
    Extract, ExtractSchedule, RenderApp,
};

pub(super) fn install(app: &mut App) {
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(ExtractSchedule, suppress.after(extract_cameras));
    }
}

#[allow(clippy::type_complexity)]
fn suppress(
    mut commands: Commands,
    args: Extract<Res<BevyZeroverseConfig>>,
    owners: Extract<Query<(Entity, &ImageCopier, &RenderEntity), With<ZeroverseCamera>>>,
    faces: Extract<Query<(&crate::camera::lens::LensFace, &RenderEntity)>>,
) {
    if !args.headless || args.editor {
        return;
    }
    let mut completed = bevy::platform::collections::HashSet::new();
    for (owner, copier, render) in &owners {
        if !render_active(&args, true, Some(copier)) {
            completed.insert(owner);
            remove(&mut commands, render.id());
        }
    }
    for (face, render) in &faces {
        if completed.contains(&face.owner) {
            remove(&mut commands, render.id());
        }
    }
}

fn remove(commands: &mut Commands, entity: Entity) {
    // Match Bevy's inactive-camera extraction. Retain the entity and capture
    // attachments; normal extraction restores these components for a new epoch.
    commands.entity(entity).remove::<(
        ExtractedCamera,
        ExtractedView,
        ResolvedCompositingSpace,
        TemporalJitter,
        MipBias,
        RenderLayers,
        Projection,
        NoIndirectDrawing,
        ViewUniformOffset,
        RenderVisibleEntities,
    )>();
}

#[cfg(test)]
#[path = "submitted_tests.rs"]
mod tests;
