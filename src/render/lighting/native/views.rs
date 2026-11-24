use super::SolariPreviewStatus;
use crate::{
    camera::{lens::LensFace, ProcessedEditorCameraMarker, ZeroverseCamera},
    scene::SceneLoadedEvent,
};
use bevy::{
    camera::{CameraMainTextureUsages, Hdr},
    core_pipeline::prepass::{
        DeferredPrepass, DeferredPrepassDoubleBuffer, DepthPrepass, DepthPrepassDoubleBuffer,
        MotionVectorPrepass,
    },
    ecs::query::QueryData,
    prelude::*,
    render::render_resource::TextureUsages,
};
use bevy_solari::realtime::SolariLighting;

#[derive(Component)]
pub(super) struct ManagedView {
    added_hdr: bool,
    added_deferred: bool,
    added_depth: bool,
    added_motion: bool,
    added_storage: bool,
    previous_msaa: Msaa,
    had_deferred_history: bool,
    had_depth_history: bool,
}

#[derive(QueryData)]
#[query_data(mutable)]
pub(super) struct PreviewView {
    entity: Entity,
    camera: &'static Camera,
    usage: &'static mut CameraMainTextureUsages,
    msaa: &'static Msaa,
    hdr: Has<Hdr>,
    deferred: Has<DeferredPrepass>,
    depth: Has<DepthPrepass>,
    motion: Has<MotionVectorPrepass>,
    deferred_history: Has<DeferredPrepassDoubleBuffer>,
    depth_history: Has<DepthPrepassDoubleBuffer>,
    lighting: Option<&'static mut SolariLighting>,
}

#[allow(clippy::type_complexity)]
pub(super) fn synchronize(
    mut commands: Commands,
    status: Res<SolariPreviewStatus>,
    mut loaded: MessageReader<SceneLoadedEvent>,
    mut views: Query<
        PreviewView,
        (
            With<Camera3d>,
            Or<(
                With<ProcessedEditorCameraMarker>,
                With<ZeroverseCamera>,
                With<LensFace>,
            )>,
        ),
    >,
) {
    let cut = loaded.read().count() != 0;
    if !status.active {
        return;
    }
    for mut view in &mut views {
        // A physical fisheye face or supersampled view can be much larger
        // than the output. WGPU validates each reservoir binding separately.
        let restir = status.temporal_reuse
            && view
                .camera
                .physical_viewport_size()
                .is_some_and(|s| u64::from(s.x) * u64::from(s.y) <= status.max_reservoir_pixels);
        if let Some(lighting) = view.lighting.as_deref_mut() {
            if cut || lighting.restir != restir {
                lighting.restir = restir;
                lighting.reset = true;
            }
            continue;
        }
        let owned = ManagedView {
            added_hdr: !view.hdr,
            added_deferred: !view.deferred,
            added_depth: !view.depth,
            added_motion: !view.motion,
            added_storage: !view.usage.0.contains(TextureUsages::STORAGE_BINDING),
            previous_msaa: *view.msaa,
            had_deferred_history: view.deferred_history,
            had_depth_history: view.depth_history,
        };
        if owned.added_storage {
            view.usage.0 |= TextureUsages::STORAGE_BINDING;
        }
        let mut e = commands.entity(view.entity);
        if *view.msaa != Msaa::Off {
            e.insert(Msaa::Off);
        }
        e.insert((
            SolariLighting {
                restir,
                ..default()
            },
            owned,
        ));
    }
}

// Run before Studio AA and lens inheritance so restored TAA can acquire its own
// prepasses the same frame, including when the user changes both settings at once.
pub(super) fn release(
    mut commands: Commands,
    status: Res<SolariPreviewStatus>,
    mut views: Query<(Entity, &ManagedView, &mut CameraMainTextureUsages, &Msaa)>,
) {
    if status.active {
        return;
    }
    for (entity, owned, mut usage, msaa) in &mut views {
        let mut e = commands.entity(entity);
        e.remove::<(ManagedView, SolariLighting)>();
        if owned.had_deferred_history {
            e.insert(DeferredPrepassDoubleBuffer);
        } else {
            e.remove::<DeferredPrepassDoubleBuffer>();
        }
        if owned.had_depth_history {
            e.insert(DepthPrepassDoubleBuffer);
        } else {
            e.remove::<DepthPrepassDoubleBuffer>();
        }
        if owned.added_hdr {
            e.remove::<Hdr>();
        }
        if owned.added_deferred {
            e.remove::<DeferredPrepass>();
        }
        if owned.added_depth {
            e.remove::<DepthPrepass>();
        }
        if owned.added_motion {
            e.remove::<MotionVectorPrepass>();
        }
        if owned.added_storage {
            usage.0.remove(TextureUsages::STORAGE_BINDING);
        }
        if *msaa == Msaa::Off && owned.previous_msaa != Msaa::Off {
            e.insert(owned.previous_msaa);
        }
    }
}
