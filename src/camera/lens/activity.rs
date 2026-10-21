//! Annotation rays run in the root ground-truth schedule. A logical angular
//! annotation view still needs extraction/calibration, but no Core3d raster.
use super::LensCamera;
use bevy::{
    ecs::schedule::ScheduleLabel,
    light::cluster::ClusterConfig,
    prelude::*,
    render::{camera::CameraRenderGraph, RenderApp},
};

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct AnnotationView;
#[derive(Component)]
struct ColorGraph {
    graph: CameraRenderGraph,
    clusters: Option<ClusterConfig>,
}

pub(super) fn install(app: &mut App) {
    app.add_systems(PostUpdate, select.before(bevy::camera::CameraUpdateSystems));
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_schedule(Schedule::new(AnnotationView));
    }
}
#[allow(clippy::type_complexity)] // Optional Bevy view settings restored on RGB entry.
fn select(
    mut commands: Commands,
    mode: Res<crate::render::RenderMode>,
    mut views: Query<
        (
            Entity,
            &mut CameraRenderGraph,
            Option<&ColorGraph>,
            Option<&mut ClusterConfig>,
        ),
        With<LensCamera>,
    >,
) {
    for (entity, mut graph, saved, clusters) in &mut views {
        if *mode == crate::render::RenderMode::Color {
            if let Some(saved) = saved {
                *graph = saved.graph.clone();
                if let (Some(mut clusters), Some(original)) = (clusters, saved.clusters) {
                    *clusters = original;
                }
                commands.entity(entity).remove::<ColorGraph>();
            }
        } else if saved.is_none() {
            let original = clusters.as_deref().copied();
            if let Some(mut clusters) = clusters {
                *clusters = ClusterConfig::None;
            }
            commands.entity(entity).insert(ColorGraph {
                graph: graph.clone(),
                clusters: original,
            });
            graph.set(AnnotationView);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn annotation_views_skip_raster_and_restore_the_owners_graph() {
        use crate::render::RenderMode;
        let mut app = App::new();
        app.insert_resource(RenderMode::Color)
            .add_systems(Update, select);
        let original = CameraRenderGraph::new(bevy::core_pipeline::schedule::Core3d);
        let owner = app
            .world_mut()
            .spawn((
                LensCamera::test_camera(),
                original.clone(),
                ClusterConfig::Single,
            ))
            .id();
        let pinhole = app.world_mut().spawn(original.clone()).id();
        for mode in [
            RenderMode::Color,
            RenderMode::Depth,
            RenderMode::Normal,
            RenderMode::CoVisibility,
            RenderMode::OpticalFlow,
            RenderMode::Color,
        ] {
            app.insert_resource(mode.clone());
            app.update();
            let graph = &app.world().get::<CameraRenderGraph>(owner).unwrap().0;
            assert_eq!(
                *graph,
                if mode == RenderMode::Color {
                    original.0
                } else {
                    AnnotationView.intern()
                }
            );
            assert_eq!(
                app.world().get::<CameraRenderGraph>(pinhole).unwrap().0,
                original.0
            );
            assert!(
                matches!(
                    app.world().get::<ClusterConfig>(owner).unwrap(),
                    ClusterConfig::Single
                ) == (mode == RenderMode::Color)
            );
        }
    }
}
