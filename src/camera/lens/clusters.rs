//! Bound light-clustering CPU work for multi-face angular views.
//! Cluster cells are a conservative acceleration structure, not light samples.
//! Keep Bevy's stable CPU assignment and preserve explicit custom configurations.
use super::{LensCamera, LensFace};
use bevy::{
    light::{
        cluster::{ClusterConfig, ClusterFarZMode, ClusterZConfig},
        LightProbe, RectLight,
    },
    prelude::*,
};

#[derive(Component)]
struct Saved(Option<ClusterConfig>);

pub(super) fn install(app: &mut App) {
    app.add_systems(Update, select.after(super::super::CameraBuildSystems::Lens));
}

fn default_grid(config: Option<&ClusterConfig>) -> bool {
    config.is_none_or(|c| {
        matches!(
            c,
            ClusterConfig::FixedZ {
                total: 4096,
                z_slices: 24,
                z_config: ClusterZConfig {
                    first_slice_depth: 5.0,
                    far_z_mode: ClusterFarZMode::MaxClusterableObjectRange
                },
                dynamic_resizing: true,
            }
        )
    })
}

pub(crate) fn grid(camera_count: usize) -> ClusterConfig {
    ClusterConfig::FixedZ {
        // A logical angular camera fans out into up to five raster views.
        // Bound the rig's CPU clustering/upload work rather than assigning
        // each face the budget of a standalone pinhole camera. Cluster cells
        // remain conservative and keep every intersecting light/probe/decal.
        total: 128 / camera_count.clamp(1, 4).next_power_of_two() as u32,
        z_slices: 8,
        z_config: ClusterZConfig::default(),
        dynamic_resizing: true,
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn select(
    mut commands: Commands,
    config: Res<crate::app::BevyZeroverseConfig>,
    mode: Res<crate::render::RenderMode>,
    objects: Query<
        (),
        Or<(
            With<PointLight>,
            With<SpotLight>,
            With<RectLight>,
            With<LightProbe>,
            With<bevy::light::ClusteredDecal>,
        )>,
    >,
    mut views: Query<
        (Entity, Option<&mut ClusterConfig>, Option<&Saved>),
        Or<(With<LensCamera>, With<LensFace>)>,
    >,
) {
    let active = !config.room_schematic
        && *mode == crate::render::RenderMode::Color
        // Bound fragment work in high-light-complexity custom scenes. Cells
        // remain conservative: no light, probe, or decal is dropped.
        && objects.iter().take(65).count() <= 64;
    for (entity, current, saved) in &mut views {
        if !active {
            if let Some(saved) = saved {
                let mut entity = commands.entity(entity);
                if let Some(original) = saved.0 {
                    entity.insert(original);
                } else {
                    entity.remove::<ClusterConfig>();
                }
                entity.remove::<Saved>();
            }
        } else if saved.is_none() && default_grid(current.as_deref()) {
            commands
                .entity(entity)
                .insert((Saved(current.as_deref().copied()), grid(config.num_cameras)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn angular_capture_preserves_custom_grids_and_restores_annotation_modes() {
        let mut app = App::new();
        app.insert_resource(crate::app::BevyZeroverseConfig {
            camera_grid: true,
            num_cameras: 4,
            ..default()
        });
        app.insert_resource(crate::render::RenderMode::Color);
        install(&mut app);
        let view = app.world_mut().spawn(LensCamera::test_camera()).id();
        let custom = app
            .world_mut()
            .spawn((LensCamera::test_camera(), ClusterConfig::Single))
            .id();
        app.update();
        assert!(matches!(
            app.world().get::<ClusterConfig>(view),
            Some(ClusterConfig::FixedZ {
                total: 32,
                z_slices: 8,
                ..
            })
        ));
        assert!(matches!(
            app.world().get::<ClusterConfig>(custom),
            Some(ClusterConfig::Single)
        ));
        app.world_mut()
            .resource_mut::<crate::app::BevyZeroverseConfig>()
            .headless = true;
        let copier = app
            .world_mut()
            .spawn(crate::io::image_copy::ImageCopier::test_idle())
            .id();
        app.update();
        assert!(matches!(
            app.world().get::<ClusterConfig>(view),
            Some(ClusterConfig::FixedZ { total: 32, .. })
        ));
        assert!(app.world().get::<Saved>(view).is_some());
        app.world_mut().despawn(copier);
        app.update();
        assert!(app.world().get::<Saved>(view).is_some());
        app.insert_resource(crate::render::RenderMode::Depth);
        app.update();
        assert!(app.world().get::<ClusterConfig>(view).is_none());
        app.insert_resource(crate::render::RenderMode::Color);
        app.update();
        for _ in 0..65 {
            app.world_mut().spawn(PointLight::default());
        }
        app.update();
        assert!(app.world().get::<ClusterConfig>(view).is_none());
    }
}
