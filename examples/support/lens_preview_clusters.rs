//! Explicit benchmark overrides must also affect cameras using Bevy's implicit
//! default. Record the actual constructed grid, not just the requested number.
use bevy::{
    light::{
        cluster::{ClusterConfig, Clusters},
        SimulationLightSystems,
    },
    prelude::*,
};

#[derive(Resource)]
struct Budget(u32);

pub fn install(app: &mut App, total: u32) {
    assert!((1..=4096).contains(&total));
    app.insert_resource(Budget(total)).add_systems(
        PostUpdate,
        apply.before(SimulationLightSystems::AssignLightsToClusters),
    );
}

fn apply(mut commands: Commands, budget: Res<Budget>, views: Query<Entity, With<Camera3d>>) {
    for entity in &views {
        commands.entity(entity).insert(if budget.0 == 1 {
            ClusterConfig::Single
        } else {
            ClusterConfig::FixedZ {
                total: budget.0,
                z_slices: budget.0.min(8),
                z_config: default(),
                dynamic_resizing: true,
            }
        });
    }
}

pub fn snapshot(app: &mut App) -> Vec<serde_json::Value> {
    let requested = app.world().get_resource::<Budget>().map(|b| b.0);
    app.world_mut().query::<(&Camera, Option<&ClusterConfig>, &Clusters)>()
        .iter(app.world()).map(|(camera, config, grid)| {
            if camera.is_active {
                if let Some(requested) = requested {
                    assert!(matches!(config, Some(ClusterConfig::FixedZ {total, ..}) if *total == requested)
                        || matches!(config, Some(ClusterConfig::Single) if requested == 1),
                        "benchmark cluster override was not applied");
                    assert!(grid.dimensions.element_product() <= requested);
                }
            }
            serde_json::json!({"active":camera.is_active,"config":format!("{config:?}"),
                "dimensions":grid.dimensions.to_array(),"count":grid.dimensions.element_product()})
        }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn override_includes_implicit_defaults_and_later_spawned_cameras() {
        let mut app = App::new();
        app.insert_resource(Budget(256)).add_systems(Update, apply);
        let first = app.world_mut().spawn(Camera3d::default()).id();
        assert!(app.world().get::<ClusterConfig>(first).is_none());
        app.update();
        let second = app
            .world_mut()
            .spawn((Camera3d::default(), ClusterConfig::Single))
            .id();
        app.update();
        for entity in [first, second] {
            assert!(matches!(
                app.world().get::<ClusterConfig>(entity),
                Some(ClusterConfig::FixedZ {
                    total: 256,
                    z_slices: 8,
                    ..
                })
            ));
        }
    }
}
