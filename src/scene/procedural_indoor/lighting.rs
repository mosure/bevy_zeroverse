//! Publish complete interactive irradiance volumes without delaying room display.
use super::{gi, IndoorEnvironment, IndoorGenerationStatus};
use bevy::{
    prelude::*,
    tasks::{Task, TaskPool, TaskPoolBuilder},
};

pub(super) fn prepare(
    scene: gi::BakeScene,
    settings: gi::BakeSettings,
    seed: u64,
) -> Task<gi::ProbeData> {
    // One bake at a time, including a canceled job still finishing its CPU work.
    // Replacement tasks are canceled on drop; no detached jobs or growing queue.
    static POOL: std::sync::OnceLock<TaskPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        TaskPoolBuilder::new()
            .num_threads(1)
            .thread_name("indoor-lighting".into())
            .build()
    })
    .spawn(async move { scene.bake(settings, seed) })
}

#[derive(Resource)]
pub(super) struct PendingLighting {
    pub root: Entity,
    pub task: Task<gi::ProbeData>,
}

pub(super) fn finish(
    mut commands: Commands,
    job: Option<ResMut<PendingLighting>>,
    roots: Query<(), With<crate::scene::ZeroverseScene>>,
    mut images: ResMut<Assets<Image>>,
    mut generation: ResMut<IndoorGenerationStatus>,
    environment: Option<ResMut<IndoorEnvironment>>,
) {
    let Some(mut job) = job else {
        return;
    };
    if !roots.contains(job.root) {
        generation.lighting_pending = false;
        commands.remove_resource::<PendingLighting>();
        return;
    }
    let Some(data) = bevy::tasks::block_on(bevy::tasks::poll_once(&mut job.task)) else {
        return;
    };
    commands.spawn((
        Name::new("indoor_diffuse_irradiance"),
        bevy::light::IrradianceVolume {
            voxels: images.add(data.image()),
            intensity: 1.0,
            ..default()
        },
        data.transform(),
        ChildOf(job.root),
    ));
    commands.insert_resource(data.statistics);
    commands.remove_resource::<PendingLighting>();
    if let Some(mut environment) = environment {
        environment.has_gi = true;
    }
    generation.lighting_pending = false;
}

/// Publish the probe only after its GPU bake has been encoded and the room root
/// has passed transform propagation. Capture keeps waiting for this publication.
#[derive(Resource)]
pub(super) struct PendingGpuLighting {
    pub root: Entity,
    pub image: Handle<Image>,
    pub transform: Transform,
    pub readiness: gi::gpu::GiGpuReadiness,
}

pub(super) fn finish_gpu(
    mut commands: Commands,
    pending: Option<Res<PendingGpuLighting>>,
    roots: Query<(), With<crate::scene::ZeroverseScene>>,
    mut generation: ResMut<IndoorGenerationStatus>,
) {
    let Some(pending) = pending else {
        return;
    };
    if !roots.contains(pending.root) {
        commands.remove_resource::<PendingGpuLighting>();
        generation.lighting_pending = false;
        return;
    }
    if pending.is_added() || !pending.readiness.ready() {
        return;
    }
    commands.spawn((
        Name::new("indoor_diffuse_irradiance"),
        bevy::light::IrradianceVolume {
            voxels: pending.image.clone(),
            intensity: 1.,
            ..default()
        },
        pending.transform,
        ChildOf(pending.root),
    ));
    commands.remove_resource::<PendingGpuLighting>();
    generation.lighting_pending = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ZeroverseScene;

    fn fixture(ready: bool) -> (App, Entity, Transform) {
        let mut app = App::new();
        app.add_plugins(bevy::transform::TransformPlugin);
        app.add_systems(PreUpdate, finish_gpu);
        app.insert_resource(IndoorGenerationStatus {
            pending: false,
            lighting_pending: true,
        });
        let parent = Transform::from_xyz(3., 1., -2.).with_rotation(Quat::from_rotation_y(0.7));
        let root = app.world_mut().spawn((ZeroverseScene, parent)).id();
        let local = Transform::from_xyz(0., 2., 1.).with_scale(Vec3::new(8., 4., 7.));
        app.insert_resource(PendingGpuLighting {
            root,
            image: Handle::default(),
            transform: local,
            readiness: if ready {
                gi::gpu::GiGpuReadiness::encoded_for_test()
            } else {
                default()
            },
        });
        (app, root, parent.mul_transform(local))
    }

    fn volumes(app: &mut App) -> usize {
        app.world_mut()
            .query::<&bevy::light::IrradianceVolume>()
            .iter(app.world())
            .count()
    }

    #[test]
    fn even_a_prefetched_probe_waits_for_parent_propagation_and_publishes_once() {
        let (mut app, root, expected) = fixture(true);
        app.update();
        assert_eq!(volumes(&mut app), 0);
        assert!(app.world().resource::<IndoorGenerationStatus>().busy());
        app.update();
        assert_eq!(volumes(&mut app), 1);
        let world = app.world_mut();
        let (parent, transform) = world
            .query_filtered::<(&ChildOf, &GlobalTransform), With<bevy::light::IrradianceVolume>>()
            .single(world)
            .unwrap();
        assert_eq!(parent.parent(), root);
        assert!(transform
            .to_matrix()
            .abs_diff_eq(expected.to_matrix(), 1e-5));
        assert!(!app.world().contains_resource::<PendingGpuLighting>());
        assert!(!app.world().resource::<IndoorGenerationStatus>().busy());
        app.update();
        assert_eq!(volumes(&mut app), 1);
    }

    #[test]
    fn an_unencoded_or_retired_probe_cannot_release_capture() {
        let (mut app, root, _) = fixture(false);
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(volumes(&mut app), 0);
        assert!(app.world().resource::<IndoorGenerationStatus>().busy());
        app.world_mut().despawn(root);
        app.update();
        assert_eq!(volumes(&mut app), 0);
        assert!(!app.world().contains_resource::<PendingGpuLighting>());
        assert!(!app.world().resource::<IndoorGenerationStatus>().busy());
    }
}
