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
