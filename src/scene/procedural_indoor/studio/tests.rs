//! Exercise the event boundary with actual prepared rooms and the real playback
//! system. Seed failures and slow work are injected without timing dependencies.
use super::*;
use crate::camera::{Playback, PlaybackMode};
use crate::sample::failure::{CaptureError, SceneRejectionKind};
use crate::scene::procedural_indoor::preparation::{prepare_indoor_typed, StagedAssets};

fn setup() -> (App, Entity) {
    let mut app = App::new();
    let mut config = BevyZeroverseConfig {
        indoor_seed: Some(70),
        indoor_quality: IndoorQuality::Portable,
        indoor_density: 0.1,
        indoor_human_density: 0.,
        num_cameras: 1,
        regenerate_ms: 1000,
        editor: true,
        headless: false,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    };
    config.indoor_gpu_materials = false;
    let settings = ZeroverseSceneSettings {
        num_cameras: 1,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    };
    app.add_plugins(bevy::app::TaskPoolPlugin::default())
        .add_message::<RegenerateSceneEvent>()
        .add_message::<SceneLoadedEvent>()
        .insert_resource(config)
        .insert_resource(settings)
        .insert_resource(IndoorSequence {
            base_seed: Some(70),
            configured_seed: Some(70),
            index: 1,
        })
        .init_resource::<PendingIndoor>()
        .init_resource::<IndoorGenerationStatus>()
        .init_resource::<gi::IndoorGiSettings>()
        .init_resource::<materials::gpu::MaterialComputeAvailable>()
        .init_resource::<preparation::residency::FutureAssets>()
        .init_resource::<preparation::IndoorPrefetch>()
        .init_resource::<Assets<Image>>()
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<StandardMaterial>>()
        .init_resource::<Time<Real>>()
        .init_resource::<Time>()
        .insert_resource(Playback {
            mode: PlaybackMode::Loop,
            progress: 0.2,
            speed: 0.1,
            ..default()
        })
        .add_systems(PreUpdate, (advance, regenerate).chain())
        .add_systems(Update, crate::camera::update_camera_trajectory);
    let root = app.world_mut().spawn(ZeroverseScene).id();
    app.world_mut()
        .resource_mut::<PendingIndoor>()
        .studio
        .presented(Duration::ZERO);
    (app, root)
}

fn key(world: &World, seed: u64) -> Request {
    Request::new(
        seed,
        world.resource(),
        world.resource(),
        *world.resource::<gi::IndoorGiSettings>(),
    )
}
fn inject(world: &mut World, seed: u64, result: Result<preparation::PreparedIndoor, CaptureError>) {
    let request = key(world, seed);
    world.resource_mut::<PendingIndoor>().studio.candidate =
        Some((request, PreparationJob::completed(result)));
}
fn tick(app: &mut App, millis: u64) {
    let dt = Duration::from_millis(millis);
    app.world_mut().resource_mut::<Time<Real>>().advance_by(dt);
    app.world_mut().resource_mut::<Time>().advance_by(dt);
    app.update();
}

#[test]
fn automatic_playback_waits_for_legacy_scene_assets_and_capture_keeps_its_request_clock() {
    let mut app = App::new();
    app.add_message::<RegenerateSceneEvent>()
        .add_message::<SceneLoadedEvent>()
        .init_resource::<Time>()
        .insert_resource(BevyZeroverseConfig {
            headless: false,
            regenerate_ms: 1000,
            scene_type: ZeroverseSceneType::CornellCube,
            ..default()
        })
        .insert_resource(Playback {
            progress: 0.4,
            mode: PlaybackMode::Still,
            ..default()
        })
        .add_systems(Update, crate::camera::update_camera_trajectory);
    app.world_mut().write_message(RegenerateSceneEvent);
    app.update();
    assert_eq!(app.world().resource::<Playback>().progress, 0.4);
    app.world_mut().write_message(SceneLoadedEvent);
    app.update();
    assert_eq!(app.world().resource::<Playback>().progress, 0.);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .headless = true;
    app.world_mut().resource_mut::<Playback>().progress = 0.6;
    app.world_mut().write_message(RegenerateSceneEvent);
    app.update();
    assert_eq!(app.world().resource::<Playback>().progress, 0.);
}

#[test]
fn yielding_between_architectural_stages_preserves_every_mesh() {
    let scene = layout::IndoorManifest::generate(70, layout::IndoorLayout::Mixed, 0.15, 1).unwrap();
    let expected = envelope::construction::build(&scene);
    let yields = std::cell::Cell::new(0);
    let actual = bevy::tasks::block_on(envelope::construction::build_with_yields(&scene, || {
        yields.set(yields.get() + 1);
        futures_lite::future::yield_now()
    }));
    assert!(yields.get() >= scene.envelope.unwrap().walls.len() + 4);
    assert_eq!(actual.parts.len(), expected.parts.len());
    for (key, geometry) in actual.parts {
        let original = &expected.parts[&key];
        assert_eq!(geometry.positions, original.positions);
        assert_eq!(geometry.normals, original.normals);
        assert_eq!(geometry.uvs, original.uvs);
        assert_eq!(geometry.indices, original.indices);
    }
}

#[test]
fn skips_consecutive_rejections_then_presents_once_with_playback_and_truth_intact() {
    let (mut app, root) = setup();
    for seed in [71, 72] {
        let error =
            CaptureError::rejected(seed, SceneRejectionKind::CameraPlacement, "fixture".into());
        inject(app.world_mut(), seed, Err(error));
        tick(&mut app, 10);
        assert!(app.world().get_entity(root).is_ok());
        assert_eq!(app.world().resource::<IndoorSequence>().index, seed - 69);
        assert!(app
            .world()
            .resource::<Messages<RegenerateSceneEvent>>()
            .is_empty());
        assert!(!app.world().resource::<IndoorGenerationStatus>().busy());
        assert!(app.world().resource::<Playback>().progress > 0.2);
    }
    let world = app.world();
    let next = bevy::tasks::block_on(prepare_indoor_typed(
        73,
        world.resource(),
        world.resource(),
        *world.resource::<gi::IndoorGiSettings>(),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
    ))
    .unwrap();
    let expected = next.manifest.clone();
    inject(app.world_mut(), 73, Ok(next));
    type Fence = preparation::studio_upload::UploadFence;
    app.init_resource::<Fence>();
    // Simulate a submitted upload still occupying the device after the deadline.
    // Playback continues and the current root survives until the GPU acknowledges it.
    tick(&mut app, 1100);
    for _ in 0..6 {
        tick(&mut app, 10);
        assert!(app.world().get_entity(root).is_ok());
        assert!(app
            .world()
            .resource::<Messages<RegenerateSceneEvent>>()
            .is_empty());
    }
    assert!(!app.world().resource::<Fence>().ready());
    assert!(app.world().resource::<Playback>().progress > 0.3);
    for _ in 0..400 {
        app.world().resource::<Fence>().complete_for_test();
        tick(&mut app, 10);
        if app.world().get_entity(root).is_err() {
            break;
        }
    }
    assert!(app.world().get_entity(root).is_err());
    assert!(app.world().resource::<Time<Real>>().elapsed() >= Duration::from_secs(1));
    assert_eq!(app.world().resource::<layout::IndoorManifest>(), &expected);
    assert_eq!(app.world().resource::<Playback>().progress, 0.);
    assert!(app
        .world()
        .resource::<crate::sample::CaptureFailure>()
        .0
        .is_none());
    assert_eq!(app.world().resource::<IndoorSequence>().index, 4);
}

#[test]
fn deadline_with_unfinished_work_keeps_the_old_scene_and_disabling_releases_it() {
    let (mut app, root) = setup();
    let request = key(app.world(), 71);
    let task = PreparationJob::pending_for_test();
    app.world_mut()
        .resource_mut::<PendingIndoor>()
        .studio
        .candidate = Some((request, task));
    for _ in 0..8 {
        tick(&mut app, 200);
    }
    assert!(app
        .world()
        .resource::<Messages<RegenerateSceneEvent>>()
        .is_empty());
    assert!(app.world().resource::<Playback>().progress > 0.3);
    assert!(app.world().get_entity(root).is_ok());
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .candidate
        .is_some());
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .regenerate_ms = 0;
    tick(&mut app, 10);
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .candidate
        .is_none());
    assert_eq!(app.world().resource::<IndoorSequence>().index, 1);
}

#[test]
fn stalled_successor_expires_after_one_minute_without_retiring_the_displayed_room() {
    let (mut app, root) = setup();
    let seed = 71;
    let request = key(app.world(), seed);
    app.world_mut()
        .resource_mut::<PendingIndoor>()
        .studio
        .candidate = Some((request, PreparationJob::pending_for_test()));
    tick(&mut app, 1);
    tick(&mut app, 59_999);
    assert_eq!(app.world().resource::<IndoorSequence>().index, 1);
    tick(&mut app, 1);
    assert_eq!(
        app.world()
            .resource::<IndoorGenerationStatus>()
            .skipped_seed,
        Some(seed)
    );
    assert_eq!(app.world().resource::<IndoorSequence>().index, 2);
    assert!(app.world().get_entity(root).is_ok());
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .candidate
        .is_none());
    tick(&mut app, 1);
    assert_eq!(
        app.world()
            .resource::<PendingIndoor>()
            .studio
            .candidate
            .as_ref()
            .unwrap()
            .0
            .seed,
        72
    );
}

#[test]
fn invalid_configuration_does_not_spin_through_seeds_and_manual_requests_still_work() {
    let (mut app, root) = setup();
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .indoor_camera = Some("{".into());
    inject(app.world_mut(), 71, Err("invalid configuration".into()));
    for _ in 0..4 {
        tick(&mut app, 1000);
    }
    assert!(app.world().get_entity(root).is_ok());
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .candidate
        .is_none());
    assert_eq!(app.world().resource::<IndoorSequence>().index, 1);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .indoor_camera = None;
    app.world_mut().write_message(RegenerateSceneEvent);
    tick(&mut app, 1);
    assert!(app.world().resource::<PendingIndoor>().requested);
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .blocked
        .is_none());
}

#[test]
fn a_failed_worker_with_valid_settings_advances_after_a_bounded_delay() {
    for initial in [false, true] {
        let (mut app, root) = setup();
        inject(
            app.world_mut(),
            71,
            Err("planning worker unavailable".into()),
        );
        if initial {
            app.world_mut().despawn(root);
            app.world_mut().write_message(RegenerateSceneEvent);
        }
        tick(&mut app, 20);
        assert_eq!(
            app.world()
                .resource::<IndoorGenerationStatus>()
                .skipped_seed,
            Some(71)
        );
        tick(&mut app, 999);
        assert!(app
            .world()
            .resource::<PendingIndoor>()
            .studio
            .candidate
            .is_none());
        tick(&mut app, 1);
        let pending = app.world().resource::<PendingIndoor>();
        assert_eq!(
            pending.studio.candidate.as_ref().map(|c| c.0.seed),
            Some(72)
        );
        assert_eq!(app.world().get_entity(root).is_ok(), !initial);
    }
}

#[test]
fn rejected_initial_room_recovers_without_waiting_for_a_display_interval() {
    let (mut app, root) = setup();
    app.world_mut().despawn(root);
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .regenerate_ms = 60_000;
    let request = key(app.world(), 71);
    inject(
        app.world_mut(),
        71,
        Err(CaptureError::rejected(
            71,
            SceneRejectionKind::CameraPlacement,
            "initial seed".into(),
        )),
    );
    app.world_mut().resource_mut::<PendingIndoor>().key = Some(request);
    app.world_mut().write_message(RegenerateSceneEvent);
    tick(&mut app, 20);
    assert!(
        app.world()
            .resource::<PendingIndoor>()
            .studio
            .recover_empty_viewport
    );
    assert_eq!(app.world().resource::<IndoorSequence>().index, 2);
    let world = app.world();
    let room = bevy::tasks::block_on(prepare_indoor_typed(
        72,
        world.resource(),
        world.resource(),
        *world.resource::<gi::IndoorGiSettings>(),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
    ))
    .unwrap();
    inject(app.world_mut(), 72, Ok(room));
    // Some WebGPU adapters delay queue completion callbacks for seconds.
    // A late or missing pacing signal still permits a completed room to appear.
    app.init_resource::<preparation::studio_upload::UploadFence>();
    for _ in 0..2000 {
        tick(&mut app, 16);
        if app.world().contains_resource::<layout::IndoorManifest>() {
            break;
        }
    }
    assert_eq!(app.world().resource::<layout::IndoorManifest>().seed, 72);
    assert!(app.world().resource::<Time<Real>>().elapsed() < Duration::from_secs(60));
}

#[test]
fn startup_timeout_includes_missing_human_assets_but_never_skips_a_capture_request() {
    for headless in [false, true] {
        let (mut app, root) = setup();
        app.world_mut().despawn(root);
        let mut config = app.world_mut().resource_mut::<BevyZeroverseConfig>();
        config.headless = headless;
        config.indoor_human_density = 0.5;
        app.world_mut().write_message(RegenerateSceneEvent);
        tick(&mut app, 1);
        tick(&mut app, 60_000);
        let pending = app.world().resource::<PendingIndoor>();
        assert_eq!(pending.requested, headless);
        assert_eq!(pending.studio.recover_empty_viewport, !headless);
        assert_eq!(
            app.world()
                .resource::<IndoorGenerationStatus>()
                .skipped_seed,
            (!headless).then_some(71)
        );
        assert_eq!(
            app.world().resource::<IndoorSequence>().index,
            if headless { 1 } else { 2 }
        );
    }
}

#[test]
fn a_ready_room_can_wait_for_a_five_minute_display_interval() {
    let (mut app, root) = setup();
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .regenerate_ms = 300_000;
    let world = app.world();
    let room = bevy::tasks::block_on(prepare_indoor_typed(
        71,
        world.resource(),
        world.resource(),
        *world.resource::<gi::IndoorGiSettings>(),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
    ))
    .unwrap();
    inject(app.world_mut(), 71, Ok(room));
    for _ in 0..500 {
        tick(&mut app, 16);
        if app
            .world()
            .resource::<PendingIndoor>()
            .studio
            .work
            .as_ref()
            .is_some_and(|w| w.phase == Phase::Ready)
        {
            break;
        }
    }
    assert_eq!(
        app.world()
            .resource::<PendingIndoor>()
            .studio
            .work
            .as_ref()
            .unwrap()
            .phase,
        Phase::Ready
    );
    tick(&mut app, 250_000);
    assert!(app.world().get_entity(root).is_ok());
    assert_eq!(
        app.world()
            .resource::<IndoorGenerationStatus>()
            .skipped_seed,
        None
    );
    tick(&mut app, 50_000);
    assert_eq!(app.world().resource::<layout::IndoorManifest>().seed, 71);
}

#[test]
fn late_upload_releases_staged_handles_and_cpu_fallback_shares_the_seed_deadline() {
    let (mut app, root) = setup();
    app.world_mut()
        .resource_mut::<BevyZeroverseConfig>()
        .indoor_gpu_materials = true;
    app.world_mut()
        .resource_mut::<materials::gpu::MaterialComputeAvailable>()
        .0 = true;
    let world = app.world();
    let room = bevy::tasks::block_on(prepare_indoor_typed(
        71,
        world.resource(),
        world.resource(),
        *world.resource::<gi::IndoorGiSettings>(),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
        StagedAssets::new(world.resource()),
    ))
    .unwrap();
    // These are the real material recipes, with no render device to complete them.
    assert!(!room.material_set.gpu_bakes.ready());
    inject(app.world_mut(), 71, Ok(room));
    tick(&mut app, 1);
    for _ in 0..300 {
        tick(&mut app, 10);
        if app
            .world()
            .resource::<PendingIndoor>()
            .studio
            .waiting_for_bakes
            .is_some()
        {
            break;
        }
    }
    assert!(app
        .world()
        .resource::<PendingIndoor>()
        .studio
        .waiting_for_bakes
        .is_some());
    assert!(!app
        .world()
        .resource::<preparation::residency::FutureAssets>()
        .material_bakes
        .ready());
    // The failed GPU attempt uses 59 of the 60 seconds. CPU fallback has only
    // the unused second, and may not display a room containing unbaked maps.
    let elapsed = app.world().resource::<Time<Real>>().elapsed().as_millis() as u64;
    tick(&mut app, 59_001 - elapsed);
    {
        let studio = &app.world().resource::<PendingIndoor>().studio;
        assert!(studio.cpu_materials && studio.candidate.is_none() && studio.uploads.is_none());
        assert!(app
            .world()
            .resource::<preparation::residency::FutureAssets>()
            .material_bakes
            .reflections()
            .is_empty());
    }
    tick(&mut app, 1000);
    assert_eq!(
        app.world()
            .resource::<IndoorGenerationStatus>()
            .skipped_seed,
        Some(71)
    );
    assert!(app.world().get_entity(root).is_ok());
}

#[test]
fn repeated_stalls_cannot_accumulate_requests_or_revisit_expired_seeds() {
    let (mut app, root) = setup();
    for seed in 71..=136 {
        let request = key(app.world(), seed);
        app.world_mut()
            .resource_mut::<PendingIndoor>()
            .studio
            .candidate = Some((request, PreparationJob::pending_for_test()));
        tick(&mut app, 1);
        tick(&mut app, 60_001);
        let pending = app.world().resource::<PendingIndoor>();
        assert!(
            pending.studio.candidate.is_none()
                && pending.studio.uploads.is_none()
                && pending.task.is_none()
        );
        assert!(app.world().get_entity(root).is_ok());
        assert_eq!(
            app.world()
                .resource::<IndoorGenerationStatus>()
                .skipped_seed,
            Some(seed)
        );
    }
    assert!(app.world().resource::<Time<Real>>().elapsed() > Duration::from_secs(3600));
}
