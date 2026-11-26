use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn same_seed_rejects_late_completion_and_selected_views_have_one_persistent_receipt() {
    let mut app = App::new();
    let manifest =
        IndoorManifest::generate_with_humans(163, IndoorLayout::Conference, 0.65, 3, 0.).unwrap();
    let seed = manifest.seed;
    app.add_message::<RegenerateSceneEvent>()
        .add_message::<SceneReadyForCapture>()
        .insert_resource(manifest)
        .insert_resource(RenderMode::Color)
        .insert_resource(CaptureReadiness {
            scene_seed: Some(seed),
            blocker: Some(CaptureBlocker::Lighting),
            blocked_updates: 0,
        })
        .init_resource::<SceneCaptureState>()
        .insert_resource(SceneCaptureViews::only([2, 0]).unwrap())
        .add_systems(PostUpdate, update);
    assert!(SceneCaptureViews::only([]).is_err());
    assert!(SceneCaptureViews::only([1, 1]).is_err());
    fn room(w: &mut World) -> Entity {
        w.init_resource::<Assets<Image>>();
        let region = crate::ovoxel::OvoxelRegion::primary_room(w.resource::<IndoorManifest>());
        let root = w
            .spawn((ZeroverseSceneRoot, crate::scene::SceneAabbNode, region))
            .id();
        for i in 0..3 {
            let image = w.resource_mut::<Assets<Image>>().add(Image::default());
            w.spawn((
                ChildOf(root),
                CaptureCameraIndex(i),
                ZeroverseCamera::default(),
                RenderTarget::from(image),
            ));
        }
        root
    }
    let first = room(app.world_mut());
    app.update();
    assert_eq!(
        app.world().resource::<SceneCaptureState>().stage(),
        SceneCaptureStage::Preparing(CaptureBlocker::Lighting)
    );
    app.world_mut().resource_mut::<CaptureReadiness>().blocker = None;
    app.update();
    let obsolete = app
        .world()
        .resource::<SceneCaptureState>()
        .intent
        .clone()
        .unwrap();
    app.world_mut().write_message(RegenerateSceneEvent);
    obsolete.completion.store(COMPLETE, Ordering::Release);
    app.update();
    assert!(app
        .world()
        .resource::<SceneCaptureState>()
        .ready()
        .is_none());
    app.update();
    assert!(app.world().resource::<SceneCaptureState>().intent.is_none());
    app.world_mut().resource_mut::<CaptureReadiness>().blocker = Some(CaptureBlocker::Failed);
    app.update();
    assert_eq!(
        app.world().resource::<SceneCaptureState>().stage(),
        SceneCaptureStage::Preparing(CaptureBlocker::Failed)
    );
    app.world_mut().resource_mut::<CaptureReadiness>().blocker = None;
    app.world_mut().despawn(first);
    let next = room(app.world_mut());
    app.update();
    let state = app.world().resource::<SceneCaptureState>();
    assert!(state.ready().is_none());
    let current = state.intent.clone().unwrap();
    assert_eq!(current.receipt.root, next);
    assert_ne!(current.receipt.generation, obsolete.receipt.generation);
    assert_eq!(
        current
            .receipt
            .views
            .iter()
            .map(|c| c.index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    current.completion.store(COMPLETE, Ordering::Release);
    let mut cursor = app
        .world()
        .resource::<Messages<SceneReadyForCapture>>()
        .get_cursor_current();
    app.update();
    let events = app.world().resource::<Messages<SceneReadyForCapture>>();
    assert_eq!(cursor.read(events).count(), 1);
    app.insert_resource(SceneCaptureViews::only([0, 2]).unwrap());
    for _ in 0..4 {
        app.update();
    }
    assert_eq!(
        cursor
            .read(app.world().resource::<Messages<SceneReadyForCapture>>())
            .count(),
        0
    );
    assert_eq!(
        app.world()
            .resource::<SceneCaptureState>()
            .ready()
            .unwrap()
            .root,
        next
    );
    app.insert_resource(SceneCaptureViews::only([3]).unwrap());
    app.update();
    assert_eq!(
        app.world().resource::<SceneCaptureState>().stage(),
        SceneCaptureStage::CameraOutOfRange(3)
    );
    app.insert_resource(SceneCaptureViews::only([1]).unwrap());
    app.update();
    assert_eq!(
        app.world().resource::<SceneCaptureState>().stage(),
        SceneCaptureStage::WaitingForDraw
    );
    app.insert_resource(RenderMode::Depth);
    app.update();
    assert_eq!(
        app.world().resource::<SceneCaptureState>().stage(),
        SceneCaptureStage::WaitingForColor
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "runs a cold native renderer and a same-seed regeneration on the selected Vulkan adapter"]
fn cold_room_and_same_seed_render_every_selected_optical_texture_without_copiers() {
    use crate::{
        app::{viewer_app, BevyZeroverseConfig},
        sample::SamplerState,
        scene::{procedural_indoor::reset_indoor_sequence, ZeroverseSceneType},
    };
    let lenses = (1..20)
        .map(|step| crate::camera::lens::LensSampling {
            probability: step as f32 / 20.,
            circular_probability: 1.,
            ..default()
        })
        .find(|l| (0..4).filter(|&i| l.sample(76104, i).is_some()).count() == 2)
        .unwrap();
    let mut app = viewer_app(
        None,
        Some(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            indoor_layout: IndoorLayout::Cafe,
            indoor_seed: Some(76104),
            indoor_camera: Some(serde_json::json!({"lenses":lenses}).to_string()),
            num_cameras: 4,
            initialize_scene: false,
            headless: true,
            editor: false,
            width: 96.,
            height: 96.,
            indoor_gi_rays: 64,
            ssaa_factor: 2,
            image_copiers: false,
            gizmos: false,
            keybinds: false,
            indoor_human_density: 0.,
            regenerate_ms: 0,
            ..default()
        }),
    );
    app.add_plugins(SceneCaptureReadyPlugin);
    app.finish();
    use bevy::render::{Render, RenderSystems};
    app.sub_app_mut(RenderApp).add_systems(
        Render,
        super::render::trace_test_frame.in_set(RenderSystems::Cleanup),
    );
    app.cleanup();
    app.update();
    assert!(app.world().get_resource::<SamplerState>().is_none());
    let mut previous = None;
    for _ in 0..2 {
        reset_indoor_sequence(app.world_mut(), 76104);
        app.world_mut().write_message(RegenerateSceneEvent);
        app.update();
        assert!(app
            .world()
            .resource::<SceneCaptureState>()
            .ready()
            .is_none());
        let start = std::time::Instant::now();
        let mut updates = 0;
        loop {
            app.update();
            updates += 1;
            let state = app.world().resource::<SceneCaptureState>();
            if let Some(ready) = state.ready() {
                assert_eq!(ready.seed, 76104);
                assert_eq!(ready.views.len(), 4);
                let mut images = std::collections::HashSet::new();
                for (i, view) in ready.views.iter().enumerate() {
                    assert_eq!(view.index, i);
                    assert!(images.insert(view.image.id()));
                    assert_eq!(
                        app.world().get::<SsaaCamera>(view.camera).unwrap().output,
                        view.image
                    );
                    assert!(app
                        .world()
                        .get::<crate::io::image_copy::ImageCopier>(view.camera)
                        .is_none());
                }
                let current = (ready.root, ready.generation);
                if let Some((root, ticket)) = previous {
                    assert_ne!(root, current.0);
                    assert_ne!(ticket, current.1);
                }
                let fisheyes = ready
                    .views
                    .iter()
                    .filter(|v| app.world().get::<LensCamera>(v.camera).is_some())
                    .count();
                assert!((1..4).contains(&fisheyes), "expected a mixed optical rig");
                println!(
                    "ready seed={} root={:?} generation={} cameras={} fisheyes={} elapsed={:?}",
                    ready.seed,
                    ready.root,
                    ready.generation,
                    ready.views.len(),
                    fisheyes,
                    start.elapsed()
                );
                previous = Some(current);
                break;
            }
            assert!(
                app.world()
                    .resource::<crate::sample::CaptureFailure>()
                    .0
                    .is_none(),
                "{:?}",
                app.world().resource::<crate::sample::CaptureFailure>().0
            );
            let gpu = app
                .world()
                .resource::<crate::io::image_copy::CapturePipelineReadiness>();
            assert!(start.elapsed().as_secs() < 180,
                "capture stalled in {updates} updates at {:?}, pipelines={} ready={} missing={} failure={:?}",
                state.stage(), gpu.pipeline_count(), gpu.ready(), gpu.missing_assets(), gpu.failure());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}
