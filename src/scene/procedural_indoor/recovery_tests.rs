//! Exercise the real CPU retirement transaction without a renderer or GPU.
use super::*;
use crate::sample::{
    failure::{take_capture_failure, CaptureError, SceneCaptureError, SceneRejectionKind},
    CaptureFailure, CaptureProgress, Sample, SamplerState, View,
};

#[test]
fn ceiling_rejections_retire_scene_state_without_resetting_request_counters() {
    let mut world = World::new();
    world.insert_resource(BevyZeroverseConfig {
        headless: true,
        editor: false,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    });
    world.init_resource::<CaptureProgress>();
    world.init_resource::<SamplerState>();
    world.init_resource::<PendingIndoor>();
    world.init_resource::<preparation::residency::FutureAssets>();
    #[cfg(not(target_arch = "wasm32"))]
    world.init_resource::<preparation::IndoorPrefetch>();
    world.init_resource::<Messages<RegenerateSceneEvent>>();
    world.resource_mut::<CaptureProgress>().completed_requests = 17;

    for seed in [2, 65_904_936, 65_991_960] {
        let mut scene = layout::IndoorManifest::generate_with_humans(
            seed,
            layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        validation::validate_prepared_layout(&scene).unwrap();
        let ceiling = scene.ceiling_installation.as_mut().unwrap();
        ceiling.modules.push(ceiling.modules[0].clone());
        let error = validation::validate_prepared_layout(&scene).unwrap_err();
        world.insert_resource(IndoorSequence {
            base_seed: Some(seed),
            ..default()
        });
        world.insert_resource(scene);
        let root = world.spawn(ZeroverseScene).id();
        let child = world.spawn(ChildOf(root)).id();
        world.insert_resource(Sample {
            views: vec![View::default()],
            ..default()
        });
        world.resource_mut::<SamplerState>().enabled = true;
        world.insert_resource(CaptureFailure(Some(error.to_string())));
        world.insert_resource(SceneCaptureError(Some(error)));

        assert!(matches!(take_capture_failure(&mut world),
            Some(CaptureError::Rejected { rejection, retired: true })
            if rejection.seed == seed && rejection.kind == SceneRejectionKind::PlacementClearance));
        assert!(world.get_entity(root).is_err() && world.get_entity(child).is_err());
        assert!(!world.contains_resource::<layout::IndoorManifest>());
        assert!(world.resource::<Sample>().views.is_empty());
        assert!(!world.resource::<SamplerState>().enabled);
        assert_eq!(world.resource::<IndoorSequence>().index, 1);
        assert_eq!(world.resource::<CaptureProgress>().completed_requests, 17);
        assert!(take_capture_failure(&mut world).is_none());
    }
    assert_eq!(world.resource::<Messages<RegenerateSceneEvent>>().len(), 3);
}

#[test]
fn retirement_failure_is_fatal_and_leaves_unrelated_transaction_intact() {
    let mut world = World::new();
    world.insert_resource(BevyZeroverseConfig {
        headless: true,
        editor: false,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    });
    world.insert_resource(IndoorSequence {
        base_seed: Some(40),
        ..default()
    });
    world.init_resource::<CaptureProgress>();
    let root = world.spawn(ZeroverseScene).id();
    world.insert_resource(Sample {
        views: vec![View::default()],
        ..default()
    });
    let error = CaptureError::rejected(
        41,
        SceneRejectionKind::PlacementClearance,
        "generated scene invalid".into(),
    );
    world.insert_resource(CaptureFailure(Some(error.to_string())));
    world.insert_resource(SceneCaptureError(Some(error)));
    assert!(
        matches!(take_capture_failure(&mut world), Some(CaptureError::Failed(reason))
        if reason.contains("cannot retire"))
    );
    assert!(world.get_entity(root).is_ok());
    assert_eq!(world.resource::<Sample>().views.len(), 1);
    assert_eq!(world.resource::<IndoorSequence>().index, 0);
    assert!(take_capture_failure(&mut world).is_none());
}
