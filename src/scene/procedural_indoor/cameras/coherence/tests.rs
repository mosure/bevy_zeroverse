use super::*;
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::Coverage,
        multiview::{MultiViewSettings, OverlapGraph},
    },
    layout::IndoorLayout,
    validation::validate_layout,
};

#[test]
fn baseline_targets_are_continuous_and_explicit_policies_remain_exact() {
    let mut settings = CameraSettings::default();
    let mut previous = (f32::INFINITY, f32::INFINITY);
    for b in 0..=100 {
        settings.multiview = Some(MultiViewSettings::from_baseline(b as f32 / 100.).unwrap());
        let c = CameraCoherencePolicy::new(&settings, 8).unwrap();
        assert!(
            c.target_min_directed_overlap <= previous.0 && c.target_min_containment <= previous.1
        );
        assert!(
            c.target_min_directed_overlap > 0.
                && c.target_min_containment > c.target_min_directed_overlap
        );
        previous = (c.target_min_directed_overlap, c.target_min_containment);
        assert_eq!(c.max_candidates, 96);
    }
    for count in [0, 1, 2] {
        assert!(CameraCoherencePolicy::new(&settings, count).is_none());
    }
    settings.multiview.as_mut().unwrap().graph = OverlapGraph::Connected;
    assert!(CameraCoherencePolicy::new(&settings, 8).is_none());
    settings.multiview = Some(MultiViewSettings::from_baseline(0.5).unwrap());
    settings.multiview.as_mut().unwrap().min_overlap += 0.01;
    assert!(CameraCoherencePolicy::new(&settings, 8).is_none());
    settings.multiview = Some(MultiViewSettings::from_baseline(0.5).unwrap());
    settings.multiview.as_mut().unwrap().mixed_rig = false;
    assert!(CameraCoherencePolicy::new(&settings, 8).is_none());
    settings.multiview = Some(MultiViewSettings::from_baseline(0.5).unwrap());
    settings.overlap_mixture = Some(Default::default());
    assert!(CameraCoherencePolicy::new(&settings, 8).is_none());
    settings.overlap_mixture = None;
    settings.handheld = Some(Default::default());
    assert!(CameraCoherencePolicy::new(&settings, 8).is_none());
}

#[test]
fn static_pair_cache_matches_independent_rays_and_never_freezes_moving_peers() {
    let mut scene =
        IndoorManifest::generate_with_humans(4, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let settings = CameraSettings {
        motion_fraction: 0.,
        multiview: None,
        ..Default::default()
    };
    scene.resample_cameras(3, settings, 1.).unwrap();
    for moving in [false, true] {
        if moving {
            scene.cameras[1].end += bevy::prelude::Vec3::new(0.3, 0.1, 0.2);
        }
        let coverage = Coverage::new(&scene);
        for pair in scene.camera_pair_overlaps() {
            for sample in pair.samples {
                let a = coverage.view(&scene.cameras[pair.reference], 1., sample.time);
                let b = coverage.view(&scene.cameras[pair.camera], 1., sample.time);
                let (ab, aa) = coverage.shared(&a, &b);
                let (ba, ba_angle) = coverage.shared(&b, &a);
                assert_eq!(sample.reference_to_view, ab);
                assert_eq!(sample.view_to_reference, ba);
                assert_eq!(sample.mean_triangulation_degrees, (aa + ba_angle) * 0.5);
            }
        }
    }
}

#[test]
fn refinement_preserves_paths_calibration_reference_and_existing_support() {
    let mut improved = 0;
    for seed in 65_832_553..65_832_559 {
        let settings = CameraSettings::parse(include_str!(
            "../ceiling/tests/mixed_training_camera_settings.json"
        ))
        .unwrap();
        let mut before =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        before.camera_settings = settings.clone();
        before.camera_aspect_ratio = 1.;
        before.sample_camera_placement(8).unwrap();
        let mut scene = before.clone();
        refine(&mut scene);
        validate_layout(&scene).unwrap();
        let diagnostics = &scene.camera_diagnostics;
        assert!(
            diagnostics.peer_framing_candidates
                <= scene.camera_coherence_policy().unwrap().max_candidates
        );
        assert!(diagnostics.peer_weak_pairs_after <= diagnostics.peer_weak_pairs_before);
        improved +=
            usize::from(diagnostics.peer_weak_pairs_after < diagnostics.peer_weak_pairs_before);
        assert_eq!(scene.cameras[0], before.cameras[0]);
        assert_eq!(scene.objects, before.objects);
        assert_eq!(scene.envelope, before.envelope);
        for (i, (a, b)) in before.cameras.iter().zip(&scene.cameras).enumerate() {
            assert_eq!(a.fisheye, b.fisheye);
            assert_eq!(a.fov_degrees, b.fov_degrees);
            assert_eq!(a.ceiling_mount, b.ceiling_mount);
            assert_eq!(a.principal_point_offset, b.principal_point_offset);
            assert_eq!(a.path_length(), b.path_length());
            assert!(scene.camera_curve_clear(b));
            assert!(settings.accepts_motion(seed, i, b));
            for t in 0..=32 {
                let (a, b) = (
                    a.transform_at(t as f32 / 32.),
                    b.transform_at(t as f32 / 32.),
                );
                assert_eq!(a.translation, b.translation);
                assert!(a.rotation.angle_between(b.rotation) <= 60_f32.to_radians() + 1e-5);
            }
        }
        let policy = scene.camera_coherence_policy().unwrap();
        let old = before.camera_pair_overlaps();
        let new = scene.camera_pair_overlaps();
        assert_eq!(new.len(), 28);
        for (a, b) in old.iter().zip(&new).filter(|(a, _)| a.reference != 0) {
            let floor = policy.pair_overlap_target(
                &scene.cameras[a.reference],
                &scene.cameras[a.camera],
                1.,
            );
            let supported = |p: &super::super::multiview::CameraPairOverlap| {
                p.samples.iter().all(|s| {
                    s.reference_to_view.min(s.view_to_reference) >= floor
                        && s.reference_to_view.max(s.view_to_reference)
                            >= policy.target_min_containment
                })
            };
            if supported(a) {
                assert!(supported(b));
            }
        }
        let mut replay = before.clone();
        refine(&mut replay);
        assert_eq!(replay, scene);
    }
    assert!(
        improved >= 2,
        "refinement did not improve weak peer support in consecutive rooms: {improved}"
    );
}

#[test]
fn preferred_peer_search_is_bounded_and_never_rejects_previously_feasible_rooms() {
    for seed in [65832633, 65832620, 65573956, 65565210] {
        let settings = CameraSettings::parse(include_str!(
            "../ceiling/tests/mixed_training_camera_settings.json"
        ))
        .unwrap();
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.resample_cameras(8, settings.clone(), 1.).unwrap();
        validate_layout(&scene).unwrap();
        let d = &scene.camera_diagnostics;
        assert!(d.peer_search_succeeded || d.peer_search_fallbacks == 1);
        assert!(d.peer_search_discarded_proposals <= 1536);
        let policy = scene.camera_coherence_policy().unwrap();
        if d.peer_search_succeeded {
            for pair in scene
                .camera_pair_overlaps()
                .into_iter()
                .filter(|p| p.reference != 0)
            {
                let floor = policy.preferred_floor(
                    &scene.cameras[pair.reference],
                    &scene.cameras[pair.camera],
                    1.,
                );
                assert!(pair
                    .samples
                    .iter()
                    .all(|s| s.reference_to_view.min(s.view_to_reference) >= floor
                        && s.reference_to_view.max(s.view_to_reference)
                            >= policy.preferred_min_containment));
            }
        }
        for (i, c) in scene.cameras.iter().enumerate() {
            assert_eq!(c.is_ceiling_mounted(), settings.ceiling_mounted(seed, i));
            assert_eq!(c.fisheye, settings.lens_for(seed, i));
            assert!(settings.accepts_motion(seed, i, c));
        }
        if seed == 65832633 {
            assert!(
                d.peer_search_succeeded,
                "spatially disjoint holdout rig must be repaired"
            );
        }
    }
}
