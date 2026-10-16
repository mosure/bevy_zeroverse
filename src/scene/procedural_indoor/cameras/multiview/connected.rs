//! Bounded connected-view sampling. Geometry, paths and the camera-zero metric
//! spacing contract remain unchanged; only the required visibility edges differ.
use super::super::program::{pool_size, retarget, ViewSummary};
use super::*;

pub(super) fn all_pairs(scene: &IndoorManifest) -> Vec<CameraPairOverlap> {
    let coverage = Coverage::new(scene);
    let views: Vec<Vec<_>> = scene
        .cameras
        .iter()
        .map(|c| {
            CHECK_TIMES
                .into_iter()
                .map(|t| coverage.view(c, scene.camera_aspect_ratio, t))
                .collect()
        })
        .collect();
    let mut result = Vec::new();
    for i in 0..scene.cameras.len() {
        for j in i + 1..scene.cameras.len() {
            result.push(CameraPairOverlap {
                reference: i,
                camera: j,
                samples: pair_samples_cached(
                    &coverage,
                    &scene.cameras[i],
                    &views[i],
                    &scene.cameras[j],
                    &views[j],
                ),
            });
        }
    }
    result
}

pub(super) fn connected(
    count: usize,
    pairs: &[CameraPairOverlap],
    policy: &MultiViewSettings,
) -> bool {
    if count < 2 {
        return true;
    }
    let mut edge = policy.clone();
    // Camera-zero distances are validated by CameraGroupGeometry. A graph edge
    // need only satisfy the all-pair minimum, not the separate reference floor.
    edge.min_reference_baseline = 0.;
    let mut seen = vec![false; count];
    seen[0] = true;
    loop {
        let mut changed = false;
        for pair in pairs {
            if pair.reference >= count || pair.camera >= count || pair.reference == pair.camera {
                return false;
            }
            if pair.samples.len() != CHECK_TIMES.len() || !edge.accepts(&pair.samples) {
                continue;
            }
            if seen[pair.reference] != seen[pair.camera] {
                seen[pair.reference] = true;
                seen[pair.camera] = true;
                changed = true;
            }
        }
        if !changed {
            return seen.into_iter().all(|x| x);
        }
    }
}

pub(super) fn sample(
    scene: &mut IndoorManifest,
    count: usize,
    policy: &MultiViewSettings,
    rng: &mut rand_chacha::ChaCha8Rng,
    coverage: &Coverage,
) -> Result<(), String> {
    let (lo, hi) = if scene.camera_settings.primary_room {
        scene.primary_room_bounds()
    } else {
        let h = scene.room_size.xz() * 0.5;
        (-h, h)
    };
    let mut edge = policy.clone();
    edge.min_reference_baseline = 0.;
    let mut most = 0;
    let mut partial_groups = Vec::new();
    let viewpoint = super::super::viewpoint::ViewpointProgram::sample(scene.seed);
    for _ in 0..GROUP_ATTEMPTS {
        scene.cameras.clear();
        if scene.sample_independent_cameras(1, rng, coverage).is_err() {
            continue;
        }
        let mut views: Vec<Vec<_>> = vec![CHECK_TIMES
            .into_iter()
            .map(|t| coverage.view(&scene.cameras[0], scene.camera_aspect_ratio, t))
            .collect()];
        let mut tracks = vec![Track::new(&scene.cameras[0])];
        let mut summaries = vec![ViewSummary::new(&scene.cameras[0], &views[0][0], scene)];
        while scene.cameras.len() < count {
            let mut best = None;
            let mut first = None;
            let mut accepted = 0;
            for attempt in 0..VIEW_ATTEMPTS {
                if first.is_some_and(|first| attempt >= first + 12)
                    || accepted >= pool_size(policy.target_variation, count)
                {
                    break;
                }
                scene.camera_diagnostics.peer_proposals += 1;
                let parent = rng.random_range(0..scene.cameras.len());
                let reference = &scene.cameras[parent];
                let reachable = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)]
                    .into_iter()
                    .map(|p| p.distance(reference.start.xz()))
                    .fold(0.0_f32, f32::max)
                    .hypot(0.9)
                    .min(policy.max_baseline);
                let proposal_policy = if parent == 0 { policy } else { &edge };
                if reachable
                    < proposal_policy
                        .min_baseline
                        .max(proposal_policy.min_reference_baseline)
                {
                    continue;
                }
                let floor = scene.floor_height(reference.start.xz());
                let fixture_margin = scene.program.as_ref().map_or(0.5, |p| p.light_drop)
                    + 0.04
                    + super::super::super::layout::CAMERA_CLEARANCE;
                let ceiling =
                    (scene.ceiling_height(reference.start.xz()) - fixture_margin).min(floor + 3.25);
                let mut candidate = propose(
                    reference,
                    reference.path_length().max(0.001),
                    reachable,
                    Vec2::new(
                        floor + 0.70 - reference.start.y,
                        ceiling - reference.start.y,
                    ),
                    proposal_policy,
                    &scene.camera_settings.intrinsics,
                    rng,
                );
                // Proposals inherit a valid path but can move below a raised
                // floor or above a sloped roof. Fit the origin to its actual
                // local height interval; the complete swept path is checked.
                let floor = scene.floor_height(candidate.start.xz());
                let low = floor + 0.70;
                let high =
                    (scene.ceiling_height(candidate.start.xz()) - fixture_margin).min(floor + 3.25);
                if low > high {
                    scene.camera_diagnostics.peer_rejections[0] += 1;
                    continue;
                }
                let offset = Vec3::Y * (candidate.start.y.clamp(low, high) - candidate.start.y);
                candidate.start += offset;
                candidate.end += offset;
                if let Some(motion) = &mut candidate.motion {
                    for point in motion.control.iter_mut().chain(&mut motion.route) {
                        *point += offset;
                    }
                }
                retarget(
                    scene,
                    &mut candidate,
                    &views[parent],
                    policy.target_variation,
                    count,
                    rng,
                );
                if policy.target_variation > 0. {
                    viewpoint.level_gaze(&mut candidate, rng);
                }
                scene.camera_settings.prepare_motion(
                    scene.seed,
                    scene.cameras.len(),
                    &mut candidate,
                    rng,
                );
                let track = Track::new(&candidate);
                let reason = if !scene.camera_clear(candidate.start) {
                    Some(0)
                } else if !geometry(&tracks, Some(&track)).unwrap().accepts(policy) {
                    Some(1)
                } else if !scene.camera_settings.accepts_motion(
                    scene.seed,
                    scene.cameras.len(),
                    &candidate,
                ) {
                    Some(2)
                } else if !scene.camera_curve_clear(&candidate) {
                    Some(3)
                } else if !coverage.suitable(&candidate, false) {
                    Some(4)
                } else {
                    None
                };
                if let Some(reason) = reason {
                    scene.camera_diagnostics.peer_rejections[reason] += 1;
                    continue;
                }
                let candidate_views: Vec<_> = CHECK_TIMES
                    .into_iter()
                    .map(|t| coverage.view(&candidate, scene.camera_aspect_ratio, t))
                    .collect();
                let samples = pair_samples_cached(
                    coverage,
                    reference,
                    &views[parent],
                    &candidate,
                    &candidate_views,
                );
                if !edge.accepts(&samples) {
                    scene.camera_diagnostics.peer_rejections[5] += 1;
                    continue;
                }
                let summary = ViewSummary::new(&candidate, &candidate_views[0], scene);
                let score = summary.score(&summaries, count, scene);
                accepted += 1;
                first.get_or_insert(attempt);
                if best.as_ref().is_none_or(|(_, _, _, _, s)| score > *s) {
                    best = Some((candidate, track, candidate_views, summary, score));
                }
            }
            let Some((candidate, track, v, summary, _)) = best else {
                break;
            };
            scene.cameras.push(candidate);
            tracks.push(track);
            views.push(v);
            summaries.push(summary);
        }
        if scene.cameras.len() == count {
            return Ok(());
        }
        most = most.max(scene.cameras.len());
        recovery::retain_diverse(&mut partial_groups, &scene.cameras);
        scene.camera_diagnostics.group_restarts += 1;
    }
    // Reuse the bounded room-space/beam recovery with graph-aware edges. In a
    // compact stepped room, radial offsets alone can miss the remaining pockets.
    let mut rejected = [0; 6];
    let recovered = recovery::complete(
        scene,
        &partial_groups,
        count,
        policy,
        rng,
        coverage,
        &mut rejected,
    );
    for (total, count) in scene
        .camera_diagnostics
        .peer_rejections
        .iter_mut()
        .zip(rejected)
    {
        *total += count;
    }
    if recovered == count {
        return Ok(());
    }
    most = most.max(recovered);
    scene.cameras.clear();
    Err(format!("seed {}: connected camera graph exhausted bounded search for {count} cameras (most_placed={most}); constraints were retained",scene.seed))
}

pub(super) fn candidate_links(
    scene: &IndoorManifest,
    coverage: &Coverage,
    camera: &IndoorCamera,
    policy: &MultiViewSettings,
) -> bool {
    let mut edge = policy.clone();
    edge.min_reference_baseline = 0.;
    let views: Vec<_> = CHECK_TIMES
        .into_iter()
        .map(|t| coverage.view(camera, scene.camera_aspect_ratio, t))
        .collect();
    scene.cameras.iter().any(|peer| {
        let peer_views: Vec<_> = CHECK_TIMES
            .into_iter()
            .map(|t| coverage.view(peer, scene.camera_aspect_ratio, t))
            .collect();
        edge.accepts(&pair_samples_cached(
            coverage,
            peer,
            &peer_views,
            camera,
            &views,
        ))
    })
}

impl IndoorManifest {
    pub fn accepts_camera_visibility(&self) -> bool {
        let pairs = self.camera_overlap();
        match &self.camera_settings.multiview {
            Some(policy) if policy.graph == OverlapGraph::Connected => {
                connected(self.cameras.len(), &pairs, policy)
            }
            _ => pairs.iter().all(|pair| self.accepts_camera_pair(pair)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visibility_chain_accepts_edges_without_a_star() {
        let policy = MultiViewSettings {
            graph: OverlapGraph::Connected,
            ..Default::default()
        };
        let pair = |a, b, overlap| CameraPairOverlap {
            reference: a,
            camera: b,
            samples: CHECK_TIMES
                .into_iter()
                .map(|time| OverlapSample {
                    time,
                    reference_to_view: overlap,
                    view_to_reference: overlap,
                    baseline_m: 0.5,
                    mean_triangulation_degrees: 10.,
                })
                .collect(),
        };
        assert!(connected(
            3,
            &[pair(0, 1, 0.7), pair(1, 2, 0.7), pair(0, 2, 0.)],
            &policy
        ));
        assert!(!connected(3, &[pair(0, 1, 0.7), pair(1, 2, 0.)], &policy));
    }
    #[test]
    fn sampled_connected_graphs_preserve_scene_and_paths() {
        for seed in 0..12 {
            let mut scene = IndoorManifest::generate(
                seed,
                crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
                0.65,
                0,
            )
            .unwrap();
            let geometry = scene.objects.clone();
            let mut settings = crate::scene::procedural_indoor::cameras::CameraSettings::default();
            settings.multiview.as_mut().unwrap().graph = OverlapGraph::Connected;
            scene.resample_cameras(4, settings, 1.6).unwrap();
            assert_eq!(scene.objects, geometry);
            assert!(scene.accepts_camera_visibility());
            crate::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        }
    }
    #[test]
    fn connected_wide_rigs_fit_stepped_room_with_shifted_lenses() {
        use super::super::super::CameraSettings;
        for aspect in [0.625, 1., 1.6] {
            let mut scene = IndoorManifest::generate(
                11,
                crate::scene::procedural_indoor::layout::IndoorLayout::Mixed,
                0.65,
                0,
            )
            .unwrap();
            let mut settings = CameraSettings {
                multiview: Some(MultiViewSettings::from_baseline(1.).unwrap()),
                ..Default::default()
            };
            settings.principal_point_jitter = Vec2::splat(0.15);
            settings.multiview.as_mut().unwrap().graph = OverlapGraph::Connected;
            scene.resample_cameras(12, settings, aspect).unwrap();
            assert!(scene.accepts_camera_visibility());
            assert_eq!(
                scene.camera_settings.multiview.as_ref().unwrap().graph,
                OverlapGraph::Connected
            );
            crate::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        }
    }
}
