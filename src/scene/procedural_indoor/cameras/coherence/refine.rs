//! Coordinate refinement of room-camera gaze only. No new rig rejection, path,
//! lens, mounting-role, reference-camera or geometry changes are introduced.
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::{Coverage, VisibilityView},
        multiview::{pair_samples_cached, OverlapSample, CHECK_TIMES},
        IndoorCamera, IndoorManifest,
    },
    layout::stream,
};
use bevy::prelude::*;
use rand::Rng;

fn views(coverage: &Coverage, camera: &IndoorCamera, aspect: f32) -> Vec<VisibilityView> {
    if camera.motion.is_none() && camera.start == camera.end {
        vec![coverage.view(camera, aspect, 0.); CHECK_TIMES.len()]
    } else {
        CHECK_TIMES
            .iter()
            .map(|&t| coverage.view(camera, aspect, t))
            .collect()
    }
}

fn deficit(
    coverage: &Coverage,
    a: &[VisibilityView],
    b: &[VisibilityView],
    floor: f32,
    containment: f32,
) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let ab = coverage.shared(a, b).0;
            let ba = coverage.shared(b, a).0;
            // Reserve a separate score band for essentially disjoint pairs.
            // Improving moderate overlaps must not create a new negative pair.
            let critical = if ab.min(ba) < floor.min(0.01) { 4. } else { 0. };
            critical + (1. - ab.min(ba) / floor).max(0.) + (1. - ab.max(ba) / containment).max(0.)
        })
        .fold(0., f32::max)
}

fn weak_pairs(scores: &[Vec<f32>]) -> usize {
    scores
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, row)| row.iter().skip(i + 1).filter(|&&v| v > 0.).count())
        .sum()
}

pub(crate) fn refine(scene: &mut IndoorManifest) {
    let Some(policy) = scene.camera_coherence_policy() else {
        return;
    };
    let count = scene.cameras.len();
    // Fixed optical-axis mounts cannot be reframed. Avoid all refinement work
    // for all-mounted rigs and explicitly prescribed orientation trajectories.
    if scene.cameras.iter().skip(1).all(|c| c.is_ceiling_mounted()) {
        return;
    }
    let coverage = Coverage::new(scene);
    let original = scene.cameras.clone();
    let mut cache: Vec<_> = scene
        .cameras
        .iter()
        .map(|c| views(&coverage, c, scene.camera_aspect_ratio))
        .collect();
    let mut floors = vec![vec![0.; count]; count];
    let mut scores = floors.clone();
    for i in 1..count {
        for j in i + 1..count {
            let floor = policy.pair_overlap_target(
                &scene.cameras[i],
                &scene.cameras[j],
                scene.camera_aspect_ratio,
            );
            floors[i][j] = floor;
            floors[j][i] = floor;
            let score = deficit(
                &coverage,
                &cache[i],
                &cache[j],
                floor,
                policy.target_min_containment,
            );
            scores[i][j] = score;
            scores[j][i] = score;
        }
    }
    scene.camera_diagnostics.peer_weak_pairs_before = weak_pairs(&scores);
    let mut rng = stream(scene.seed, 0x5045_4552_4741_5a45);
    let mut budget = policy.max_candidates;
    let contract = scene.mixed_rig_contract();
    let reference_floors: Vec<_> = scene
        .cameras
        .iter()
        .map(|c| {
            contract.map_or(0., |c0| {
                c0.pair_overlap_floor(&scene.cameras[0], c, scene.camera_aspect_ratio)
            })
        })
        .collect();
    for _ in 0..2 {
        let mut order: Vec<_> = (1..count).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .iter()
                .sum::<f32>()
                .total_cmp(&scores[a].iter().sum())
        });
        let mut changed = false;
        for i in order {
            if budget == 0 {
                break;
            }
            let camera = &scene.cameras[i];
            if camera.is_ceiling_mounted()
                || camera
                    .motion
                    .as_ref()
                    .is_some_and(|m| m.orientations.is_some())
            {
                continue;
            }
            let Some(j) = (1..count)
                .filter(|&j| j != i && scores[i][j] > 0.)
                .max_by(|&a, &b| scores[i][a].total_cmp(&scores[i][b]))
            else {
                continue;
            };
            let initial = camera.clone();
            // Interpolate independent aims and actual first surfaces rather than
            // forcing every optical axis to one room-centre point.
            let mut aims = vec![scene.cameras[j].target, scene.cameras[0].target];
            let surfaces: Vec<_> = cache[j][2]
                .surfaces()
                .chain(cache[0][2].surfaces())
                .collect();
            for _ in 0..8 {
                if surfaces.is_empty() {
                    break;
                }
                aims.push(surfaces[rng.random_range(0..surfaces.len())]);
            }
            let mut best = None;
            let mut best_score = scores[i].iter().sum::<f32>();
            for (k, aim) in aims.into_iter().enumerate() {
                if budget == 0 {
                    break;
                }
                budget -= 1;
                scene.camera_diagnostics.peer_framing_candidates += 1;
                if best_score == 0. {
                    continue;
                }
                let mut candidate = initial.clone();
                let target = initial
                    .target
                    .lerp(aim, if k % 2 == 0 { 0.5 } else { 0.85 });
                let delta = target - candidate.target;
                candidate.target = target;
                if let Some(motion) = &mut candidate.motion {
                    motion.target_end += delta;
                }
                if CHECK_TIMES.iter().any(|&t| {
                    original[i]
                        .transform_at(t)
                        .rotation
                        .angle_between(candidate.transform_at(t).rotation)
                        > policy.max_turn_degrees.to_radians()
                }) {
                    continue;
                }
                let multiview = scene.camera_settings.multiview.as_ref().unwrap();
                let accepts_reference = |samples: &[OverlapSample]| {
                    contract.map_or_else(
                        || multiview.accepts(samples),
                        |c| {
                            c.accepts_visibility(
                                multiview,
                                &scene.cameras[0],
                                &candidate,
                                samples,
                                reference_floors[i],
                            )
                        },
                    )
                };
                // Midpoint loss is a lower bound on the worst-time objective.
                // Reject impossible improvements before tracing the other four
                // poses. This pruning cannot change the accepted solution.
                let mid = coverage.view(&candidate, scene.camera_aspect_ratio, 0.5);
                let sample = OverlapSample {
                    time: 0.5,
                    reference_to_view: coverage.shared(&cache[0][2], &mid).0,
                    view_to_reference: coverage.shared(&mid, &cache[0][2]).0,
                    baseline_m: scene.cameras[0]
                        .transform_at(0.5)
                        .translation
                        .distance(candidate.transform_at(0.5).translation),
                    mean_triangulation_degrees: 0.,
                };
                if !accepts_reference(&std::array::from_fn::<_, 5, _>(|_| sample.clone())) {
                    continue;
                }
                let mid_losses: Vec<_> = (0..count)
                    .map(|j| {
                        if j == 0 || j == i {
                            0.
                        } else {
                            deficit(
                                &coverage,
                                std::slice::from_ref(&mid),
                                &cache[j][2..3],
                                floors[i][j],
                                policy.target_min_containment,
                            )
                        }
                    })
                    .collect();
                if mid_losses.iter().sum::<f32>() + 0.001 >= best_score
                    || (1..count).any(|j| {
                        (scores[i][j] == 0. && mid_losses[j] > 0.)
                            || (scores[i][j] < 4. && mid_losses[j] >= 4.)
                    })
                {
                    continue;
                }
                let candidate_views: Vec<_> = CHECK_TIMES
                    .iter()
                    .map(|&t| {
                        if t == 0.5
                            || (candidate.motion.is_none() && candidate.start == candidate.end)
                        {
                            mid.clone()
                        } else {
                            coverage.view(&candidate, scene.camera_aspect_ratio, t)
                        }
                    })
                    .collect();
                let samples = pair_samples_cached(
                    &coverage,
                    &scene.cameras[0],
                    &cache[0],
                    &candidate,
                    &candidate_views,
                );
                if !accepts_reference(&samples) {
                    continue;
                }
                let mut next = vec![0.; count];
                let mut acceptable = true;
                for j in 1..count {
                    if j == i {
                        continue;
                    }
                    next[j] = deficit(
                        &coverage,
                        &candidate_views,
                        &cache[j],
                        floors[i][j],
                        policy.target_min_containment,
                    );
                    // No previously supported peer pair may become unsupported.
                    if (scores[i][j] == 0. && next[j] > 0.) || (scores[i][j] < 4. && next[j] >= 4.)
                    {
                        acceptable = false;
                        break;
                    }
                }
                let score = next.iter().sum::<f32>();
                if acceptable
                    && score + 0.001 < best_score
                    && scene.camera_curve_clear(&candidate)
                    && coverage.suitable(
                        &candidate,
                        super::super::sampling::requires_person(scene, i),
                    )
                {
                    best_score = score;
                    best = Some((candidate, candidate_views, next));
                }
            }
            if let Some((camera, views, next)) = best {
                scene.cameras[i] = camera;
                cache[i] = views;
                for j in 1..count {
                    scores[i][j] = next[j];
                    scores[j][i] = next[j];
                }
                scene.camera_diagnostics.peer_framing_updates += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    scene.camera_diagnostics.peer_weak_pairs_after = weak_pairs(&scores);
}
