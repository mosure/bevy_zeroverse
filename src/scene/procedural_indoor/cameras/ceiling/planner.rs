//! Joint search for mixed room/ceiling rigs. Roles and lenses are fixed before
//! search; all cameras share synchronized geometry and visibility constraints.
use super::placement::{surface, MountSpace};
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::{Coverage, VisibilityView},
        diversity::Track,
        multiview::{pair_samples_cached, OverlapGraph, CHECK_TIMES},
        program::ViewSummary,
        rig::{self, MixedRigContract},
    },
    layout::{stream, IndoorManifest},
};
use rand::{seq::SliceRandom, Rng};

struct Candidate {
    camera: IndoorCamera,
    track: Track,
    views: Vec<VisibilityView>,
    summary: ViewSummary,
    score: f32,
}

pub(crate) fn sample(scene: &mut IndoorManifest, count: usize) -> Result<(), String> {
    scene.camera_settings.validate()?;
    scene.cameras.clear();
    if count == 0 {
        return Ok(());
    }
    let mounting = scene
        .camera_settings
        .ceiling_mount
        .clone()
        .unwrap_or_default();
    let policy = scene.camera_settings.multiview.clone();
    let policy = policy.as_ref();
    let contract = policy
        .and_then(MixedRigContract::new)
        .filter(|_| count > 1 && scene.camera_settings.overlap_mixture.is_none());
    let targets = scene
        .camera_settings
        .overlap_mixture
        .as_ref()
        .map_or_else(Vec::new, |m| m.targets(scene.seed, count));
    let mounted = (0..count).any(|i| scene.camera_settings.ceiling_mounted(scene.seed, i));
    let coverage = if mounted {
        Coverage::for_ceiling_search(scene)
    } else {
        Coverage::new(scene)
    };
    let space = MountSpace::new(scene);
    let (lo, hi) = scene.primary_room_bounds();
    let lo = lo + Vec2::splat(mounting.wall_clearance_m);
    let hi = hi - Vec2::splat(mounting.wall_clearance_m);
    if lo.cmpge(hi).any() {
        return Err("ceiling camera wall clearance leaves no mounting area".into());
    }
    let mut rng = stream(scene.seed, 0x4343_5456);
    // Fix fisheye lens, standoff and roll per slot before placement rejection.
    // The pinhole reference retains the content-conditioned intrinsic sampler:
    // a narrow fixed draw may not frame a person plus the required room content.
    // Peers keep their draws; their gaze can change to find rich shared content.
    let prototypes: Vec<_> = (0..count)
        .map(|index| {
            let mut prior = stream(scene.seed, 0x4343_0000 + index as u64);
            (
                scene.camera_settings.lens_for(scene.seed, index),
                scene
                    .camera_settings
                    .intrinsics_for(scene.seed, index)
                    .sample(&mut prior),
                prior.random_range(mounting.standoff_m[0]..=mounting.standoff_m[1]),
                prior.random_range(0.0..std::f32::consts::TAU),
            )
        })
        .collect();
    let narrowest_room_fov = prototypes
        .iter()
        .enumerate()
        .filter(|(i, (lens, ..))| {
            *i > 0 && lens.is_none() && !scene.camera_settings.ceiling_mounted(scene.seed, *i)
        })
        .map(|(_, (_, fov, ..))| *fov)
        .min_by(f32::total_cmp);
    // Keep camera zero as the reference, then place constrained mounts before
    // free room paths. Slot identities remain fixed; restore export order below.
    let mut most = 0;
    let mut rejected = [0usize; 4];
    let mut room_rejected = [0usize; 3];
    // Peer intrinsics are fixed per slot. Caps do not depend on positions;
    // changing the content-conditioned reference invalidates the cache below.
    let mut overlap_floors = vec![vec![None; count]; count];
    // The baseline program uses informed proposals. Keep failures bounded rather
    // than spending the legacy exact-policy budget on an infeasible mixed rig.
    let restarts = if contract.is_some() { 8 } else { 32 };
    for restart in 0..restarts {
        let mut order: Vec<_> = (0..count).collect();
        // Different constrained slots can block each other. Retry order, never
        // the roles or lens assignments, with an independent deterministic RNG.
        if restart % 4 >= 2 {
            order[1..].shuffle(&mut stream(scene.seed, 0x4f52_4445 + restart));
        }
        if restart % 2 == 0 {
            order[1..].sort_by_key(|&i| !scene.camera_settings.ceiling_mounted(scene.seed, i));
        }
        scene.cameras.clear();
        let mut tracks = Vec::with_capacity(count);
        let mut views: Vec<Vec<_>> = Vec::with_capacity(count);
        let mut summaries = Vec::with_capacity(count);
        // Retain a few feasible alternatives per prefix. A late constrained
        // slot can replace its predecessor rather than discard a good anchor.
        // All attempts share the original worst-case per-restart budget.
        let mut alternatives: Vec<Vec<Candidate>> = Vec::with_capacity(count);
        let mut proposals = 0;
        while scene.cameras.len() < count && proposals < 384 * count {
            let stage = scene.cameras.len();
            let index = order[stage];
            let (lens, fov, standoff, roll) = prototypes[index];
            let mut pool: Vec<Candidate> = Vec::with_capacity(4);
            let mut first = None;
            for attempt in 0..384 {
                if proposals >= 384 * count || first.is_some_and(|a| attempt >= a + 12) {
                    break;
                }
                proposals += 1;
                scene.camera_diagnostics.peer_proposals += 1;
                let camera = if !scene.camera_settings.ceiling_mounted(scene.seed, index) {
                    let proposed = if let Some(contract) =
                        contract.filter(|_| stage > 0 && attempt % 4 != 3)
                    {
                        super::coherent::room(
                            scene,
                            index,
                            fov,
                            views.first().and_then(|v| v.first()),
                            contract,
                            &mut rng,
                        )
                    } else {
                        super::room::propose(scene, index, &mut rng, &coverage)
                    };
                    let Some(mut camera) = proposed else {
                        rejected[0] += 1;
                        continue;
                    };
                    if let Some(contract) = contract {
                        if index != 0 {
                            camera.fov_degrees = fov;
                        }
                        if stage == 0
                            && mounted
                            && scene.camera_settings.handheld.is_none()
                            && rng.random_bool((0.65 * (1.0 - contract.baseline)) as f64)
                        {
                            // A room anchor aimed at upper walls cannot share
                            // much content with a downward overview. Propose a
                            // lower furnished target before visibility admission.
                            let old = camera.target.y;
                            camera.target.y = scene.floor_height(camera.target.xz())
                                + rng.random_range(0.2..=1.0);
                            if let Some(motion) = &mut camera.motion {
                                motion.target_end.y += camera.target.y - old;
                            }
                        }
                        if mounted && contract.baseline < 0.5 {
                            super::coherent::motion(
                                scene,
                                index,
                                count,
                                &mut camera,
                                contract,
                                &mut rng,
                            );
                        }
                        // Qualify a rich anchor before spending a whole prefix
                        // on it. Small lenses otherwise discover a blank patch
                        // only after many broad views have already been placed.
                        // Later proposals retain unrestricted anchor recovery;
                        // this is a search preference, not an admission rule.
                        if contract.baseline < 0.5
                            && attempt < 128
                            && scene.cameras.iter().all(IndoorCamera::is_ceiling_mounted)
                        {
                            if let Some(fov) = narrowest_room_fov.filter(|f| *f < 60.) {
                                let mut probe = camera.clone();
                                probe.fisheye = None;
                                probe.fov_degrees = fov;
                                if !coverage.suitable(&probe, false) {
                                    rejected[0] += 1;
                                    room_rejected[2] += 1;
                                    continue;
                                }
                            }
                        }
                        let require_person = super::super::sampling::requires_person(scene, index);
                        let reason = if !scene
                            .camera_settings
                            .accepts_motion(scene.seed, index, &camera)
                        {
                            Some(0)
                        } else if !scene.camera_curve_clear(&camera) {
                            Some(1)
                        } else if !coverage.suitable(&camera, require_person) {
                            Some(2)
                        } else {
                            None
                        };
                        if let Some(reason) = reason {
                            room_rejected[reason] += 1;
                            rejected[0] += 1;
                            continue;
                        }
                    }
                    camera
                } else {
                    let p = if let Some(contract) =
                        contract.filter(|_| stage > 0 && attempt % 4 != 3)
                    {
                        super::coherent::point(scene, true, contract, &mut rng).unwrap()
                    } else if let Some(policy) = policy.filter(|_| index > 0 && attempt % 3 != 0) {
                        let a = rng.random_range(0.0..std::f32::consts::TAU);
                        let d = rng.random_range(
                            policy.min_baseline.max(policy.min_reference_baseline)
                                ..=policy.max_baseline,
                        );
                        scene.cameras[0].start.xz() + Vec2::new(a.cos(), a.sin()) * d
                    } else {
                        Vec2::new(rng.random_range(lo.x..hi.x), rng.random_range(lo.y..hi.y))
                    };
                    if p.cmplt(lo).any() || p.cmpgt(hi).any() {
                        continue;
                    }
                    let (surface, normal) = surface(scene, p);
                    let start = surface + normal * standoff;
                    let rotation = Transform::IDENTITY.looking_to(normal, Vec3::Z).rotation
                        * Quat::from_rotation_z(roll);
                    let mut camera = IndoorCamera {
                        ceiling_mount: Some(CeilingCameraMount {
                            surface,
                            inward_normal: normal,
                            rotation,
                            standoff_m: standoff,
                            body_radius_m: mounting.body_radius_m,
                        }),
                        fisheye: lens,
                        principal_point_offset: None,
                        start,
                        end: start,
                        target: start + normal * 2.,
                        fov_degrees: fov,
                        motion: None,
                    };
                    super::trajectory::propose(scene, index, &mut camera, &mut rng);
                    if let Some(contract) = contract {
                        super::coherent::motion(
                            scene,
                            index,
                            count,
                            &mut camera,
                            contract,
                            &mut rng,
                        );
                    }
                    if !space.clear(scene, &camera)
                        || !scene
                            .camera_settings
                            .accepts_motion(scene.seed, index, &camera)
                    {
                        rejected[0] += 1;
                        continue;
                    }
                    camera
                };
                if scene
                    .cameras
                    .iter()
                    .any(|c| super::bodies_overlap(c, &camera))
                {
                    rejected[0] += 1;
                    continue;
                }
                let start = camera.start;
                if scene.cameras.iter().any(|c| {
                    c.start.distance(start) < 0.18_f32.min(policy.map_or(0.18, |p| p.min_baseline))
                }) {
                    continue;
                }
                let track = Track::new(&camera);
                if policy.is_some_and(|p| {
                    !rig::accepts_geometry(
                        p,
                        contract,
                        &scene.cameras,
                        &tracks,
                        Some((&camera, &track)),
                        stage + 1 == count,
                    )
                }) {
                    rejected[1] += 1;
                    continue;
                }
                let candidate_views: Vec<_> = if camera.motion.is_none()
                    && camera.start == camera.end
                {
                    vec![coverage.view(&camera, scene.camera_aspect_ratio, 0.); CHECK_TIMES.len()]
                } else {
                    CHECK_TIMES
                        .into_iter()
                        .map(|t| coverage.view(&camera, scene.camera_aspect_ratio, t))
                        .collect()
                };
                // Qualify the downward overview independently of peripheral
                // ceiling context in >180-degree lenses. Pair visibility below
                // still uses the complete calibrated sensor at every time.
                if camera.is_ceiling_mounted()
                    && !super::overview::suitable(scene, &camera, &candidate_views)
                {
                    rejected[2] += 1;
                    continue;
                }
                if let Some(policy) = policy.filter(|_| index > 0) {
                    let mut accepts = |parent: usize| {
                        let samples = pair_samples_cached(
                            &coverage,
                            &scene.cameras[parent],
                            &views[parent],
                            &camera,
                            &candidate_views,
                        );
                        if let Some(contract) = contract.filter(|_| targets.is_empty()) {
                            let floor =
                                *overlap_floors[order[parent]][index].get_or_insert_with(|| {
                                    contract.pair_overlap_floor(
                                        &scene.cameras[parent],
                                        &camera,
                                        scene.camera_aspect_ratio,
                                    )
                                });
                            contract.accepts_visibility(
                                policy,
                                &scene.cameras[parent],
                                &camera,
                                &samples,
                                floor,
                            )
                        } else if policy.graph == OverlapGraph::Connected {
                            let mut edge = policy.clone();
                            edge.min_reference_baseline = 0.;
                            edge.accepts(&samples)
                        } else {
                            targets.get(index - 1).map_or_else(
                                || policy.accepts(&samples),
                                |t| t.accepts(policy, &samples),
                            )
                        }
                    };
                    if !(if policy.graph == OverlapGraph::Connected {
                        (0..stage).any(accepts)
                    } else {
                        accepts(0)
                    }) {
                        rejected[3] += 1;
                        continue;
                    }
                }
                let summary = ViewSummary::new(&camera, &candidate_views[0], scene);
                let score = summary.score(&summaries, count, scene);
                first.get_or_insert(attempt);
                let candidate = Candidate {
                    camera,
                    track,
                    views: candidate_views,
                    summary,
                    score,
                };
                // Descending stable ordering keeps the first proposal on ties.
                let position = pool.partition_point(|c| c.score >= score);
                pool.insert(position, candidate);
                pool.truncate(4);
            }
            let selected = if pool.is_empty() {
                most = most.max(scene.cameras.len());
                let mut selected = None;
                while let Some(mut previous) = alternatives.pop() {
                    scene.cameras.pop();
                    tracks.pop();
                    views.pop();
                    summaries.pop();
                    if !previous.is_empty() {
                        selected = Some(previous.remove(0));
                        alternatives.push(previous);
                        scene.camera_diagnostics.prefix_backtracks += 1;
                        break;
                    }
                }
                selected
            } else {
                let selected = pool.remove(0);
                alternatives.push(pool);
                Some(selected)
            };
            let Some(candidate) = selected else {
                break;
            };
            if scene.cameras.is_empty() {
                // An alternate reference may have a different admitted pinhole
                // focal length. Never reuse caps from the previous calibration.
                for row in &mut overlap_floors {
                    row.fill(None);
                }
            }
            scene.cameras.push(candidate.camera);
            tracks.push(candidate.track);
            views.push(candidate.views);
            summaries.push(candidate.summary);
        }
        if scene.cameras.len() == count {
            let mut ordered: Vec<_> = std::mem::take(&mut scene.cameras)
                .into_iter()
                .zip(&order)
                .collect();
            ordered.sort_by_key(|(_, index)| **index);
            scene.cameras = ordered.into_iter().map(|(camera, _)| camera).collect();
            return Ok(());
        }
        most = most.max(scene.cameras.len());
        scene.camera_diagnostics.group_restarts += 1;
    }
    Err(format!("seed {}: mixed mounting rig placed {most}/{count} cameras within the search budget; rejections proposal_or_clearance/spacing/overview/overlap={rejected:?}, room_motion/clearance/content={room_rejected:?}; mounting clearance, lens assignments and overlap constraints were retained", scene.seed))
}
