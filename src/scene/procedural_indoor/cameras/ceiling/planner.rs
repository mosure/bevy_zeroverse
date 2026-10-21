//! Bounded static-rig search. Reuses the calibrated first-surface and group
//! geometry oracles, without moving-camera proposals or trajectory allocation.
use super::placement::{surface, MountSpace};
use super::*;
use crate::scene::procedural_indoor::{
    cameras::{
        coverage::Coverage,
        diversity::{geometry, Track},
        multiview::{OverlapGraph, OverlapSample},
        program::ViewSummary,
    },
    layout::{stream, IndoorManifest},
};
use rand::Rng;

pub(crate) fn sample(scene: &mut IndoorManifest, count: usize) -> Result<(), String> {
    scene.camera_settings.validate()?;
    scene.cameras.clear();
    if count == 0 {
        return Ok(());
    }
    let mounting = scene.camera_settings.ceiling_mount.as_ref().unwrap();
    let policy = scene.camera_settings.multiview.as_ref();
    let targets = scene
        .camera_settings
        .overlap_mixture
        .as_ref()
        .map_or_else(Vec::new, |m| m.targets(scene.seed, count));
    let coverage = Coverage::new(scene);
    let space = MountSpace::new(scene);
    let (lo, hi) = scene.primary_room_bounds();
    let lo = lo + Vec2::splat(mounting.wall_clearance_m);
    let hi = hi - Vec2::splat(mounting.wall_clearance_m);
    if lo.cmpge(hi).any() {
        return Err("ceiling camera wall clearance leaves no mounting area".into());
    }
    let mut rng = stream(scene.seed, 0x4343_5456);
    // Fix lens, standoff and roll per slot, before placement rejection. Neither
    // a failed anchor nor a larger camera count changes these assignments.
    let prototypes: Vec<_> = (0..count)
        .map(|index| {
            let mut prior = stream(scene.seed, 0x4343_0000 + index as u64);
            (
                mounting.lenses.sample(scene.seed, index),
                mounting.intrinsics.sample(&mut prior),
                prior.random_range(mounting.standoff_m[0]..=mounting.standoff_m[1]),
                prior.random_range(0.0..std::f32::consts::TAU),
            )
        })
        .collect();
    let mut most = 0;
    let mut rejected = [0usize; 4];
    for _ in 0..32 {
        scene.cameras.clear();
        let mut tracks = Vec::with_capacity(count);
        let mut views = Vec::with_capacity(count);
        let mut summaries = Vec::with_capacity(count);
        for (index, &(lens, fov, standoff, roll)) in prototypes.iter().enumerate() {
            let mut best = None;
            let mut first = None;
            for attempt in 0..384 {
                if first.is_some_and(|a| attempt >= a + 12) {
                    break;
                }
                scene.camera_diagnostics.peer_proposals += 1;
                let p = if let Some(policy) = policy.filter(|_| index > 0 && attempt % 3 != 0) {
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
                let camera = IndoorCamera {
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
                if !space.clear(scene, &camera) {
                    rejected[0] += 1;
                    continue;
                }
                if scene.cameras.iter().any(|c| {
                    c.ceiling_mount
                        .as_ref()
                        .is_some_and(|m| m.overlaps(camera.ceiling_mount.as_ref().unwrap()))
                }) {
                    rejected[0] += 1;
                    continue;
                }
                if scene.cameras.iter().any(|c| {
                    c.start.distance(start) < 0.18_f32.min(policy.map_or(0.18, |p| p.min_baseline))
                }) {
                    continue;
                }
                let track = Track::new(&camera);
                if policy
                    .is_some_and(|p| geometry(&tracks, Some(&track)).is_some_and(|g| !g.accepts(p)))
                {
                    rejected[1] += 1;
                    continue;
                }
                let view = coverage.view(&camera, scene.camera_aspect_ratio, 0.);
                // Overview views legitimately contain broad continuous floors.
                // Require distributed primary-room surfaces below the mount,
                // rather than a walkthrough-specific semantic-class quota.
                let mut hits = 0;
                let mut below = 0;
                let mut footprint_lo = Vec2::splat(f32::INFINITY);
                let mut footprint_hi = Vec2::splat(f32::NEG_INFINITY);
                for point in view.surfaces() {
                    hits += 1;
                    if point.y < start.y - 0.7 && scene.in_primary_room(point, 0.) {
                        below += 1;
                        footprint_lo = footprint_lo.min(point.xz());
                        footprint_hi = footprint_hi.max(point.xz());
                    }
                }
                if hits < 12
                    || below * 2 < hits
                    || (footprint_hi - footprint_lo).min_element() < 1.5
                {
                    rejected[2] += 1;
                    continue;
                }
                if let Some(policy) = policy.filter(|_| index > 0) {
                    let accepts = |parent: usize| {
                        let (forward, angle) = coverage.shared(&views[parent], &view);
                        let backward = coverage.shared(&view, &views[parent]).0;
                        let samples = [OverlapSample {
                            time: 0.,
                            reference_to_view: forward,
                            view_to_reference: backward,
                            baseline_m: scene.cameras[parent].start.distance(start),
                            mean_triangulation_degrees: angle,
                        }];
                        if policy.graph == OverlapGraph::Connected {
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
                        (0..index).any(accepts)
                    } else {
                        accepts(0)
                    }) {
                        rejected[3] += 1;
                        continue;
                    }
                }
                let summary = ViewSummary::new(&camera, &view, scene);
                let score = summary.score(&summaries, count, scene);
                first.get_or_insert(attempt);
                if best.as_ref().is_none_or(|(_, _, _, _, s)| score > *s) {
                    best = Some((camera, track, view, summary, score));
                }
            }
            let Some((camera, track, view, summary, _)) = best else {
                break;
            };
            scene.cameras.push(camera);
            tracks.push(track);
            views.push(view);
            summaries.push(summary);
        }
        if scene.cameras.len() == count {
            return Ok(());
        }
        most = most.max(scene.cameras.len());
        scene.camera_diagnostics.group_restarts += 1;
    }
    Err(format!("seed {}: ceiling-mounted rig placed {most}/{count} cameras within the search budget; rejections mount/spacing/overview/overlap={rejected:?}; mounting clearance, lens assignments and overlap constraints were retained", scene.seed))
}
