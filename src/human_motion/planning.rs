//! Reproducible affordance-aware requests and swept navigation through real doors.
use super::{prompts, HumanMotionConfig, MotionPlan, MotionRejection};
use crate::scene::procedural_indoor::layout::{
    segment_hits_box, stream, IndoorManifest, ObjectKind, NEIGHBOR_DEPTH,
};
use bevy::prelude::*;
use burn_human_motion::Waypoint;
use rand::{seq::SliceRandom, Rng};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

pub const BODY_CLEARANCE: f32 = 0.34;

/// Independent ARDY noise for a scene/actor pair, separate from appearance and
/// prompt sampling. Replaying the pair repeats the seed; regenerating the scene
/// changes it even when the same person and prompt are reused. Random access
/// keeps actor selection, batch order and preceding planning failures irrelevant.
pub fn motion_seed(scene_seed: u64, actor_id: usize) -> u64 {
    let mut rng = stream(scene_seed, 0x4d4f544e4f495345); // MOTNOISE
    rng.set_word_pos(actor_id as u128 * 2);
    rng.random()
}

/// Condition in a person-centred frame rather than asking the model to denoise
/// large absolute room offsets. This is a rigid coordinate change, not path
/// snapping: generated errors remain visible to the admission checks.
pub fn canonical_request(
    plan: &MotionPlan,
    fallback_heading: f32,
) -> (burn_human_motion::MotionRequest, Transform) {
    let first = &plan.request.waypoints[0];
    let yaw = first.heading.unwrap_or(fallback_heading);
    let world = Transform::from_translation(first.position.with_y(0.0))
        .with_rotation(Quat::from_rotation_y(yaw));
    let inverse = world.to_matrix().inverse();
    let mut request = plan.request.clone();
    for w in &mut request.waypoints {
        w.position = inverse.transform_point3(w.position);
        w.heading = w.heading.map(|h| (h - yaw).sin().atan2((h - yaw).cos()));
    }
    (request, world)
}

pub fn scene_clip(
    mut clip: burn_human_motion::MotionClip,
    transform: Transform,
) -> burn_human_motion::MotionClip {
    for frame in &mut clip.frames {
        frame.root_translation = transform.transform_point(frame.root_translation);
        frame.local_rotations[0] = (transform.rotation * frame.local_rotations[0]).normalize();
    }
    clip.provenance.push_str(&format!(
        "; scene_from_motion={:?}",
        transform.to_matrix().to_cols_array()
    ));
    clip
}

/// Glass partitions and the open leaf are solid navigation obstacles. The
/// main doorway is open only between its actual frame edges.
pub fn obstacles(
    scene: &IndoorManifest,
    actor: usize,
    support: Option<usize>,
) -> Vec<(Vec3, Vec3)> {
    let half = scene.room_size * 0.5;
    let mut boxes = scene.camera_obstacles();
    let wall = |a: Vec3, b: Vec3| (a, b);
    boxes.extend([
        wall(
            Vec3::new(-half.x - 0.2, 0.0, -half.z - 0.2),
            Vec3::new(-half.x + 0.08, scene.room_size.y, half.z + NEIGHBOR_DEPTH),
        ),
        wall(
            Vec3::new(half.x - 0.08, 0.0, -half.z - 0.2),
            Vec3::new(half.x + 0.2, scene.room_size.y, half.z + NEIGHBOR_DEPTH),
        ),
        wall(
            Vec3::new(-half.x, 0.0, -half.z - 0.2),
            Vec3::new(half.x, scene.room_size.y, -half.z + 0.08),
        ),
        wall(
            Vec3::new(-half.x, 0.0, half.z + NEIGHBOR_DEPTH - 0.08),
            Vec3::new(half.x, scene.room_size.y, half.z + NEIGHBOR_DEPTH + 0.2),
        ),
        wall(
            Vec3::new(-half.x, 0.0, half.z - 0.06),
            Vec3::new(scene.door_x - 0.50, scene.room_size.y, half.z + 0.06),
        ),
        wall(
            Vec3::new(scene.door_x + 0.50, 0.0, half.z - 0.06),
            Vec3::new(half.x, scene.room_size.y, half.z + 0.06),
        ),
        wall(
            Vec3::new(scene.door_x + 0.48, 0.0, half.z + 0.01),
            Vec3::new(scene.door_x + 0.57, 2.26, half.z + 1.11),
        ),
    ]);
    boxes.extend(
        scene
            .objects
            .iter()
            .filter(|o| Some(o.id) != support)
            .map(|o| o.bounds()),
    );
    boxes.extend(
        scene
            .humans
            .iter()
            .filter(|h| h.id != actor)
            .map(|h| h.bounds()),
    );
    boxes
}

pub fn segment_clear(a: Vec3, b: Vec3, height: f32, radius: f32, boxes: &[(Vec3, Vec3)]) -> bool {
    let pad = Vec3::new(radius, 0.0, radius);
    boxes
        .iter()
        .filter(|(lo, hi)| hi.y > 0.04 && lo.y < height)
        .all(|(lo, hi)| {
            !segment_hits_box(
                a.with_y(height * 0.5),
                b.with_y(height * 0.5),
                (*lo - pad).with_y(-1.0),
                (*hi + pad).with_y(height + 1.0),
            )
        })
}

/// A* uses swept edges (including diagonals), then visibility simplifies the
/// polyline. The final endpoints never snap to a grid cell inside furniture.
pub fn route(
    scene: &IndoorManifest,
    a: Vec3,
    b: Vec3,
    height: f32,
    radius: f32,
    boxes: &[(Vec3, Vec3)],
) -> Option<Vec<Vec3>> {
    if !segment_clear(a, a, height, radius, boxes) || !segment_clear(b, b, height, radius, boxes) {
        return None;
    }
    if segment_clear(a, b, height, radius, boxes) {
        return Some(vec![a, b]);
    }
    let step = 0.25;
    let hx = scene.room_size.x * 0.5;
    let hz = scene.room_size.z * 0.5;
    let pos = |(x, z): (i32, i32)| Vec3::new(x as f32 * step, 0.0, z as f32 * step);
    let cell = |p: Vec3| ((p.x / step).round() as i32, (p.z / step).round() as i32);
    let start = cell(a);
    let end = cell(b);
    if !segment_clear(a, pos(start), height, radius, boxes)
        || !segment_clear(pos(end), b, height, radius, boxes)
    {
        return None;
    }
    let heuristic = |p: (i32, i32)| (p.0 - end.0).abs().max((p.1 - end.1).abs()) * 10;
    let mut queue = BinaryHeap::from([Reverse((heuristic(start), 0, start))]);
    let mut cost = HashMap::from([(start, 0)]);
    let mut parents = HashMap::new();
    while let Some(Reverse((_, g, current))) = queue.pop() {
        if current == end {
            let mut path = vec![b, pos(end)];
            let mut c = end;
            while c != start {
                c = parents[&c];
                path.push(pos(c));
            }
            path.push(a);
            path.reverse();
            let mut simplified = vec![a];
            let mut i = 0;
            while i + 1 < path.len() {
                let j = (i + 1..path.len())
                    .rev()
                    .find(|&j| segment_clear(path[i], path[j], height, radius, boxes))?;
                simplified.push(path[j]);
                i = j;
            }
            return Some(simplified);
        }
        if cost.len() > 12000 {
            return None;
        }
        if g != cost[&current] {
            continue;
        }
        for dx in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let next = (current.0 + dx, current.1 + dz);
                let p = pos(next);
                if p.x.abs() > hx - radius
                    || p.z < -hz + radius
                    || p.z > hz + NEIGHBOR_DEPTH - radius
                    || !segment_clear(pos(current), p, height, radius, boxes)
                {
                    continue;
                }
                let next_cost = g + if dx == 0 || dz == 0 { 10 } else { 14 };
                if cost.get(&next).is_none_or(|old| next_cost < *old) {
                    parents.insert(next, current);
                    cost.insert(next, next_cost);
                    queue.push(Reverse((next_cost + heuristic(next), next_cost, next)));
                }
            }
        }
    }
    None
}

fn waypoints(path: &[Vec3], frames: usize) -> Vec<Waypoint> {
    let total: f32 = path.windows(2).map(|p| p[0].distance(p[1])).sum();
    let mut distance = 0.0;
    let mut points: Vec<Waypoint> = Vec::new();
    for (i, &p) in path.iter().enumerate() {
        if i > 0 {
            distance += p.distance(path[i - 1]);
        }
        let frame = if i + 1 == path.len() {
            frames - 1
        } else if total > 0.001 {
            (distance / total * (frames - 1) as f32).round() as usize
        } else {
            0
        };
        let delta = if i > 0 && i + 1 < path.len() {
            path[i + 1] - path[i - 1]
        } else if i + 1 < path.len() {
            path[i + 1] - p
        } else {
            // A tiny terminal grid connector must not demand a 180-degree
            // turn in the last two frames. Face along the approach corridor.
            let mut j = i - 1;
            while j > 0 && p.distance(path[j]) < 0.5 {
                j -= 1;
            }
            p - path[j]
        };
        let heading = (delta.length_squared() > 1e-6).then(|| delta.x.atan2(delta.z));
        if points.last().is_some_and(|last| last.frame == frame) {
            continue;
        }
        points.push(Waypoint {
            frame,
            position: p,
            heading,
            constrain_height: false,
        });
    }
    points
}

/// Selection is independent of behavior search and model rejection. A high
/// fraction cannot silently become a second Bernoulli sampling after staging.
fn selected_actors(scene: &IndoorManifest, config: &HumanMotionConfig) -> Vec<usize> {
    let mut ids: Vec<_> = scene
        .humans
        .iter()
        .map(|h| h.id)
        .filter(|id| !config.trajectories.iter().any(|t| t.actor_id == *id))
        .collect();
    ids.shuffle(&mut stream(scene.seed, 0x53454c454354));
    let count = ((scene.humans.len() as f32 * config.fraction).ceil() as usize)
        .min(config.max_actors.saturating_sub(config.trajectories.len()));
    ids.truncate(count);
    ids
}

/// Furnish first, then stage moving actors in walkable free space. Seated people
/// are not forced through their desks. This happens before mesh, GI and camera
/// preparation, so a rejected clip still leaves a valid static person in place.
pub fn prepare_scene(scene: &mut IndoorManifest, config: &HumanMotionConfig) -> Result<(), String> {
    config.validate()?;
    let selected = selected_actors(scene, config);
    let count = (selected.len() as f32 * config.locomotion_fraction).round() as usize;
    if count == 0 {
        return Ok(());
    }
    for id in selected.into_iter().take(count) {
        let index = scene.humans.iter().position(|h| h.id == id).unwrap();
        let original = scene.humans[index].clone();
        let mut rng = stream(original.seed, 0x5354414745);
        let boxes = obstacles(scene, id, None);
        let radius = BODY_CLEARANCE.max(original.shoulder_width * 0.75);
        for attempt in 0..128 {
            let p = if attempt == 0 && original.chair.is_none() {
                original.position
            } else {
                Vec3::new(
                    rng.random_range(-0.40..0.40) * scene.room_size.x,
                    0.0,
                    rng.random_range(-0.40..0.40) * scene.room_size.z,
                )
            };
            if !segment_clear(p, p, original.stature, radius + 0.12, &boxes) {
                continue;
            }
            let yaw = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
            let candidate = crate::scene::procedural_indoor::humans::standing_at(
                &original,
                p,
                yaw,
                p.z > scene.room_size.z * 0.5,
            );
            if !crate::scene::procedural_indoor::humans::placement_clear(scene, &candidate) {
                continue;
            }
            // Reject isolated free pockets with no useful locomotion segment.
            if !(0..16).any(|k| {
                let a = k as f32 * std::f32::consts::TAU / 16.0;
                let end = p + Vec3::new(a.cos(), 0.0, a.sin()) * 1.4;
                segment_clear(p, end, original.stature, radius, &boxes)
            }) {
                continue;
            }
            scene.humans[index] = candidate;
            break;
        }
    }
    // Camera framing and clearance must use the staged people, not old chairs.
    let cameras = scene.cameras.len();
    scene.cameras.clear();
    scene.sample_cameras(cameras)?;
    crate::scene::procedural_indoor::humans::validate(scene)
}

/// Minimum simultaneous separation along piecewise linear timed paths. This
/// allows two people to use an aisle at different times, unlike a full-path AABB.
fn paths_conflict(a: &[Waypoint], b: &[Waypoint]) -> bool {
    for aa in a.windows(2) {
        for bb in b.windows(2) {
            let start = aa[0].frame.max(bb[0].frame);
            let end = aa[1].frame.min(bb[1].frame);
            if start > end {
                continue;
            }
            let at = |p: &[Waypoint], f: usize| {
                p[0].position
                    .lerp(
                        p[1].position,
                        (f - p[0].frame) as f32 / (p[1].frame - p[0].frame) as f32,
                    )
                    .with_y(0.0)
            };
            let x = at(aa, start) - at(bb, start);
            let d = at(aa, end) - at(bb, end) - x;
            let t = (-x.dot(d) / d.length_squared().max(1e-8)).clamp(0.0, 1.0);
            if (x + d * t).length() < 0.80 {
                return true;
            }
        }
    }
    false
}

fn camera_path_clear(scene: &IndoorManifest, plan: &MotionPlan, stature: f32) -> bool {
    (0..plan.request.frames).all(|frame| {
        let position = super::validation::expected_position(plan, frame).with_y(0.0);
        let t = frame as f32 / (plan.request.frames - 1) as f32;
        scene.cameras.iter().all(|camera| {
            let p = camera.transform_at(t).translation;
            p.y > stature + 0.15 || p.with_y(0.0).distance(position) > 0.65
        })
    })
}

pub fn plan(
    scene: &IndoorManifest,
    config: &HumanMotionConfig,
) -> Result<(Vec<MotionPlan>, Vec<MotionRejection>), String> {
    config.validate()?;
    let half = scene.room_size * 0.5;
    for explicit in &config.trajectories {
        if explicit.waypoints.iter().any(|w| {
            w.position.x.abs() > half.x - BODY_CLEARANCE
                || w.position.z < -half.z + BODY_CLEARANCE
                || w.position.z > half.z + NEIGHBOR_DEPTH - BODY_CLEARANCE
        }) {
            return Err(format!(
                "motion actor {} leaves the generated floor",
                explicit.actor_id
            ));
        }
    }

    let mut rng = stream(scene.seed, 0x4d4f54494f4e);
    let mut plans = Vec::new();
    let mut rejected = Vec::new();
    let mut reserved: Vec<Vec<Waypoint>> = Vec::new();
    for explicit in &config.trajectories {
        let Some(person) = scene.humans.iter().find(|h| h.id == explicit.actor_id) else {
            return Err(format!("motion actor {} does not exist", explicit.actor_id));
        };
        let support = explicit.support_chair.or(person.chair);
        if let Some(id) = support {
            if !scene
                .objects
                .iter()
                .any(|o| o.id == id && o.kind == ObjectKind::Chair)
            {
                return Err(format!(
                    "motion actor {} references a missing support chair",
                    person.id
                ));
            }
        }
        let boxes = obstacles(scene, person.id, support);
        if explicit.waypoints.windows(2).any(|w| {
            !segment_clear(
                w[0].position,
                w[1].position,
                person.stature,
                if support.is_some() {
                    0.30
                } else {
                    BODY_CLEARANCE
                },
                &boxes,
            )
        }) {
            return Err(format!(
                "motion actor {} trajectory intersects an obstacle",
                person.id
            ));
        }
        let request = config.request(
            explicit.prompt.clone(),
            motion_seed(scene.seed, person.id),
            explicit.waypoints.clone(),
        );
        if reserved
            .iter()
            .any(|p| paths_conflict(&request.waypoints, p))
        {
            return Err(format!(
                "motion actor {} conflicts with another timed path",
                person.id
            ));
        }
        let plan = MotionPlan {
            actor_id: person.id,
            behavior: "explicit".into(),
            request,
            support_chair: support,
            prompt_recipe: None,
        };
        if !camera_path_clear(scene, &plan, person.stature) {
            return Err(format!(
                "motion actor {} trajectory intersects a camera path",
                person.id
            ));
        }
        reserved.push(plan.request.waypoints.clone());
        plans.push(plan);
    }
    for id in selected_actors(scene, config) {
        let person = scene.humans.iter().find(|h| h.id == id).unwrap();
        let mut found = None;
        // Sample a family per actor before searching geometry. Resampling it on
        // every failed path would overwhelm travel/exercise weights with the
        // easiest stationary gestures, especially in furnished rooms.
        let preferred_family = config.prompt_sampling.sample_family(&mut rng);
        for attempt in 0..48 {
            let family = if attempt < 32 {
                preferred_family
            } else {
                config.prompt_sampling.sample_family(&mut rng)
            };
            let support = person.chair;
            let boxes = obstacles(scene, person.id, support);
            let headroom = (scene.room_size.y - person.stature - 0.05).max(0.0);
            let mut points;
            let draft = if let Some(id) = support {
                let chair = scene
                    .objects
                    .iter()
                    .find(|o| o.id == id && o.kind == ObjectKind::Chair)
                    .ok_or("missing support chair")?;
                if family == prompts::Family::Locomotion {
                    let front = chair.position
                        + Quat::from_rotation_y(chair.yaw) * Vec3::new(0.0, 0.0, -0.72);
                    let sit = rng.random_bool(0.5);
                    let path = if sit {
                        [front, person.position]
                    } else {
                        [person.position, front]
                    };
                    points = waypoints(&path, config.frames);
                    for (i, w) in points.iter_mut().enumerate() {
                        w.heading = Some(person.yaw + std::f32::consts::PI);
                        w.constrain_height = true;
                        w.position.y = if sit == (i == 0) { 0.94 } else { 0.61 };
                    }
                    prompts::chair(if sit { "sit" } else { "stand" }, config, &mut rng)
                } else {
                    let Some(action) = prompts::sample_action(
                        &config.prompt_sampling,
                        Some(family),
                        config.frames - 1,
                        headroom,
                        true,
                        &mut rng,
                    ) else {
                        continue;
                    };
                    points = waypoints(&[person.position, person.position], config.frames);
                    for w in &mut points {
                        w.heading = Some(person.yaw + std::f32::consts::PI);
                    }
                    prompts::stationary(action, &mut points, config, true, &mut rng)
                }
            } else if family == prompts::Family::Locomotion {
                let doorway = Vec3::new(scene.door_x, 0.0, half.z + 1.60);
                let (a, b, behavior) = match rng.random_range(0..10) {
                    0 => (doorway, person.position, "enter"),
                    1 => (person.position, doorway, "leave"),
                    _ => {
                        let angle = rng.random_range(0.0..std::f32::consts::TAU);
                        let distance =
                            rng.random_range(1.2..(config.frames as f32 / 20.0 * 0.9).max(1.4));
                        (
                            person.position,
                            person.position + Vec3::new(angle.cos(), 0.0, angle.sin()) * distance,
                            "walk",
                        )
                    }
                };
                let Some(path) = route(scene, a, b, person.stature, BODY_CLEARANCE, &boxes) else {
                    continue;
                };
                let length: f32 = path.windows(2).map(|p| p[0].distance(p[1])).sum();
                if length < 1.2 || length > config.frames as f32 / 20.0 * 1.25 {
                    continue;
                }
                points = waypoints(&path, config.frames);
                let sequence = (behavior == "walk")
                    .then(|| {
                        prompts::walking_sequence(
                            &mut points,
                            config,
                            person.stature,
                            headroom,
                            &boxes,
                            &mut rng,
                        )
                    })
                    .flatten();
                sequence.unwrap_or_else(|| {
                    prompts::locomotion(behavior, &mut points, config, headroom, &mut rng)
                })
            } else {
                let Some(action) = prompts::sample_action(
                    &config.prompt_sampling,
                    Some(family),
                    config.frames - 1,
                    headroom,
                    false,
                    &mut rng,
                ) else {
                    continue;
                };
                points = waypoints(&[person.position, person.position], config.frames);
                for w in &mut points {
                    w.heading = Some(person.yaw);
                }
                prompts::stationary(action, &mut points, config, false, &mut rng)
            };
            let radius = if support.is_some() {
                if draft.behavior == "seated_gesture" {
                    0.25
                } else {
                    0.30
                }
            } else {
                draft.radius.max(BODY_CLEARANCE)
            };
            // Check the final timed polyline too: integer frame allocation must
            // not delete a navigation corner and cut across an obstacle.
            if points.windows(2).any(|p| {
                !segment_clear(
                    p[0].position,
                    p[1].position,
                    person.stature + draft.headroom + 0.05,
                    radius,
                    &boxes,
                )
            }) {
                continue;
            }
            if draft.behavior == "seated_gesture" {
                // The supported lower body is already placed. Upper-body gestures
                // need clearance above desktops, rather than a floor-to-head disk.
                let upper: Vec<_> = boxes
                    .iter()
                    .copied()
                    .filter(|(_, hi)| hi.y > 0.95)
                    .collect();
                if !segment_clear(
                    person.position,
                    person.position,
                    person.stature + 0.05,
                    draft.radius,
                    &upper,
                ) {
                    continue;
                }
            }
            let request = config.request(draft.text, motion_seed(scene.seed, person.id), points);
            if reserved
                .iter()
                .any(|p| paths_conflict(&request.waypoints, p))
            {
                continue;
            }
            let plan = MotionPlan {
                actor_id: person.id,
                behavior: draft.behavior,
                request,
                support_chair: support,
                prompt_recipe: Some(draft.recipe),
            };
            if !camera_path_clear(scene, &plan, person.stature) {
                continue;
            }
            reserved.push(plan.request.waypoints.clone());
            found = Some(plan);
            break;
        }
        if let Some(plan) = found {
            plans.push(plan);
        } else {
            rejected.push(MotionRejection {
                actor_id: person.id,
                reason: "no collision-free behavior fits the available space".into(),
            });
        }
    }
    Ok((plans, rejected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_seeds_separate_scenes_actors_and_high_bits() {
        let mut seen = std::collections::HashSet::new();
        for scene in 0..4096 {
            for actor in 0..16 {
                let seed = motion_seed(scene, actor);
                assert!(seen.insert(seed), "repeated noise seed at {scene}/{actor}");
                assert_eq!(seed, motion_seed(scene, actor));
                assert_ne!(seed, motion_seed(scene ^ (1 << 63), actor));
            }
        }
        let forward: Vec<_> = (0..16).map(|id| motion_seed(42, id)).collect();
        let reverse: Vec<_> = (0..16).rev().map(|id| motion_seed(42, id)).collect();
        assert!(forward.iter().eq(reverse.iter().rev()));
    }

    #[test]
    fn repeated_prompt_and_person_get_scene_specific_noise_without_changing_conditioning() {
        use super::super::HumanTrajectory;
        use crate::scene::procedural_indoor::layout::IndoorLayout;
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.35, 0, 0.7).unwrap();
        let mut config = HumanMotionConfig {
            fraction: 1.0,
            locomotion_fraction: 1.0,
            max_actors: 16,
            ..default()
        };
        prepare_scene(&mut scene, &config).unwrap();
        let (automatic, _) = plan(&scene, &config).unwrap();
        assert!(!automatic.is_empty());
        for p in &automatic {
            assert_eq!(p.request.seed, motion_seed(scene.seed, p.actor_id));
        }
        let first = &automatic[0];
        config.fraction = 0.0;
        config.trajectories = vec![HumanTrajectory {
            actor_id: first.actor_id,
            prompt: "A person waves the right hand.".into(),
            support_chair: first.support_chair,
            waypoints: first.request.waypoints.clone(),
        }];
        let request = |scene: &IndoorManifest| {
            let (plans, rejected) = plan(scene, &config).unwrap();
            assert!(rejected.is_empty());
            assert_eq!(plans.len(), 1);
            let canonical = canonical_request(&plans[0], 0.0).0;
            assert_eq!(canonical.seed, plans[0].request.seed);
            canonical
        };
        let a = request(&scene);
        // Hold geometry, identity, prompt and waypoints fixed; only the scene
        // seed changes, as it does when advancing the indoor sequence.
        scene.seed += 1;
        let b = request(&scene);
        assert_ne!(a.seed, b.seed);
        let mut same_conditioning = b.clone();
        same_conditioning.seed = a.seed;
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&same_conditioning).unwrap()
        );
        scene.seed -= 1;
        for h in &mut scene.humans {
            h.seed = h.seed.wrapping_add(1); // appearance is not the noise source
        }
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(request(&scene)).unwrap()
        );
    }

    fn wp(frame: usize, x: f32, z: f32) -> Waypoint {
        Waypoint {
            frame,
            position: Vec3::new(x, 0.0, z),
            heading: None,
            constrain_height: false,
        }
    }
    #[test]
    fn reservations_use_simultaneous_paths_not_their_union_aabbs() {
        let a = [wp(0, -2.0, 0.0), wp(100, 2.0, 0.0)];
        assert!(paths_conflict(&a, &[wp(0, 0.0, -2.0), wp(100, 0.0, 2.0)]));
        assert!(!paths_conflict(&a, &[wp(0, 0.0, -4.0), wp(100, 0.0, 0.0)]));
        assert!(!paths_conflict(
            &a,
            &[wp(101, -2.0, 0.0), wp(200, 2.0, 0.0)]
        ));
    }
    #[test]
    fn selected_fraction_is_a_quota_and_staging_preserves_identity_and_clearance() {
        use crate::scene::procedural_indoor::layout::IndoorLayout;
        for seed in 0..12 {
            let mut scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 0, 0.7)
                    .unwrap();
            let config = HumanMotionConfig {
                fraction: 1.0,
                locomotion_fraction: 1.0,
                max_actors: 16,
                ..default()
            };
            let before = scene.humans.clone();
            assert_eq!(
                selected_actors(&scene, &config).len(),
                scene.humans.len().min(16)
            );
            prepare_scene(&mut scene, &config).unwrap();
            let first = serde_json::to_string(&scene.humans).unwrap();
            scene.humans = before.clone();
            prepare_scene(&mut scene, &config).unwrap();
            assert_eq!(first, serde_json::to_string(&scene.humans).unwrap());
            for (a, b) in before.iter().zip(&scene.humans) {
                assert_eq!(
                    (a.id, a.seed, a.stature, a.outfit),
                    (b.id, b.seed, b.stature, b.outfit)
                );
                assert_eq!(a.appearance, b.appearance);
                assert!(crate::scene::procedural_indoor::humans::placement_clear(
                    &scene, b
                ));
            }
        }
    }
}
