//! Place an existing standing occupant at a usable length of installed worktop.
use super::*;
use crate::scene::procedural_indoor::{layout::IndoorObject, objects::kitchen};

pub(super) fn installed_counters(scene: &IndoorManifest) -> Vec<usize> {
    scene
        .assembly_diagnostics
        .kitchens
        .iter()
        .flat_map(|k| k.units.iter().copied())
        .filter(|&id| {
            scene
                .objects
                .get(id)
                .is_some_and(|o| o.kind == ObjectKind::KitchenCounter && !o.neighbor)
        })
        .collect()
}

fn hands_over(person: &IndoorHuman, counter: &IndoorObject) -> bool {
    let inv = counter.transform().compute_affine().inverse() * person.transform().compute_affine();
    let (low, high) = kitchen::counter_top_bounds(counter);
    [8, 12].into_iter().any(|index| {
        let hand = inv.transform_point3(person.joints[index]);
        let height = hand.y - counter.size.y;
        (0.055..0.26).contains(&height)
            && hand.xz().cmpgt(low + Vec2::splat(0.025)).all()
            && hand.xz().cmplt(high - Vec2::splat(0.025)).all()
    })
}

pub(in super::super) fn valid_relationship(scene: &IndoorManifest, person: &IndoorHuman) -> bool {
    match (person.pose, person.workstation) {
        (HumanPoseKind::StandingPreparing, Some(id)) => {
            scene.objects.get(id).is_some_and(|counter| {
                counter.kind == ObjectKind::KitchenCounter
                    && counter.neighbor == person.neighbor
                    && scene
                        .assembly_diagnostics
                        .kitchens
                        .iter()
                        .any(|k| k.units.contains(&id))
                    && hands_over(person, counter)
            })
        }
        (HumanPoseKind::StandingPreparing, None) | (_, Some(_)) => false,
        (_, None) => true,
    }
}

pub(super) fn stage(scene: &mut IndoorManifest, first: usize) {
    let counters = installed_counters(scene);
    if counters.is_empty() {
        return;
    }
    let occupancy = stream(scene.seed, 0x5052_4550_5241_5445).random_range(0.40..0.88);
    for index in first..scene.humans.len() {
        let mut rng = stream(scene.seed, 0x5052_4550_574f_524b ^ (index - first) as u64);
        if rng.random_bool(occupancy) {
            if let Some(worker) = fit(scene, &counters, &scene.humans[index], &mut rng) {
                scene.humans[index] = worker;
            }
        }
    }
}

fn fit(
    scene: &IndoorManifest,
    counters: &[usize],
    original: &IndoorHuman,
    rng: &mut ChaCha8Rng,
) -> Option<IndoorHuman> {
    // Retain the admitted actor's phenotype, clothing and ordinary fallback position.
    let mut person = original.clone();
    person.pose = HumanPoseKind::StandingPreparing;
    person.head_yaw *= 0.35;
    let mut pose = poses::PoseProgram::sample(person.seed, person.pose);
    pose.head_pitch = rng.random_range(-0.42..-0.12);
    pose.lean.y = rng.random_range(-0.38..-0.15);
    pose.stride *= 0.18;
    let torso = pose.torso_rotation();
    let neutral = pose.solve(person.stature, person.build, person.shoulder_width, false);
    let id_start = rng.random_range(0..counters.len());
    for attempt in 0..10 {
        let counter = &scene.objects[counters[(id_start + attempt) % counters.len()]];
        let (low, high) = kitchen::counter_top_bounds(counter);
        let side = if kitchen::is_island(counter) && rng.random_bool(0.5) {
            -1.
        } else {
            1.
        };
        let edge = if side > 0. { high.y } else { low.y };
        let lateral = rng.random_range(-0.72..0.72) * (high.x - low.x) * 0.5;
        person.yaw = counter.yaw + if side < 0. { PI } else { 0. } + rng.random_range(-0.30..0.30);
        let counter_tf = counter.transform();
        let relative = Quat::from_rotation_y(person.yaw - counter.yaw);
        // Stand back from the reserved fronts/handles and the individual's nearest
        // shoe. A fixed root-to-slab distance penalizes staggered or larger feet.
        let shoe_reach = [15, 16, 19, 20]
            .into_iter()
            .map(|joint| -side * (relative * neutral[joint]).z)
            .fold(0., f32::max)
            + 0.083 * person.stature / 1.75;
        let root_z = side * (counter.size.z * 0.5 + shoe_reach + rng.random_range(0.050..0.14));
        person.position = counter_tf.transform_point(Vec3::new(lateral, 0., root_z));
        person.position.y = scene.floor_height(person.position.xz());
        let inv = person.transform().compute_affine().inverse() * counter_tf.compute_affine();
        let active = rng.random_range(0..2);
        let both = rng.random_bool(0.42);
        for (i, shoulder) in [5, 9].into_iter().enumerate() {
            let arm_side = if i == 0 { -1. } else { 1. };
            if i != active && !both {
                pose.arm_elevation[i] = rng.random_range(-1.45..-1.02);
                pose.arm_reach[i] = rng.random_range(0.65..0.84);
                pose.arm_sweep[i] = rng.random_range(-0.15..0.24);
                continue;
            }
            let wrist_x = (lateral + arm_side * side * rng.random_range(0.09..0.23))
                .clamp(low.x + 0.04, high.x - 0.04);
            let goal = inv.transform_point3(Vec3::new(
                wrist_x,
                counter.size.y + rng.random_range(0.18..0.29),
                edge - side * rng.random_range(0.035..0.13),
            ));
            let delta = torso.inverse() * (goal - neutral[shoulder]);
            pose.aim_wrist(i, delta, person.stature);
        }
        person.joints = pose.solve(person.stature, person.build, person.shoulder_width, false);
        if !hands_over(&person, counter) {
            continue;
        }
        person.pose_program = Some(pose.clone());
        person.workstation = Some(counter.id);
        update_bounds(&mut person);
        if placement_clear(scene, &person) {
            return Some(person);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn kitchen_occupants_use_clear_counter_fronts_with_replayable_reach() {
        use std::collections::BTreeSet;
        let mut workers = 0;
        let mut visitors = 0;
        let mut lateral = BTreeSet::new();
        let mut yaw = BTreeSet::new();
        for seed in 2600..2664 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Kitchen, 0.65, 0, 0.75)
                    .unwrap();
            crate::scene::procedural_indoor::humans::validate(&scene).unwrap();
            for person in &scene.humans {
                let Some(id) = person.workstation else {
                    visitors += usize::from(!person.pose.seated());
                    continue;
                };
                let counter = &scene.objects[id];
                let local = counter
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(person.position);
                let (lo, _) = kitchen::counter_top_bounds(counter);
                let rear = local.z < lo.y;
                let sign = if rear { -1. } else { 1. };
                let relative = counter.transform().compute_affine().inverse()
                    * person.transform().compute_affine();
                let nearest_shoe = [15, 16, 19, 20]
                    .into_iter()
                    .map(|i| sign * relative.transform_point3(person.joints[i]).z)
                    .fold(f32::INFINITY, f32::min)
                    - 0.083 * person.stature / 1.75
                    - counter.size.z * 0.5;
                assert!((0.049..0.141).contains(&nearest_shoe));
                assert!(!rear || kitchen::is_island(counter));
                assert_eq!(person.position.y, scene.floor_height(person.position.xz()));
                let angle = (person.yaw - counter.yaw - if rear { PI } else { 0. } + PI)
                    .rem_euclid(TAU)
                    - PI;
                assert!(angle.abs() <= 0.301);
                lateral.insert((local.x / counter.size.x * 100.).round() as i32);
                yaw.insert((angle * 30.).round() as i32);
                assert!(hands_over(person, counter) && placement_clear(&scene, person));
                let walking = standing_at(person, person.position, person.yaw, false);
                assert!(walking.workstation.is_none() && valid_relationship(&scene, &walking));
                if workers == 0 {
                    let replay = IndoorManifest::generate_with_humans(
                        seed,
                        IndoorLayout::Kitchen,
                        0.65,
                        0,
                        0.75,
                    )
                    .unwrap();
                    assert_eq!(scene.humans, replay.humans);
                }
                workers += 1;
            }
        }
        assert!(
            workers >= 12 && visitors >= 12 && lateral.len() >= 8 && yaw.len() >= 5,
            "workers {workers}, visitors {visitors}, lateral {}, yaw {}",
            lateral.len(),
            yaw.len()
        );
    }

    #[test]
    fn blocked_workstations_fall_back_and_old_people_deserialize() {
        let mut scene =
            IndoorManifest::generate_with_humans(2631, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
        assert!(scene.humans.is_empty());
        let counters = installed_counters(&scene);
        assert!(!counters.is_empty());
        let mut block = scene.objects[counters[0]].clone();
        block.id = scene.objects.len();
        block.kind = ObjectKind::Cabinet;
        block.position = Vec3::ZERO;
        block.size = scene.room_size;
        block.yaw = 0.;
        scene.objects.push(block);
        let visitor = sample_person(
            42,
            scene.objects.len(),
            Vec3::ZERO,
            0.,
            HumanPoseKind::StandingReading,
            None,
            false,
        );
        assert!(
            (0..24).all(|slot| fit(&scene, &counters, &visitor, &mut stream(slot, 92)).is_none())
        );
        let legacy = serde_json::to_value(&visitor).unwrap();
        assert!(legacy.get("workstation").is_none());
        assert_eq!(
            serde_json::from_value::<IndoorHuman>(legacy).unwrap(),
            visitor
        );
        let mut bad = visitor;
        bad.pose = HumanPoseKind::StandingPreparing;
        assert!(!valid_relationship(&scene, &bad));
    }
}
