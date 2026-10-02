//! Support-aware population placement. Activity context biases continuous
//! candidates; no person is anchored to a fixed normalized room coordinate.
use super::*;
use rand::seq::{IteratorRandom, SliceRandom};
use rand_chacha::ChaCha8Rng;

fn standing_candidate(
    scene: &IndoorManifest,
    pose: HumanPoseKind,
    rng: &mut ChaCha8Rng,
) -> (Vec3, f32) {
    if pose == HumanPoseKind::StandingPresenting && rng.random_bool(0.75) {
        if let Some(board) = scene
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(o.kind, ObjectKind::Whiteboard | ObjectKind::Display)
                    && o.position.y > 0.6
                    && o.size.z < 0.2
            })
            .choose(rng)
        {
            // Wall displays face local +Z; people face local -Z. Sample a band
            // in front of the actual display, then face the audience with varied
            // body orientation. The normal placement test still rejects walls,
            // furniture, unsupported floor levels and other people.
            let lateral = rng.random_range(-1.0..1.0) * (board.size.x * 0.5 + 0.45);
            let distance = rng.random_range(0.65..1.65);
            let mut position = board.position
                + Quat::from_rotation_y(board.yaw) * Vec3::new(lateral, 0., distance);
            position.y = scene.floor_height(position.xz());
            return (position, board.yaw + PI + rng.random_range(-0.65..0.65));
        }
    }
    let x = rng.random_range(-0.42..0.42) * scene.room_size.x;
    let z = rng.random_range(-0.42..0.42) * scene.room_size.z;
    (
        Vec3::new(x, scene.floor_height(Vec2::new(x, z)), z),
        rng.random_range(-PI..PI),
    )
}

pub fn populate(scene: &mut IndoorManifest, density: f32) {
    if density == 0.0 {
        return;
    }
    let mut rng = stream(scene.seed, 39);
    let mut chairs: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::Chair)
        .map(|o| o.id)
        .collect();
    chairs.shuffle(&mut rng);
    for chair_id in chairs {
        if scene.humans.len() >= 16 || !rng.random_bool(density as f64) {
            continue;
        }
        let chair = &scene.objects[chair_id];
        let working_surface = chair.interaction_target.is_some();
        let poses = if working_surface {
            [
                HumanPoseKind::SeatedWorking,
                HumanPoseKind::SeatedListening,
                HumanPoseKind::SeatedTalking,
            ]
        } else {
            [
                HumanPoseKind::SeatedListening,
                HumanPoseKind::SeatedListening,
                HumanPoseKind::SeatedTalking,
            ]
        };
        let pose = poses[rng.random_range(0..3)];
        let person = sample_person(
            rng.random(),
            scene.objects.len() + scene.humans.len(),
            chair.position,
            chair.yaw,
            pose,
            Some(chair_id),
            chair.neighbor,
        );
        if placement_clear(scene, &person) {
            scene.humans.push(person);
        } else {
            scene.rejected_human_placements += 1;
        }
    }
    let standing = (density * 2.0).floor() as usize
        + usize::from(rng.random_bool((density * 2.0).fract() as f64));
    for _ in 0..standing {
        for _ in 0..96 {
            let pose = [
                HumanPoseKind::StandingRelaxed,
                HumanPoseKind::StandingPresenting,
                HumanPoseKind::StandingConversation,
                HumanPoseKind::StandingReading,
                HumanPoseKind::StandingWalking,
            ][rng.random_range(0..5)];
            let (p, yaw) = standing_candidate(scene, pose, &mut rng);
            let person = sample_person(
                rng.random(),
                scene.objects.len() + scene.humans.len(),
                p,
                yaw,
                pose,
                None,
                false,
            );
            if placement_clear(scene, &person) {
                scene.humans.push(person);
                break;
            } else {
                scene.rejected_human_placements += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presenters_use_continuous_supported_placements_without_a_shared_anchor() {
        use std::collections::BTreeSet;
        let mut positions = BTreeSet::new();
        let mut headings = BTreeSet::new();
        let mut cells = [0usize; 24 * 24];
        let mut count = 0;
        for seed in 0..512 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                super::super::super::layout::IndoorLayout::Mixed,
                0.65,
                0,
                0.25,
            )
            .unwrap();
            super::super::validate(&scene).unwrap();
            for person in scene
                .humans
                .iter()
                .filter(|p| !p.neighbor && p.pose == HumanPoseKind::StandingPresenting)
            {
                let p = person.position.xz() / scene.room_size.xz();
                assert!((p - Vec2::new(0.18, -0.34)).length() > 1e-5);
                assert_eq!(person.position.y, scene.floor_height(person.position.xz()));
                positions.insert((
                    (p.x * 10_000.).round() as i32,
                    (p.y * 10_000.).round() as i32,
                ));
                headings.insert(((person.yaw.rem_euclid(TAU) / TAU) * 24.) as u32);
                let cell = ((p + Vec2::splat(0.5)) * 24.)
                    .floor()
                    .as_uvec2()
                    .min(UVec2::splat(23));
                cells[(cell.y * 24 + cell.x) as usize] += 1;
                count += 1;
            }
        }
        assert!(
            count >= 30,
            "presenters must remain in the population: {count}"
        );
        assert_eq!(
            positions.len(),
            count,
            "repeated normalized presenter anchors"
        );
        assert!(
            headings.len() >= 12,
            "presenters must face varied directions"
        );
        assert!(
            *cells.iter().max().unwrap() < count / 6,
            "single-cell concentration"
        );
    }
}
