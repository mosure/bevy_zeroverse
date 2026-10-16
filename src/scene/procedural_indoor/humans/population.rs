//! Support-aware population placement. Activity context biases continuous
//! candidates; no person is anchored to a fixed normalized room coordinate.
use super::*;
use rand::seq::{IteratorRandom, SliceRandom};
use rand_chacha::ChaCha8Rng;

fn seated_pose(
    surface: bool,
    computer: bool,
    engagement: f32,
    rng: &mut ChaCha8Rng,
) -> HumanPoseKind {
    // A continuous engagement prior changes the mixture; it does not switch
    // everyone in a room to the same pose, or force people toward the cameras.
    let working = if computer {
        0.28 + 0.42 * engagement
    } else if surface {
        0.12 + 0.22 * engagement
    } else {
        0.
    };
    let talking = if computer {
        0.12 + 0.16 * (1. - engagement)
    } else {
        0.22 + 0.24 * (1. - engagement)
    };
    let draw: f32 = rng.random();
    if draw < working {
        HumanPoseKind::SeatedWorking
    } else if draw < working + talking {
        HumanPoseKind::SeatedTalking
    } else {
        HumanPoseKind::SeatedListening
    }
}

fn contextual_gaze(scene: &IndoorManifest, h: &mut IndoorHuman, rng: &mut ChaCha8Rng) {
    if !matches!(
        h.pose,
        HumanPoseKind::SeatedTalking
            | HumanPoseKind::SeatedListening
            | HumanPoseKind::StandingConversation
    ) || !rng.random_bool(0.65)
    {
        return;
    }
    let target = scene
        .humans
        .iter()
        .filter(|peer| peer.neighbor == h.neighbor)
        .filter_map(|peer| {
            let d = peer.position - h.position;
            let angle = ((-d.x).atan2(-d.z) - h.yaw + PI).rem_euclid(TAU) - PI;
            (d.xz().length_squared() < 16. && angle.abs() < 0.90).then_some((
                d.length_squared(),
                angle,
                peer.transform().transform_point(peer.joints[4]),
            ))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, angle, target)) = target {
        let eye = h.transform().transform_point(h.joints[4]);
        if scene
            .camera_obstacles()
            .iter()
            .any(|&(lo, hi)| super::super::layout::segment_hits_box(eye, target, lo, hi))
        {
            return;
        }
        h.head_yaw = (angle + rng.random_range(-0.10..0.10)).clamp(-0.85, 0.85);
        update_bounds(h);
    }
}

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
                    && (o.kind == ObjectKind::TeachingBoard
                        || matches!(o.kind, ObjectKind::Whiteboard | ObjectKind::Display)
                            && o.position.y > 0.6
                            && o.size.z < 0.2)
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
    if scene.layout.is_public() && rng.random_bool(0.60) {
        if let Some(anchor) = scene
            .objects
            .iter()
            .filter(|o| {
                matches!(
                    o.kind,
                    ObjectKind::InformationStand
                        | ObjectKind::ExhibitPlinth
                        | ObjectKind::ServiceCounter
                        | ObjectKind::Elevator
                )
            })
            .choose(rng)
        {
            let mut p = anchor.position
                + Quat::from_rotation_y(anchor.yaw)
                    * Vec3::new(
                        rng.random_range(-0.45..0.45),
                        0.,
                        anchor.size.z * 0.5 + rng.random_range(0.70..1.3),
                    );
            p.y = scene.floor_height(p.xz());
            return (p, anchor.yaw + rng.random_range(-0.45..0.45));
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
    let engagement = stream(scene.seed, 0x454e47414745).random_range(0.25..0.95);
    let computer_surfaces: std::collections::BTreeSet<_> = scene
        .objects
        .iter()
        .filter(|o| matches!(o.kind, ObjectKind::Laptop | ObjectKind::Monitor))
        .filter_map(|o| o.support)
        .collect();
    let mut chairs: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::Chair && !super::super::objects::chairs::is_folded(o))
        .map(|o| o.id)
        .collect();
    chairs.shuffle(&mut rng);
    for chair_id in chairs {
        if scene.humans.len() >= 16 || !rng.random_bool(density as f64) {
            continue;
        }
        let chair = &scene.objects[chair_id];
        let surface = chair
            .interaction_target
            .and_then(|i| scene.objects.get(i))
            .filter(|o| o.kind.is_surface());
        let working_surface = surface.is_some();
        let pose = seated_pose(
            working_surface,
            surface.is_some_and(|s| computer_surfaces.contains(&s.id)),
            engagement,
            &mut rng,
        );
        let mut person = sample_person(
            rng.random(),
            scene.objects.len() + scene.humans.len(),
            chair.position,
            chair.yaw,
            pose,
            Some(chair_id),
            chair.neighbor,
        );
        contextual_gaze(scene, &mut person, &mut rng);
        let fitted = (0..if working_surface && pose != HumanPoseKind::SeatedListening {
            WORKTOP_POSE_ATTEMPTS
        } else {
            1
        })
            .find_map(|proposal| {
                let mut h = person.clone();
                if pose == HumanPoseKind::SeatedWorking
                    || (pose == HumanPoseKind::SeatedTalking && proposal > 0)
                {
                    if let Some(table) = chair.interaction_target.map(|id| &scene.objects[id]) {
                        fit_worktop(&mut h, table, proposal);
                    }
                }
                placement_clear(scene, &h).then_some(h)
            });
        if let Some(person) = fitted {
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
            let mut person = sample_person(
                rng.random(),
                scene.objects.len() + scene.humans.len(),
                p,
                yaw,
                pose,
                None,
                false,
            );
            contextual_gaze(scene, &mut person, &mut rng);
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
    fn furniture_and_continuous_engagement_condition_pose_mixture() {
        let work_fraction = |surface, computer, engagement| {
            let mut rng = stream(17, 0x504f5345);
            (0..8192)
                .filter(|_| {
                    seated_pose(surface, computer, engagement, &mut rng)
                        == HumanPoseKind::SeatedWorking
                })
                .count() as f32
                / 8192.
        };
        assert_eq!(work_fraction(false, false, 1.), 0.);
        let low = work_fraction(true, true, 0.);
        let high = work_fraction(true, true, 1.);
        assert!((0.24..0.32).contains(&low));
        assert!((0.66..0.74).contains(&high));
        assert!(work_fraction(true, false, 1.) < high);
    }

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
