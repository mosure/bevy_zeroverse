use super::*;
use crate::scene::procedural_indoor::{
    items::distribution::ClutterSettings,
    layout::programs::access,
    objects::public::furnishing::{lectern, parameters},
    validation::validate_layout,
};

#[test]
fn presenters_read_from_the_level_side_facing_the_actual_seats() {
    for activity in [IndoorLayout::Classroom, IndoorLayout::LectureHall] {
        let mut flanks = [0; 2];
        let mut notes = 0;
        for seed in 0..64 {
            let room = IndoorManifest::generate_with_humans(seed, activity, 0.65, 0, 0.).unwrap();
            validate_layout(&room).unwrap();
            let p = &room.functional_programs()[0];
            let members = || p.members.iter().map(|&id| &room.objects[id]);
            let desk = members().find(|o| o.kind == ObjectKind::Lectern).unwrap();
            let spot = lectern::presenter_position(desk);
            let floor = room.floor_height(spot.xz());
            assert!((floor - desk.position.y).abs() < 0.03);
            let (lo, hi) = access::envelope(desk).unwrap().bounds();
            assert!(lo.xz().cmpge(p.min).all() && hi.xz().cmple(p.max).all());
            let chairs: Vec<_> = members().filter(|o| o.kind == ObjectKind::Chair).collect();
            let center = chairs.iter().map(|o| o.position).sum::<Vec3>() / chairs.len() as f32;
            let forward = desk.transform().rotation * Vec3::NEG_Z;
            assert!(
                (center - spot).with_y(0.).normalize().dot(forward) > 0.55,
                "{activity:?}/{seed}"
            );
            assert!(chairs.iter().all(|c| !access::blocks(desk, c)));
            flanks[usize::from(desk.position.x > (p.min.x + p.max.x) * 0.5)] += 1;
            notes += usize::from(parameters(desk).reading_desk.unwrap().notes.is_some());
        }
        assert!(flanks.into_iter().all(|n| n > 10) && (3..55).contains(&notes));
    }
}

#[test]
fn reversed_presenter_or_obstructed_standing_space_fails_validation() {
    let mut room = IndoorManifest::generate_with_clutter(
        76200,
        IndoorLayout::LectureHall,
        0.65,
        0,
        0.,
        ClutterSettings {
            probability: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let id = room
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::Lectern)
        .unwrap()
        .id;
    assert!(parameters(&room.objects[id])
        .reading_desk
        .unwrap()
        .notes
        .is_none());
    room.objects[id].yaw += std::f32::consts::PI;
    assert!(room
        .validate_public_programs()
        .unwrap_err()
        .contains("presenter"));
    room.objects[id].yaw -= std::f32::consts::PI;
    let spot = lectern::presenter_position(&room.objects[id]);
    let seat = room
        .objects
        .iter_mut()
        .find(|o| o.kind == ObjectKind::Chair)
        .unwrap();
    seat.position = spot;
    assert!(!room.placement_clear(&room.objects[id], 0.));
}

#[test]
fn presenter_in_occupied_classroom_keeps_its_place_after_final_furnishing() {
    let room = IndoorManifest::generate_with_humans(76243, IndoorLayout::Classroom, 0.65, 3, 0.35)
        .unwrap();
    let p = &room.functional_programs()[0];
    let desk = p
        .members
        .iter()
        .map(|&id| &room.objects[id])
        .find(|o| o.kind == ObjectKind::Lectern)
        .unwrap();
    let (lo, hi) = access::bounds(desk);
    for route in &p.reservations {
        assert!(!route.overlaps(lo, hi));
    }
    validate_layout(&room).unwrap();
}

#[test]
fn presenters_can_use_the_reading_side_with_varied_collision_checked_poses() {
    use crate::scene::procedural_indoor::humans::{placement_clear, HumanPoseKind};
    let mut positions = std::collections::BTreeSet::new();
    let mut elsewhere = 0;
    for seed in 600..696 {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Classroom, 0.65, 0, 0.85)
                .unwrap();
        let desk = room.objects.iter().find(|o| lectern::installed(o)).unwrap();
        let local = desk.transform().compute_affine().inverse();
        for h in room
            .humans
            .iter()
            .filter(|h| !h.neighbor && h.pose == HumanPoseKind::StandingPresenting)
        {
            let rel = local.transform_point3(h.position);
            let from_edge = (rel.z - desk.size.z * 0.5) / lectern::standing_depth(desk);
            let angle = (h.yaw - desk.yaw).cos();
            if rel.x.abs() <= 0.16 * desk.size.x + 1e-4
                && (0.429..0.701).contains(&from_edge)
                && angle > 0.97
            {
                assert!(placement_clear(&room, h));
                positions.insert((
                    (rel.x * 1000.).round() as i32,
                    (from_edge * 1000.).round() as i32,
                ));
            } else {
                elsewhere += 1;
            }
        }
    }
    assert!(
        positions.len() >= 4 && elsewhere >= 4,
        "{} at desk, {elsewhere} elsewhere",
        positions.len()
    );
}
