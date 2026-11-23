use super::*;
use crate::scene::procedural_indoor::{objects::audit::fixture, validation};

#[test]
fn audience_rejects_back_faces_blocked_sightlines_and_the_back_of_a_classroom() {
    let mut room =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    room.envelope = None;
    let mut tv = fixture(ObjectKind::Display, 6);
    tv.id = 18;
    tv.position = Vec3::new(0., 1.25, 0.);
    tv.yaw = 0.;
    tv.size = Vec3::new(1.51, 0.85, 0.10);
    let mut space = MountSpace {
        structure: vec![],
        objects: vec![],
    };
    let mut audience = Audience {
        eyes: vec![(Some(2), Vec3::new(0., 1.22, 4.))],
        teaching_front: None,
    };
    assert!(audience.score(&room, &tv, &space, None).is_some());
    tv.yaw = std::f32::consts::PI;
    assert!(audience.score(&room, &tv, &space, None).is_none());
    tv.yaw = 0.;
    let blocker = (Vec3::new(-1., 0., 1.), Vec3::new(1., 2., 1.5));
    space.structure.push(blocker);
    assert!(audience.score(&room, &tv, &space, None).is_none());
    space.structure.clear();
    space.objects.push((5, blocker));
    assert!(audience.score(&room, &tv, &space, None).is_none());
    space.objects.clear();
    audience.teaching_front = Some(Vec2::new(0., 6.));
    assert!(audience.score(&room, &tv, &space, None).is_none());
}

#[test]
fn office_teaching_and_service_programs_have_distinct_nonexclusive_tv_demand() {
    let mut room =
        IndoorManifest::generate_with_humans(202, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let expected = [
        (IndoorLayout::Kitchen, 0.15),
        (IndoorLayout::OpenOffice, 0.80),
        (IndoorLayout::Coworking, 0.80),
        (IndoorLayout::Conference, 0.89),
        (IndoorLayout::LectureHall, 0.92),
    ];
    for (layout, expected) in expected {
        room.layout = layout;
        let present = (0..2048)
            .filter(|seed| {
                room.seed = *seed;
                room.display_demand() > 0
            })
            .count();
        assert!(
            (present as f32 / 2048. - expected).abs() < 0.04,
            "{layout:?}: {present}"
        );
    }
}

#[test]
fn generated_wall_and_pillar_mounts_replay_and_reject_detached_or_lowered_sets() {
    let mut column_tv = 0;
    let mut wall_tv = 0;
    for seed in 50800..50864 {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.65, 0, 0.)
                .unwrap();
        let region = room.primary_region();
        for tv in room
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Display)
        {
            assert!(room.wall_attachment_clear(tv));
            if let Some(index) = room.display_pillar(tv) {
                column_tv += 1;
                assert!(region.contains_projection(tv.position.xz(), 0.));
                assert!(room.display_pillar_clear(tv, index));
                let back = tv.transform().rotation * Vec3::Z;
                let mut floating = tv.clone();
                floating.position += back * 0.035;
                assert!(!room.wall_attachment_clear(&floating));
                let mut lowered = tv.clone();
                lowered.position.y = room.floor_height(tv.position.xz()) + 1.4;
                assert!(!room.display_pillar_clear(&lowered, index));
                if column_tv == 1 {
                    validation::validate_geometry(&room).unwrap();
                    let replay: IndoorManifest =
                        serde_json::from_slice(&serde_json::to_vec(&room).unwrap()).unwrap();
                    assert_eq!(replay.display_pillar(&replay.objects[tv.id]), Some(index));
                }
            } else {
                wall_tv += 1;
            }
        }
    }
    assert!(
        wall_tv > 12 && column_tv > 0,
        "wall {wall_tv}, pillar {column_tv}"
    );
}

#[test]
fn interior_tv_needs_an_opaque_span_on_the_correct_side_of_the_door() {
    use crate::scene::procedural_indoor::program::Partition;
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    scene.envelope = None;
    let program = scene.program.as_mut().unwrap();
    program.zones.clear();
    program.primary_zone = None;
    program.partitions = vec![Partition {
        axis: 0,
        coordinate: 0.,
        start: -3.5,
        end: 3.5,
        door_center: 0.,
        door_width: 1.2,
        door_height: 2.1,
        arch_rise: 0.,
        door: None,
        thickness: 0.14,
        sill: 0.85,
        glazing_fraction: 0.,
        mullion_pitch: 1.4,
        transom_fraction: None,
    }];
    let mut tv = fixture(ObjectKind::Display, 2);
    tv.support = None;
    tv.solid = false;
    tv.size = Vec3::new(1.2, 0.68, 0.12);
    tv.position = Vec3::new(0.131, 1.2, 2.05);
    tv.yaw = std::f32::consts::FRAC_PI_2;
    assert_eq!(
        scene.television_mount(&tv),
        Some(TelevisionMount::Partition(0))
    );
    let mut detached = tv.clone();
    detached.position.x += 0.025;
    assert_eq!(scene.television_mount(&detached), None);
    let mut doorway = tv.clone();
    doorway.position.z = 0.;
    assert_eq!(scene.television_mount(&doorway), None);
    tv.position.x = -tv.position.x;
    tv.yaw = -tv.yaw;
    assert_eq!(
        scene.television_mount(&tv),
        Some(TelevisionMount::Partition(0))
    );
    scene.program.as_mut().unwrap().partitions[0].glazing_fraction = 0.75;
    assert_eq!(scene.television_mount(&tv), None);
}

#[test]
fn presentation_groups_respect_unequipped_rooms_in_both_wall_pass_orders() {
    let mut orders = [0; 2];
    for seed in 0..48 {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
        if room.display_demand() != 0 {
            continue;
        }
        assert!(
            !room.objects.iter().any(|o| o.kind == ObjectKind::Display),
            "seed {seed}"
        );
        // Reapply the two passes directly, in alternating order, to a room with
        // cleared decor. A later mixed-media group cannot install a surprise TV.
        let mut replay = room;
        replay.objects.clear();
        let order = (seed % 2) as usize;
        if order == 0 {
            replay.compose_wall_groups();
            replay.furnish_displays();
        } else {
            replay.furnish_displays();
            replay.compose_wall_groups();
        }
        assert!(!replay.objects.iter().any(|o| o.kind == ObjectKind::Display));
        orders[order] += 1;
    }
    assert!(orders.into_iter().all(|n| n > 8));
}
