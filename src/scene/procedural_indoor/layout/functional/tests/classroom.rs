use super::*;
use crate::scene::procedural_indoor::{objects::tables, validation::validate_layout};

#[test]
fn classroom_groups_share_worktops_and_keep_writing_places_out_of_the_aisle() {
    let (mut paired, mut single, mut hybrid, mut rows, mut props) = (0, 0, 0, 0, 0);
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Classroom, 0.65, 0, 0.25)
                .unwrap();
        validate_layout(&scene).unwrap();
        let p = scene
            .functional_programs()
            .iter()
            .find(|p| p.activity == IndoorLayout::Classroom)
            .unwrap();
        let zone = || p.members.iter().map(|&i| &scene.objects[i]);
        let tables: Vec<_> = zone()
            .filter(|o| tables::classroom_table(o).is_some())
            .collect();
        rows += usize::from(tables.is_empty());
        hybrid += usize::from(
            !tables.is_empty()
                && zone().any(|o| {
                    o.kind == ObjectKind::Desk || (o.kind == ObjectKind::Chair && o.variant == 10)
                }),
        );
        let axis = (p.min.x + p.max.x) * 0.5;
        for &table in &tables {
            let mate = tables
                .iter()
                .copied()
                .find(|t| tables::classroom_mates(table, t));
            let chairs: Vec<_> = zone()
                .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id))
                .collect();
            if let Some(other) = mate {
                assert!(!chairs.is_empty());
                assert_eq!(table.size, other.size);
                paired += usize::from(table.id < other.id);
            } else {
                assert!(chairs.len() >= 2);
                single += 1;
            }
            for &o in chairs.iter().chain([&table]) {
                let (a, b) = o.bounds();
                assert!(
                    b.x <= axis - p.parameters.aisle_metres * 0.5
                        || a.x >= axis + p.parameters.aisle_metres * 0.5
                );
                assert!(a.z >= p.min.y + 2.64);
            }
            for chair in chairs {
                let direction = (table.position - chair.position).with_y(0.).normalize();
                let forward = Quat::from_rotation_y(chair.yaw) * Vec3::NEG_Z;
                assert!(direction.dot(forward) > 0.75);
            }
            for piece in scene.objects.iter().filter(|o| o.support == Some(table.id)) {
                assert!(matches!(
                    piece.kind,
                    ObjectKind::Notebook
                        | ObjectKind::Notepad
                        | ObjectKind::Pencil
                        | ObjectKind::Phone
                        | ObjectKind::WaterBottle
                ));
                assert!(p.members.contains(&piece.id));
                assert!(scene.prop_clear(piece, table, 0.));
                assert!((piece.position.y - table.position.y - table.size.y).abs() < 0.004);
                props += 1;
            }
        }
    }
    assert!(
        paired > 4 && single > 4 && hybrid > 4 && rows > 4 && props > 4,
        "paired={paired} single={single} hybrid={hybrid} rows={rows} props={props}"
    );
}

#[test]
fn classroom_groups_follow_density_and_leave_clear_tables_when_clutter_is_off() {
    use crate::scene::procedural_indoor::items::distribution::ClutterSettings;
    let mut seats = [0; 2];
    for (cohort, density) in [0.15, 0.95].into_iter().enumerate() {
        for seed in 0..64 {
            let room = IndoorManifest::generate_with_clutter(
                seed,
                IndoorLayout::Classroom,
                density,
                0,
                0.,
                ClutterSettings {
                    probability: 0.,
                    ..Default::default()
                },
            )
            .unwrap();
            for table in room
                .objects
                .iter()
                .filter(|o| tables::classroom_table(o).is_some())
            {
                seats[cohort] += room
                    .objects
                    .iter()
                    .filter(|o| {
                        o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id)
                    })
                    .count();
                assert!(!room.objects.iter().any(|o| o.support == Some(table.id)));
            }
        }
    }
    assert!(
        seats[0] > 0 && seats[1] > seats[0],
        "low/high density seats {seats:?}"
    );
}
