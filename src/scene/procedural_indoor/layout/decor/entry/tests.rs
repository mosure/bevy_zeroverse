use super::*;
use crate::scene::procedural_indoor::objects::audit::fixture;

#[test]
fn both_jambs_use_built_wall_backing_and_leave_reachable_floor_space() {
    let mut accepted = 0;
    let mut alternatives = 0;
    let mut sided = [0; 2];
    let mut offsets = std::collections::BTreeSet::new();
    for seed in 0..48 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let door = scene.entry_door.as_ref().unwrap();
        let plate = fixture(ObjectKind::LightSwitch, seed);
        let Some(mounted) = scene.fit_entry_switch(&plate, 1.12, &[]) else {
            continue;
        };
        accepted += 1;
        assert_eq!(
            scene.fit_entry_switch(&plate, 1.12, &[]),
            Some(mounted.clone())
        );
        assert!(scene.wall_attachment_clear(&mounted));
        let dx = mounted.position.x - door.center.x;
        sided[usize::from(dx > 0.)] += 1;
        offsets.insert(((dx.abs() - door.width * 0.5) * 100.).round() as i32);

        // This obstruction is in front of the fixture, leaving the mounting
        // surface itself untouched. Blocking one side may require crossing
        // the doorway. Blocking the entire approach must reject the proposal.
        let z = door.center.z;
        let barrier = |x1, x2| (Vec3::new(x1, 0.05, z - 0.65), Vec3::new(x2, 1.8, z - 0.18));
        let one = if dx > 0. {
            barrier(door.center.x, 100.)
        } else {
            barrier(-100., door.center.x)
        };
        if let Some(alternate) = scene.fit_entry_switch(&plate, 1.12, &[one]) {
            assert!(dx * (alternate.position.x - door.center.x) < 0.);
            alternatives += 1;
        }
        assert!(scene
            .fit_entry_switch(&plate, 1.12, &[barrier(-100., 100.)])
            .is_none());

        let mut invalid = mounted.clone();
        invalid.position.x = door.center.x + door.width * 0.5;
        assert!(!scene.wall_attachment_clear(&invalid), "plate on the jamb");
        let f = interior_glazing::shared_facade(&scene, interior_glazing::partition_height(&scene));
        for window in &f.openings {
            invalid.position.x = -(window.min.x + window.max.x) * 0.5;
            invalid.position.y = (window.min.y + window.max.y - invalid.size.y) * 0.5;
            assert!(!scene.wall_attachment_clear(&invalid), "plate on a pane");
        }
        invalid = mounted.clone();
        invalid.position.z -= 0.012;
        assert!(!scene.wall_attachment_clear(&invalid), "floating plate");
        invalid = mounted;
        invalid.yaw = 0.;
        assert!(!scene.wall_attachment_clear(&invalid), "backward plate");
    }
    assert!(
        accepted >= 20 && alternatives >= 5,
        "accepted={accepted} alternative={alternatives}"
    );
    assert!(
        sided.iter().all(|&n| n >= 6) && offsets.len() >= 12,
        "{sided:?} {offsets:?}"
    );
}
