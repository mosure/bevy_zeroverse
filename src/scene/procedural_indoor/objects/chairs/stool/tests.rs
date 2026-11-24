use super::*;
use crate::scene::procedural_indoor::{layout::ObjectKind, objects::audit};

#[test]
fn installed_height_follows_the_local_surface_and_replays() {
    for seed in 0..96 {
        let mut chair = audit::fixture(ObjectKind::Chair, seed);
        chair.variant = 6 + (seed % 2) as u32;
        let mut table = audit::fixture(ObjectKind::Table, seed / 4);
        let mut previous = 0.;
        for height in [0.36, 0.75, 0.94, 1.07] {
            table.size.y = height;
            chair.size.y = fitted_envelope(&chair, Some(&table));
            let p = parameters(&chair);
            assert!((0.36..0.835).contains(&p.seat_height_m));
            assert!(
                p.seat_height_m > previous + 0.002,
                "{seed} {height} {previous} {p:?}"
            );
            previous = p.seat_height_m;
            assert_eq!(parameters_at(&chair, 216).seat_height_m, 0.47);
            table.position.y += 1.25;
            chair.position.y += 1.25;
            assert!((fitted_envelope(&chair, Some(&table)) - chair.size.y).abs() < 1e-5);
        }
    }
}

#[test]
fn camera_proposals_respect_stool_height_and_backless_airspace() {
    for seed in 0..96 {
        let mut chair = audit::fixture(ObjectKind::Chair, seed);
        chair.variant = 6 + (seed % 2) as u32;
        chair.size = Vec3::new(0.57, if is_backless(&chair) { 0.72 } else { 1.03 }, 0.54);
        let p = parameters(&chair);
        let mut saw_back = false;
        p.view_boxes(&chair, |center, half| {
            assert!(half.is_finite() && half.cmpgt(Vec3::ZERO).all());
            if center.y + half.y > p.seat_height_m + 0.001 {
                saw_back = true;
                assert!(center.z > 0., "proxy in front of occupant");
            }
        });
        assert_eq!(saw_back, !is_backless(&chair));
    }
}

#[test]
fn stools_at_workstations_and_cafe_tables_keep_their_installed_support() {
    use crate::scene::procedural_indoor::{
        humans,
        layout::{IndoorLayout, IndoorManifest},
    };
    let mut found = [false; 3];
    let mut witnessed_seated = false;
    for seed in [93410, 93456, 93468, 93640] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.75).unwrap();
        humans::validate(&scene).unwrap();
        for chair in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && matches!(o.variant, 6 | 7))
        {
            let Some(table) = chair.interaction_target.map(|id| &scene.objects[id]) else {
                continue;
            };
            assert!((fitted_envelope(chair, Some(table)) - chair.size.y).abs() < 1e-4);
            assert!(same_floor(chair, table));
            let h = parameters(chair).seat_height_m;
            found[if h < 0.43 {
                0
            } else if h > 0.60 {
                2
            } else {
                1
            }] = true;
        }
        if let Some(h) = scene
            .humans
            .iter_mut()
            .find(|h| h.foot_support_height() > 0.)
        {
            witnessed_seated = true;
            let p = h
                .pose_program
                .as_mut()
                .unwrap()
                .stool_support
                .as_mut()
                .unwrap();
            p.rail_top_m += 0.08;
            assert!(humans::validate(&scene)
                .unwrap_err()
                .contains("support dimensions"));
        }
    }
    assert_eq!(found, [true; 3]);
    assert!(witnessed_seated);
}
