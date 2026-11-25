use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    objects::{audit, build_object},
};
#[test]
fn fitted_thermostats_keep_controls_inside_the_case_and_calibrate_the_face() {
    let mut styles = [0; 3];
    let mut displays = [0; 2];
    for seed in 0..160 {
        let mut o = audit::fixture(ObjectKind::Thermostat, seed);
        o.finish_seed = None;
        let p = parameters(&o);
        let a = build_object(&o);
        let check = audit::check(&o, &a).unwrap();
        assert!(check.triangles < 1600, "{} triangles", check.triangles);
        styles[p.style as usize] += 1;
        for ((_, l), g) in &a.parts {
            assert_eq!(
                SemanticLabel::from_label(super::super::super::part_label(l)),
                Some(SemanticLabel::Thermostat)
            );
            for &v in &g.positions {
                let q = Vec3::from_array(v);
                assert!(
                    q.x.abs() <= o.size.x * 0.5 + 1e-6
                        && q.z.abs() <= o.size.z * 0.5 + 1e-6
                        && (-1e-6..=o.size.y + 1e-6).contains(&q.y),
                    "{q:?}/{:?}",
                    o.size
                );
            }
        }
        let face = &a.parts[&(Surface::Ink, p.film.label("thermostat"))];
        if let Some((center, size)) = p.screen(&o) {
            displays[usize::from(p.fahrenheit)] += 1;
            let c = if p.fahrenheit {
                (p.display_value() as f32 - 32.) / 1.8
            } else {
                p.display_value() as f32
            };
            assert!((c - p.setpoint_c).abs() <= 0.501);
            // The whole face is covered exactly once by coplanar cells with
            // outward winding. Their first two atlas cells spell the setpoint.
            let sum = face
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|tri| {
                    let [a, b, c] = tri.map(|i| Vec3::from_array(face.positions[i as usize]));
                    assert!((a.z - center.z).abs() < 1e-6);
                    let area = (b - a).cross(c - a).z * 0.5;
                    assert!(area > 0.);
                    area
                })
                .sum::<f32>();
            assert!((sum - size.x * size.y).abs() < 1e-7);
            for (index, glyph) in [
                Glyph::Number(p.display_value() / 10),
                Glyph::Number(p.display_value() % 10),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(face.uvs[index * 4..index * 4 + 4], glyph.uvs());
            }
        } else {
            let derived = 15. + (0.5 - p.pointer_angle() / 5.) * 15.;
            assert!((derived - p.setpoint_c).abs() < 1e-5);
        }
    }
    assert!(styles.into_iter().chain(displays).all(|n| n >= 12));
}

#[test]
fn thermostats_require_wall_backing_a_freestanding_operator_and_clear_of_heat_sources() {
    let mut accepted = 0;
    let mut omitted = 0;
    for seed in 400..432 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let items = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Thermostat)
            .collect::<Vec<_>>();
        assert!(items.len() <= 1);
        let Some(&o) = items.first() else {
            omitted += 1;
            continue;
        };
        let o = o.clone();
        accepted += 1;
        let structure = scene.camera_obstacles();
        assert!(scene.thermostat_accessible(&o, &structure));
        let mut floating = o.clone();
        floating.position += o.transform().rotation * Vec3::Z * 0.02;
        assert!(!scene.thermostat_accessible(&floating, &structure));
        let door = scene.entry_door.as_ref().unwrap();
        let mut opening = o.clone();
        opening.yaw = PI;
        opening.position.x = door.center.x;
        opening.position.z = door.center.z - super::super::mount_offset(&o).unwrap();
        assert!(!scene.wall_attachment_clear(&opening));
        let facade = crate::scene::procedural_indoor::architecture::interior_glazing::shared_facade(
            &scene,
            crate::scene::procedural_indoor::architecture::interior_glazing::partition_height(
                &scene,
            ),
        );
        for pane in &facade.openings {
            opening.position.x = -(pane.min.x + pane.max.x) * 0.5;
            opening.position.y = (pane.min.y + pane.max.y - opening.size.y) * 0.5;
            assert!(!scene.wall_attachment_clear(&opening));
        }
        let mut obstacle = audit::fixture(ObjectKind::Cabinet, seed);
        obstacle.id = scene.objects.len();
        obstacle.size = Vec3::new(0.24, 1.15, 0.23);
        obstacle.position = standing_point(&o);
        obstacle.position.y = scene.floor_height(obstacle.position.xz());
        scene.objects.push(obstacle);
        assert!(!scene.thermostat_accessible(&o, &structure));
        let other = scene.objects.last_mut().unwrap();
        other.kind = ObjectKind::BaseboardHeater;
        other.position = o.position.with_y(0.0);
        other.size = Vec3::new(0.18, 0.04, 0.04);
        assert!(!scene.thermostat_accessible(&o, &structure));
    }
    assert!(accepted > 3 && omitted > 3);
}
