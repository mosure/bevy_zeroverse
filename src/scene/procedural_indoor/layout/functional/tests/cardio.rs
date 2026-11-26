use super::*;
use crate::scene::procedural_indoor::{
    materials,
    objects::{audit, build_object, part_label, public::cardio},
    validation::{validate_geometry, validate_layout},
};
use std::collections::BTreeSet;

#[test]
fn cardio_products_replay_their_mechanical_geometry_and_instrument_materials() {
    let mut drives = BTreeSet::new();
    let mut saw_console_free = false;
    let mut films = BTreeSet::new();
    let mut phases = BTreeSet::new();
    for kind in [ObjectKind::StationaryBike, ObjectKind::Treadmill] {
        for seed in 0..80 {
            let mut o = audit::fixture(kind, seed);
            // The catalog fixture freezes finishes. Sample independent installed fleets here.
            o.finish_seed = Some(seed.wrapping_add(301));
            let m = build_object(&o);
            let check = audit::check(&o, &m).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            assert!(check.min_y_m.abs() < 0.001 && check.max_envelope_overrun_m < 0.001);
            assert!(check.triangles < 6600 && check.batches <= 9, "{check:?}");
            let replay = (seed < 3).then(|| build_object(&o));
            let mut mounted = false;
            for ((surface, name), g) in &m.parts {
                assert_eq!(part_label(name), "fitness_equipment");
                if let Some(slot) = materials::finish_slot(*surface, name) {
                    assert!(materials::registered_finish(*surface, slot));
                    if *surface == materials::Surface::Ink {
                        mounted = true;
                        assert_eq!(g.indices.len(), 30);
                        films.insert(slot);
                    }
                }
                if let Some(other) = &replay {
                    let again = &other.parts[&(*surface, name.clone())];
                    assert_eq!(g.positions, again.positions);
                    assert_eq!(g.uvs, again.uvs);
                    assert_eq!(g.indices, again.indices);
                }
            }
            assert_eq!(mounted, cardio::display_film(&o).is_some());
            if kind == ObjectKind::StationaryBike {
                let product = serde_json::to_value(cardio::parameters(&o)).unwrap();
                let fit = &product["Cycle"];
                saw_console_free |= !mounted;
                drives.insert(fit["rear_drive"].as_bool().unwrap());
                phases.insert((fit["crank_angle"].as_f64().unwrap() * 8.) as i32);
            }
        }
    }
    assert!(saw_console_free);
    assert_eq!(drives.len(), 2);
    assert_eq!(films.len(), 4);
    assert!(phases.len() > 24);
}

#[test]
fn mixed_fitness_preserves_clear_floor_at_the_exit_of_each_moving_belt() {
    let mut orientations = BTreeSet::new();
    let mut strength_only = 0;
    let mut treadmills = 0;
    for seed in 54100..54164 {
        let s =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::FitnessRoom, 0.65, 0, 0.25)
                .unwrap();
        validate_layout(&s).unwrap();
        let p = &s.functional_programs()[0];
        let stations = || p.members.iter().map(|&id| &s.objects[id]);
        strength_only += usize::from(
            stations()
                .all(|o| !matches!(o.kind, ObjectKind::StationaryBike | ObjectKind::Treadmill)),
        );
        for treadmill in stations().filter(|o| o.kind == ObjectKind::Treadmill) {
            orientations.insert(treadmill.yaw.cos().is_sign_positive());
            treadmills += 1;
            let (lo, hi) = treadmill.bounds();
            let half = treadmill.size.xz() * 0.5;
            // Independent probes in the installed machine frame including its
            // far rear corners, both lateral step-off areas and overhead room.
            let rear = [-half.x - 0.49, 0., half.x + 0.49].map(|x| Vec2::new(x, -half.y - 1.99));
            for local in rear
                .into_iter()
                .chain([Vec2::new(-half.x - 0.49, 0.), Vec2::new(half.x + 0.49, 0.)])
            {
                let world = treadmill
                    .transform()
                    .transform_point(Vec3::new(local.x, 0., local.y))
                    .xz();
                assert!(world.cmplt(lo.xz()).any() || world.cmpgt(hi.xz()).any());
                assert!(
                    p.reservations
                        .iter()
                        .any(|r| r.role == ReservationRole::EquipmentAccess
                            && world.cmpge(r.min).all()
                            && world.cmple(r.max).all()),
                    "seed {seed} at {local}"
                );
                assert!(s.ceiling_height(world) > lo.y + 2.35);
            }
        }
    }
    assert!(treadmills > 10 && strength_only > 5 && strength_only < 52);
    assert_eq!(orientations.len(), 2);
}

#[test]
fn cycling_classes_face_an_admitted_instructor_share_a_fleet_and_leave_an_aisle() {
    let mut directions = BTreeSet::new();
    let mut capacities = BTreeSet::new();
    let mut headings = BTreeSet::new();
    for seed in 54300..54332 {
        let density = [0., 0.65, 1.][(seed % 3) as usize];
        let s = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::CyclingStudio,
            density,
            3,
            0.25,
        )
        .unwrap();
        validate_layout(&s).unwrap();
        if seed < 54305 {
            validate_geometry(&s).unwrap();
        }
        assert!(s.accepts_camera_visibility() && s.cameras.iter().all(|c| s.camera_curve_clear(c)));
        let p = &s.functional_programs()[0];
        assert_eq!(p.capacity_unit, CapacityUnit::ExerciseStations);
        let coach = p.cycling_instructor(&s.objects).unwrap();
        assert!(p.accepted_capacity >= 2 && p.accepted_capacity <= p.requested_capacity);
        capacities.insert(p.accepted_capacity);
        directions.insert(coach.yaw.cos().is_sign_positive());
        let aisle = p
            .reservations
            .iter()
            .find(|r| r.role == ReservationRole::Circulation)
            .unwrap();
        assert_eq!((aisle.min.y, aisle.max.y), (p.min.y, p.max.y));
        let bikes: Vec<_> = p
            .members
            .iter()
            .map(|&id| &s.objects[id])
            .filter(|o| o.kind == ObjectKind::StationaryBike)
            .collect();
        assert_eq!(bikes.len(), p.accepted_capacity);
        for o in bikes {
            assert_eq!(o.size.xz(), coach.size.xz());
            assert_eq!(o.finish_seed, coach.finish_seed);
            if o.id == coach.id {
                continue;
            }
            let to_coach = (coach.position - o.position).xz().normalize();
            assert!(to_coach.dot(Vec2::new(o.yaw.sin(), o.yaw.cos())) > 0.55);
            headings.insert((o.yaw * 50.).round() as i32);
        }
        if seed == 54300 {
            let mut invalid = s.clone();
            invalid.assembly_diagnostics.programs[0].focus.x += 1.;
            assert!(invalid
                .validate_public_programs()
                .unwrap_err()
                .contains("instructor"));
        }
    }
    assert_eq!(directions.len(), 2);
    assert!(capacities.len() > 5 && headings.len() > 16);
}
