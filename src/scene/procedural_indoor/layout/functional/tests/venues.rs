use super::*;
use crate::scene::procedural_indoor::{
    appearance::AppearanceSettings,
    materials::Surface,
    objects::{audit, build_object, part_label},
    validation::{validate_geometry, validate_layout},
};
use std::collections::BTreeSet;

/// Independent exhaustive reference, with the original tie order for equal areas.
fn exhaustive_region(scene: &IndoorManifest) -> (Vec2, Vec2) {
    let zone = &scene.program.as_ref().unwrap().zones[0];
    let lo = zone.min + Vec2::splat(0.55);
    let hi = zone.max - Vec2::splat(0.55);
    let footprint = &scene.envelope.as_ref().unwrap().footprint;
    let step = (hi - lo) * 0.075;
    let offsets: Vec<_> = (0..5)
        .flat_map(|x| (0..5).map(move |z| Vec2::new(x as f32, z as f32)))
        .collect();
    let mut best = (0., lo, hi);
    for lower in &offsets {
        for upper in &offsets {
            let min = lo + step * *lower;
            let max = hi - step * *upper;
            let extent = max - min;
            let area = extent.x * extent.y;
            if area > best.0
                && crate::scene::procedural_indoor::envelope::polygon::box_inside(
                    footprint, min, max, 0.35,
                )
            {
                best = (area, min, max);
            }
        }
    }
    (best.1, best.2)
}

#[test]
fn workout_products_fit_their_metric_reservations_and_replay() {
    let mut inclined = 0;
    let mut levels = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut head_shapes = BTreeSet::new();
    for seed in 0..96 {
        for kind in [
            ObjectKind::WeightBench,
            ObjectKind::DumbbellRack,
            ObjectKind::ExerciseMat,
        ] {
            let o = audit::fixture(kind, seed);
            let mesh = build_object(&o);
            let (lo, hi) = mesh.bounds();
            let half = o.size.xz() * 0.5;
            assert!(
                lo.xz().cmpge(-half - Vec2::splat(0.001)).all()
                    && hi.xz().cmple(half + Vec2::splat(0.001)).all()
                    && lo.y.abs() < 0.001
                    && hi.y <= o.size.y + 0.001,
                "{kind:?} {seed}: {lo:?}, {hi:?}, {:?}",
                o.size
            );
            assert!(mesh
                .parts
                .keys()
                .all(|(_, label)| part_label(label) == kind.class_name()));
            if seed < 4 {
                let replay = build_object(&o);
                for (key, g) in &mesh.parts {
                    let repeated = &replay.parts[key];
                    assert_eq!(g.positions, repeated.positions);
                    assert_eq!(g.indices, repeated.indices);
                }
            }
            if kind == ObjectKind::WeightBench {
                let fit = serde_json::to_value(
                    crate::scene::procedural_indoor::objects::public::fitness::parameters(&o),
                )
                .unwrap();
                let angle = fit["tilt"].as_f64().unwrap();
                inclined += usize::from(angle > 0.15);
                assert!((hi.y - o.size.y).abs() < 0.02);
                assert!((0.40..0.48).contains(&fit["seat_y"].as_f64().unwrap()));
            } else if kind == ObjectKind::DumbbellRack {
                let rack = crate::scene::procedural_indoor::objects::public::dumbbells::FreeWeightRack::for_object(&o);
                levels.insert(rack.tiers.len());
                head_shapes.insert(rack.head);
            }
            for (surface, _) in mesh.parts.keys() {
                finishes.insert(*surface);
            }
        }
    }
    assert!(
        (32..83).contains(&inclined)
            && levels.len() == 2
            && head_shapes.len() == 3
            && finishes.len() >= 4
    );
}

#[test]
fn restaurants_and_fitness_rooms_have_usable_stations_routes_and_cameras() {
    let mut paired_booths = 0;
    let mut stocked = 0;
    let mut served = 0;
    let mut mats = 0;
    let mut rubber_floors = 0;
    let mut counter_walls = BTreeSet::new();
    for seed in 0..12 {
        for layout in [IndoorLayout::Restaurant, IndoorLayout::FitnessRoom] {
            let density = [0., 0.65, 1.][seed as usize % 3];
            let scene = IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25)
                .unwrap_or_else(|e| panic!("{layout:?} {seed}: {e}"));
            validate_layout(&scene).unwrap();
            validate_geometry(&scene).unwrap();
            assert!(
                scene.accepts_camera_visibility()
                    && scene.cameras.len() == 3
                    && scene.cameras.iter().all(|c| scene.camera_curve_clear(c))
            );
            let p = &scene.functional_programs()[0];
            assert_eq!((p.min, p.max), exhaustive_region(&scene));
            let objects = || p.members.iter().map(|&i| &scene.objects[i]);
            assert!(!p
                .reservations
                .iter()
                .any(|r| r.role == ReservationRole::Queue));
            if layout == IndoorLayout::FitnessRoom {
                if scene.floor_style == 2 {
                    rubber_floors += 1;
                    let verify_floor = |s: &IndoorManifest| {
                        let f = &s.program.as_ref().unwrap().materials[Surface::Floor as usize];
                        f.validate().unwrap();
                        assert!((0.76..0.93).contains(&f.roughness));
                        let mineral = f.mineral.as_ref().unwrap();
                        assert_eq!((mineral.polish, mineral.vein_strength), (0., 0.));
                        let cells = f.flooring.as_ref().unwrap().cells;
                        assert_eq!(cells[0], cells[1]);
                        assert!((0.439..1.051).contains(&(f.period_m / cells[0] as f32)));
                    };
                    verify_floor(&scene);
                    if rubber_floors == 1 {
                        let mut recolored = scene.clone();
                        recolored
                            .apply_appearance(AppearanceSettings {
                                material_seed: Some(9911),
                                ..Default::default()
                            })
                            .unwrap();
                        verify_floor(&recolored);
                        assert_ne!(
                            scene.program.as_ref().unwrap().materials[Surface::Floor as usize]
                                .color,
                            recolored.program.as_ref().unwrap().materials[Surface::Floor as usize]
                                .color
                        );
                    }
                } else {
                    assert_eq!(scene.floor_style, 0);
                }
                assert_eq!(p.capacity_unit, CapacityUnit::ExerciseStations);
                let stations = objects()
                    .filter(|o| {
                        matches!(
                            o.kind,
                            ObjectKind::WeightBench
                                | ObjectKind::PowerRack
                                | ObjectKind::ExerciseMat
                                | ObjectKind::StationaryBike
                                | ObjectKind::Treadmill
                        )
                    })
                    .count();
                assert_eq!(p.accepted_capacity, stations);
                assert!(
                    stations > 0
                        && p.reservations
                            .iter()
                            .filter(|r| r.role == ReservationRole::EquipmentAccess)
                            .count()
                            > 4 * stations
                );
                mats += objects()
                    .filter(|o| o.kind == ObjectKind::ExerciseMat)
                    .count();
                let (a, b) = (p.reservations[0].min, p.reservations[0].max);
                assert_eq!((a.y, b.y), (p.min.y, p.max.y));
            } else {
                assert_eq!(p.capacity_unit, CapacityUnit::SeatingStations);
                assert!(p.accepted_capacity >= 2);
                let counter = objects()
                    .find(|o| o.kind == ObjectKind::ServiceCounter)
                    .unwrap();
                let staff = p
                    .reservations
                    .iter()
                    .find(|r| r.role == ReservationRole::StaffAccess)
                    .unwrap();
                let middle = (staff.min + staff.max) * 0.5;
                let local = counter
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(Vec3::new(middle.x, 0., middle.y));
                assert!(
                    local.x.abs() < 0.001 && (local.z + counter.size.z * 0.5 + 0.55).abs() < 0.001
                );
                counter_walls.insert((counter.yaw / std::f32::consts::FRAC_PI_2).round() as i8);
                for table in objects().filter(|o| o.kind == ObjectKind::Table) {
                    assert!(objects().any(|seat| matches!(
                        seat.kind,
                        ObjectKind::Chair | ObjectKind::Banquette
                    ) && seat.interaction_target == Some(table.id)));
                }
                stocked += objects()
                    .filter(|o| o.kind == ObjectKind::Dishware && o.support == Some(counter.id))
                    .count();
                paired_booths += objects()
                    .filter(|o| o.kind == ObjectKind::Banquette && o.interaction_target.is_some())
                    .count();
                served += scene
                    .objects
                    .iter()
                    .filter(|o| o.kind == ObjectKind::Dishware && o.interaction_target.is_some())
                    .count();
            }
        }
    }
    assert!(
        mats > 3 && stocked > 8 && served > 0 && paired_booths > 0,
        "mats={mats} stock={stocked} served={served} booths={paired_booths}"
    );
    assert!((1..12).contains(&rubber_floors));
    assert!(counter_walls.len() >= 3);
}

#[test]
fn waiter_counter_and_staff_fit_the_narrow_mixed_restaurant() {
    for density in [0., 0.65, 1.] {
        let scene = IndoorManifest::generate_with_profiles(
            69_531_153,
            IndoorLayout::Mixed,
            density,
            3,
            0.5,
            &IndoorLayout::PROFILES[..22],
            Default::default(),
        )
        .unwrap();
        assert_eq!(scene.layout, IndoorLayout::Restaurant);
        validate_layout(&scene).unwrap();
        validate_geometry(&scene).unwrap();
        assert!(scene.accepts_camera_visibility());
    }
}

#[test]
fn strength_rack_dense_rooms_keep_the_required_dumbbell_station_accessible() {
    for seed in [68_968, 68_979] {
        let room =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::FitnessRoom, 0.93, 0, 0.)
                .unwrap();
        validate_layout(&room).unwrap();
        validate_geometry(&room).unwrap();
        let program = &room.functional_programs()[0];
        let shelf = program
            .members
            .iter()
            .map(|&id| &room.objects[id])
            .find(|o| o.kind == ObjectKind::DumbbellRack)
            .unwrap();
        let access = shelf
            .transform()
            .transform_point(Vec3::Z * (shelf.size.z * 0.5 + 0.79))
            .xz();
        assert!(program
            .reservations
            .iter()
            .any(|r| r.role == ReservationRole::EquipmentAccess
                && access.cmpge(r.min).all()
                && access.cmple(r.max).all()));
    }
}

#[test]
fn strength_rack_admission_keeps_both_loading_ends_and_the_front_clear() {
    let mut banks = BTreeSet::new();
    let mut headings = BTreeSet::new();
    let mut installed = 0;
    let mut absent = 0;
    for seed in 68700..68772 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::FitnessRoom, 0.65, 0, 0.)
                .unwrap();
        validate_layout(&scene).unwrap();
        let p = &scene.functional_programs()[0];
        let Some(o) = p
            .members
            .iter()
            .map(|&i| &scene.objects[i])
            .find(|o| o.kind == ObjectKind::PowerRack)
        else {
            absent += 1;
            continue;
        };
        installed += 1;
        let spine = &p.reservations[0];
        let midpoint = (spine.min.x + spine.max.x) * 0.5;
        banks.insert(o.position.x < midpoint);
        let side = o.yaw.sin().abs() > 0.7;
        headings.insert(side);
        if side {
            assert!((midpoint - o.position.x) * o.yaw.sin() > 0.);
        } else {
            assert!(((p.min.y + p.max.y) * 0.5 - o.position.z) * o.yaw.cos() > 0.);
        }
        for side in [-1., 1.] {
            let x = side * (o.size.x * 0.5 + 0.70);
            for z in [-o.size.z * 0.5, 0., o.size.z * 0.5] {
                let at = o.transform().transform_point(Vec3::new(x, 0., z)).xz();
                assert!(p
                    .reservations
                    .iter()
                    .any(|r| r.role == ReservationRole::EquipmentAccess
                        && at.cmpge(r.min).all()
                        && at.cmple(r.max).all()));
            }
        }
        if installed == 1 {
            let mut broken = scene.clone();
            broken.assembly_diagnostics.programs[0]
                .reservations
                .retain(|r| r.role != ReservationRole::EquipmentAccess);
            assert!(broken
                .validate_public_programs()
                .unwrap_err()
                .contains("lifting rack lacks"));
        }
    }
    assert!(
        installed > 8 && absent > 8 && banks.len() == 2 && headings.len() == 2,
        "{installed} installed, {absent} absent, sides {banks:?}"
    );
}
