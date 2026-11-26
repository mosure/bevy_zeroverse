use super::*;
use crate::scene::procedural_indoor::{objects, validation};
use std::collections::BTreeSet;

#[test]
fn dining_station_recovery_preserves_side_level_and_locality() {
    let mut table = objects::audit::fixture(ObjectKind::Table, 0);
    table.position = Vec3::new(2., 0.8, -3.);
    table.yaw = 1.17;
    let station = Vec2::new(0.25, 1.10);
    let mut seat = objects::audit::fixture(ObjectKind::Chair, 0);
    for (offset, accepted) in [
        (Vec3::new(0.25, 0., 1.10), true),
        (Vec3::new(0.40, 0., 1.20), true),
        (Vec3::new(0.25, 0., -1.10), false),
        (Vec3::new(0.80, 0., 1.10), false),
        (Vec3::new(0.25, 0.05, 1.10), false),
    ] {
        seat.position = table.transform().transform_point(offset);
        assert_eq!(station_clear(&table, &seat, station), accepted);
    }
}

#[test]
fn dining_seats_cover_ends_counts_and_both_bench_sides_without_clipping() {
    let mut counts = BTreeSet::new();
    let mut tables_with_end_seats = 0;
    let mut tables_with_side_seats = 0;
    let mut benches = [0; 2];
    for layout in [IndoorLayout::Cafe, IndoorLayout::Cafeteria] {
        for seed in 0..32 {
            let density = [0., 0.35, 0.65, 1.][seed as usize % 4];
            let humans = if seed % 4 == 0 { 0.8 } else { 0. };
            let scene = IndoorManifest::generate_with_humans(seed, layout, density, 0, humans)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            scene.validate_public_programs().unwrap();
            validation::validate_layout(&scene).unwrap();
            // Bench fronts on opposing sides face in opposite directions. The
            // separate wall bench is perpendicular to the table collection.
            let collection_yaw = scene.functional_programs()[0]
                .parameters
                .group_rotation_radians;
            for bench in scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && o.kind == ObjectKind::Banquette)
            {
                let direction = (bench.yaw - collection_yaw).cos();
                if direction.abs() > 0.8 {
                    benches[usize::from(direction < 0.)] += 1;
                }
            }
            for table in scene.objects.iter().filter(|o| {
                !o.neighbor
                    && o.kind == ObjectKind::Table
                    && o.variant != objects::tables::SHARED_ROW_VARIANT
            }) {
                let inv = table.transform().compute_affine().inverse();
                let mut seats = 0;
                let mut has_end = false;
                let mut has_side = false;
                for seat in scene.objects.iter().filter(|o| {
                    o.interaction_target == Some(table.id) && o.kind == ObjectKind::Chair
                }) {
                    assert!(
                        scene.placement_clear(seat, 0.008),
                        "{layout:?}/{seed}/{}",
                        seat.id
                    );
                    assert!((seat.position.y - table.position.y).abs() < 0.015);
                    let local = inv.transform_point3(seat.position);
                    seats += 1;
                    let normalized = local / table.size;
                    has_end |= normalized.x.abs() > normalized.z.abs();
                    has_side |= normalized.x.abs() <= normalized.z.abs();
                    let facing = inv
                        .transform_vector3(seat.transform().rotation * -Vec3::Z)
                        .xz();
                    assert!(
                        facing.dot(-local.xz().normalize()) > 0.25,
                        "{layout:?}/{seed}/{}: chair faces away from table",
                        seat.id
                    );
                    let mesh = objects::build_object(seat);
                    objects::audit::check(seat, &mesh).unwrap();
                }
                counts.insert(seats);
                tables_with_end_seats += usize::from(has_end);
                tables_with_side_seats += usize::from(has_side);
            }
            if seed < 3 {
                validation::validate_geometry(&scene).unwrap();
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(seed, layout, density, 0, humans).unwrap()
                );
                let decoded: IndoorManifest =
                    serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
                assert_eq!(scene, decoded);
            }
        }
    }
    eprintln!("dining counts={counts:?}; end tables={tables_with_end_seats}; long-side tables={tables_with_side_seats}; bench sides={benches:?}");
    assert!(counts.contains(&2) && counts.iter().any(|&n| n >= 3));
    assert!(tables_with_end_seats >= 15 && tables_with_side_seats >= 15);
    assert!(
        benches.into_iter().all(|n| n >= 4),
        "bench placement must not prefer a fixed side"
    );
}

#[test]
fn dining_groups_follow_the_actual_service_frontage_and_cross_aisle() {
    let scene = IndoorManifest::generate_with_humans(9, IndoorLayout::Cafe, 0.65, 0, 0.).unwrap();
    let mut p = scene.functional_programs()[0].clone();
    p.min = Vec2::new(-7.0, -6.0);
    p.max = Vec2::new(9.0, 8.0);
    let route = |role, min, max| Reservation { role, min, max };
    // The order and positions differ from a centred standalone café.
    p.reservations = vec![
        route(
            ReservationRole::Queue,
            Vec2::new(-5., -4.),
            Vec2::new(-2., -1.),
        ),
        route(ReservationRole::Circulation, Vec2::new(-7., 6.4), p.max),
        route(
            ReservationRole::Circulation,
            Vec2::new(-0.4, -3.),
            Vec2::new(1.2, 8.),
        ),
    ];
    let mut counter = objects::audit::fixture(ObjectKind::ServiceCounter, 2);
    counter.position = Vec3::new(3., 0., -4.5);
    counter.size = Vec3::new(2.8, 1., 1.);
    counter.yaw = 0.;
    let bounds = bays::dining_bays(&p, std::slice::from_ref(&counter)).unwrap();
    assert_eq!(bounds[0], Rect::new(-7., -1., -0.4, 6.4));
    assert_eq!(bounds[1], Rect::new(1.2, -4., 9., 6.4));
    let mut across = [false; 3];
    let mut headings = [0; 2];
    let mut benches = [0; 2];
    let mut both_banks = 0;
    let mut table_counts = BTreeSet::new();
    for seed in 0..96 {
        let plan = bays::DiningPlan::sample(&p, std::slice::from_ref(&counter), seed, false);
        let mut matched_rows = [Vec::new(), Vec::new()];
        assert!(!plan.tables.is_empty() && plan.tables.len() <= 32);
        for site in &plan.tables {
            let side = usize::from(site.position.x > 0.);
            let bay = bounds[side];
            let uv = (site.position.xz() - bay.min) / bay.size();
            assert!(uv.cmpgt(Vec2::ZERO).all() && uv.cmplt(Vec2::ONE).all());
            across[(uv.x * 3.) as usize] = true;
            headings[usize::from(site.yaw.sin().abs() > 0.8)] += 1;
            matched_rows[side].push(site.position.z);
        }
        table_counts.insert(plan.tables.len());
        if matched_rows.iter().all(|v| !v.is_empty()) {
            both_banks += 1;
            assert!(matched_rows[0]
                .iter()
                .any(|z| matched_rows[1].iter().all(|w| (z - w).abs() > 0.001)));
        }
        let bench = plan.perimeter_bench.unwrap();
        let side = usize::from(bench.position.x > 0.);
        let footprint = Vec2::new(bench.size.z, bench.size.x) * 0.5;
        assert!(bounds[side].contains(bench.position.xz() - footprint));
        assert!(bounds[side].contains(bench.position.xz() + footprint));
        // Banquette local +Z faces away from its backrest, into the room.
        assert!(bench.yaw.sin() * bench.position.x < 0.);
        benches[side] += 1;
    }
    assert!(both_banks > 60 && table_counts.len() >= 4);
    assert!(across.into_iter().all(|v| v));
    assert!(headings.into_iter().all(|n| n > 50));
    assert!(benches.into_iter().all(|n| n > 20));
    // A cross aisle that consumes both banks must produce no forced stations.
    p.reservations[1].min.y = p.min.y;
    let empty = bays::DiningPlan::sample(&p, &[counter], 0, false);
    assert!(empty.tables.is_empty() && empty.perimeter_bench.is_none());
}
