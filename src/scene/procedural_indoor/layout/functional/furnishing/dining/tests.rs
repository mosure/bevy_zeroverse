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
