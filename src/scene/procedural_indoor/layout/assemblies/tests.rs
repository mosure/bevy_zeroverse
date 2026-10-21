use super::*;
use crate::scene::procedural_indoor::{objects::tables, validation};

#[test]
fn meeting_perimeters_cover_curved_ends_and_scale_to_large_boards() {
    let scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let mut o = scene
        .objects
        .iter()
        .find(|o| o.kind.is_surface())
        .unwrap()
        .clone();
    o.kind = ObjectKind::Table;
    for seed in 0..96 {
        for size in [
            Vec3::new(1.4, 0.75, 1.4),
            Vec3::new(1.5, 0.75, 3.5),
            Vec3::new(2.1, 0.75, 8.5),
        ] {
            o.seed = seed;
            o.size = size;
            let outline = tables::outline(&o);
            let path = perimeter::SeatPath::new(&o, 0.52);
            assert!(path.length.is_finite());
            let mut quadrants = [0; 4];
            for i in 0..96 {
                let (point, yaw) = path.sample(i as f32 / 96.);
                assert!(point.is_finite() && yaw.is_finite());
                quadrants[usize::from(point.x > 0.) + 2 * usize::from(point.y > 0.)] += 1;
                let distance = outline
                    .iter()
                    .enumerate()
                    .map(|(j, &a)| {
                        let edge = outline[(j + 1) % outline.len()] - a;
                        let q = a + edge
                            * ((point - a).dot(edge) / edge.length_squared()).clamp(0., 1.);
                        point.distance(q)
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    (0.519..0.68).contains(&distance),
                    "{seed}/{size}: {distance}"
                );
                let outward = Vec2::new(yaw.sin(), yaw.cos());
                assert!(point.dot(outward) > 0., "chair faces away from the table");
            }
            assert!(quadrants.into_iter().all(|n| n > 8));
        }
    }
}

#[test]
fn conference_population_includes_long_boards_and_many_real_clear_seats() {
    let mut meetings = 0;
    let mut large = 0;
    let mut max_seats = 0;
    let mut max_length = 0_f32;
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.85, 0, 0.)
                .unwrap();
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        for group in &scene.assembly_diagnostics.accepted {
            if group.role != AssemblyRole::Meeting {
                continue;
            }
            meetings += 1;
            let table = &scene.objects[group.surface];
            max_length = max_length.max(table.size.x.max(table.size.z));
            max_seats = max_seats.max(group.seats.len());
            if table.size.x.max(table.size.z) > 4.5 && group.seats.len() >= 10 {
                large += 1;
            }
            assert!(group.seats.len() >= (group.requested_capacity as f32 * 0.65).ceil() as usize);
            for &id in &group.seats {
                let chair = &scene.objects[id];
                assert_eq!(chair.interaction_target, Some(table.id));
                assert!(
                    scene.placement_clear(chair, 0.008),
                    "{seed}/{id}: seat must remain clear after tucking"
                );
            }
        }
    }
    eprintln!("conference cohort: {meetings} meetings, {large} large with >=10 seats; max {max_length:.2} m / {max_seats} seats");
    assert!(meetings >= 35 && large >= 4 && max_length > 6. && max_seats >= 14);
}

/// Measure relative to the rendered tabletop's nearest edge, not room axes or
/// the table centre (which misclassifies seats along a long table's sides).
fn edge_relative_yaw(table: &IndoorObject, chair: &IndoorObject) -> f32 {
    let p = table
        .transform()
        .compute_affine()
        .inverse()
        .transform_point3(chair.position)
        .xz();
    let outline = tables::outline(table);
    let (_, normal) = outline
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let edge = outline[(i + 1) % outline.len()] - a;
            let nearest = a + edge * ((p - a).dot(edge) / edge.length_squared()).clamp(0., 1.);
            (p.distance_squared(nearest), -edge.normalize().perp())
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    let delta = chair.yaw - table.yaw - normal.x.atan2(normal.y);
    delta.sin().atan2(delta.cos()).to_degrees()
}

#[test]
fn conference_swivel_survives_tucking_with_clear_occupied_and_empty_seats() {
    let mut cohorts = [Vec::<f32>::new(), Vec::new(), Vec::new()];
    let mut occupied = 0;
    let mut paired_turns = 0;
    let mut varied_pairs = 0;
    for human_density in [0., 0.7] {
        for seed in 0..128 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                IndoorLayout::Conference,
                0.85,
                0,
                human_density,
            )
            .unwrap();
            validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
            if seed % 32 == 0 {
                let replay = IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::Conference,
                    0.85,
                    0,
                    human_density,
                )
                .unwrap();
                assert_eq!(scene, replay, "seating must replay deterministically");
            }
            for group in &scene.assembly_diagnostics.accepted {
                if group.role != AssemblyRole::Meeting {
                    continue;
                }
                let table = &scene.objects[group.surface];
                let large = table.size.x.max(table.size.z) > 4.5;
                let mut angles = Vec::new();
                for &id in &group.seats {
                    let chair = &scene.objects[id];
                    let angle = edge_relative_yaw(table, chair);
                    cohorts[0].push(angle);
                    if large {
                        cohorts[1].push(angle);
                    }
                    angles.push(angle);
                    assert!(scene.placement_clear(chair, 0.008), "{seed}/{id}");
                    if let Some(person) = scene.humans.iter().find(|h| h.chair == Some(id)) {
                        occupied += 1;
                        cohorts[2].push(angle);
                        assert_eq!(person.position, chair.position);
                        assert_eq!(person.yaw, chair.yaw);
                        assert!(crate::scene::procedural_indoor::humans::placement_clear(
                            &scene, person
                        ));
                    }
                }
                // Neighbouring seats must vary independently, rather than
                // sharing one room/table rotation that looks diverse in world space.
                for pair in angles.windows(2) {
                    paired_turns += 1;
                    varied_pairs += usize::from((pair[0] - pair[1]).abs() > 10.);
                }
            }
        }
    }
    for (name, angles) in ["all", "large tables", "occupied"].into_iter().zip(cohorts) {
        let n = angles.len();
        let turned = angles.iter().filter(|v| v.abs() > 10.).count();
        let broad = angles.iter().filter(|v| v.abs() > 30.).count();
        let left = angles.iter().filter(|&&v| v < -10.).count();
        let right = angles.iter().filter(|&&v| v > 10.).count();
        eprintln!("conference swivel {name}: {n} seats, >10deg={turned}, >30deg={broad}, left={left}, right={right}");
        assert!(n >= 20, "insufficient {name} coverage: {n}");
        assert!(turned * 3 > n, "tucking erased {name} swivel: {turned}/{n}");
        assert!(broad * 20 > n, "{name} lost broad swivel: {broad}/{n}");
        assert!(left * 10 > n && right * 10 > n, "{name} directional bias");
    }
    assert!(occupied >= 20);
    assert!(
        varied_pairs * 2 > paired_turns,
        "chairs share one facing offset"
    );
}
