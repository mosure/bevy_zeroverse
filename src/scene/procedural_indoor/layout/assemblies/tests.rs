use super::*;
use crate::scene::procedural_indoor::{objects::tables, validation};

#[test]
fn mixed_conference_uses_the_same_boardroom_design_and_archives_read_without_it() {
    use crate::scene::procedural_indoor::program::{boardroom::BoardroomPlan, IndoorProgram};
    let mut compared = 0;
    for seed in 0..1024 {
        let mut mix = stream(seed, 0);
        if IndoorLayout::sample_profile(seed, &IndoorLayout::PROFILES, &mut mix)
            != IndoorLayout::Conference
            || BoardroomPlan::sample(seed, IndoorLayout::Conference).is_none()
        {
            continue;
        }
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.75, 0, 0.).unwrap();
        let explicit =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.75, 0, 0.)
                .unwrap();
        let program = scene.program.as_ref().unwrap();
        assert_eq!(
            program.executive_boardroom,
            explicit.program.unwrap().executive_boardroom
        );
        assert!(scene.executive_boardroom().unwrap().seat_count >= 8);
        let mut archived = serde_json::to_value(program).unwrap();
        archived
            .as_object_mut()
            .unwrap()
            .remove("executive_boardroom");
        assert!(serde_json::from_value::<IndoorProgram>(archived)
            .unwrap()
            .executive_boardroom
            .is_none());
        compared += 1;
        if compared == 3 {
            break;
        }
    }
    assert_eq!(compared, 3);
}

#[test]
fn executive_boardrooms_keep_four_sided_seating_and_circulation_at_all_densities() {
    use crate::scene::procedural_indoor::envelope::polygon;
    let mut counts = [0; 2];
    let mut axes = [0; 2];
    let mut tight_and_spacious = [false; 2];
    for seed in 2100..2164 {
        let density = [0., 0.65, 1.][seed as usize % 3];
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, density, 3, 0.45)
                .unwrap_or_else(|e| panic!("{seed}: {e}"));
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let program = scene.program.as_ref().unwrap();
        let Some(plan) = program.executive_boardroom else {
            counts[0] += 1;
            assert!(scene.executive_boardroom().is_none());
            continue;
        };
        counts[1] += 1;
        axes[usize::from(plan.long_axis_x)] += 1;
        assert_eq!(program.zones.len(), 1);
        assert!(program.partitions.is_empty());
        let e = scene.envelope.as_ref().unwrap();
        assert!(
            polygon::box_inside(
                &e.footprint,
                plan.center() - plan.reserved_half(),
                plan.center() + plan.reserved_half(),
                0.
            ),
            "{seed}: circulation reservation"
        );
        let b = scene.executive_boardroom().unwrap();
        let table = &scene.objects[b.table_id];
        assert!(table.size.xz().abs_diff_eq(plan.table_size_m, 0.001));
        assert_eq!(
            scene
                .objects
                .iter()
                .filter(|o| !o.neighbor && o.kind == ObjectKind::Table)
                .count(),
            1
        );
        assert!(b.seat_count >= 8 && b.tabletop_area_m2 > 0.);
        assert!(b.tabletop_area_m2 < b.seating_envelope_area_m2);
        let occupied = b.seating_envelope_area_m2 / b.room_floor_area_m2;
        assert!((0.20..0.85).contains(&occupied), "{seed}: {occupied}");
        tight_and_spacious[usize::from(occupied > 0.5)] = true;
        let group = scene
            .assembly_diagnostics
            .accepted
            .iter()
            .find(|g| g.surface == table.id)
            .unwrap();
        let inv = table.transform().compute_affine().inverse();
        let mut sides = [0; 4];
        for &s in &group.seats {
            let chair = &scene.objects[s];
            assert_eq!(chair.interaction_target, Some(table.id));
            assert!(
                scene.placement_clear(chair, 0.008),
                "{seed}/{s}: installed chair clearance"
            );
            let p = inv.transform_point3(chair.position);
            // End chairs are sampled on the centre portion of their edge. A long
            // side chair remains laterally separated even after being pushed in.
            let i = if p.x.abs() < table.size.x * 0.26 {
                2 + usize::from(p.z > 0.)
            } else {
                usize::from(p.x > 0.)
            };
            sides[i] += 1;
        }
        assert!(
            sides[0] >= 3 && sides[1] >= 3 && sides[2] >= 1 && sides[3] >= 1,
            "{seed}: {sides:?}"
        );
        assert_eq!(scene.cameras.len(), 3);
    }
    assert!(counts.into_iter().min().unwrap() >= 14 && axes.into_iter().min().unwrap() >= 5);
    assert_eq!(tight_and_spacious, [true, true]);
}

#[test]
fn compact_boardrooms_leave_the_entry_clear_after_seats_are_installed() {
    for seed in [182, 194, 265, 380, 428, 505] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.65, 3, 0.25)
                .unwrap();
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let board = scene.executive_boardroom().unwrap();
        assert!(board.seat_count >= 8);
        for o in scene
            .objects
            .iter()
            .filter(|o| o.interaction_target == Some(board.table_id) && o.kind == ObjectKind::Chair)
        {
            let (lo, hi) = o.bounds();
            assert!(!scene.entry_approach_overlaps(lo, hi), "{seed}/{}", o.id);
        }
    }
}

#[test]
fn flagship_boardroom_retains_a_long_table_and_requires_the_installed_group() {
    use crate::sample::failure::{CaptureError, SceneRejectionKind};
    let mut scene =
        IndoorManifest::generate_with_humans(1057, IndoorLayout::Conference, 0.65, 0, 0.).unwrap();
    let b = scene.executive_boardroom().unwrap();
    assert!(b.table_length_m > 14. && b.table_width_m < 4.4, "{b:?}");
    assert!(
        b.seating_envelope_area_m2 > 0.5 * b.room_floor_area_m2,
        "{b:?}"
    );
    assert!(b.seat_count >= 20);
    scene
        .assembly_diagnostics
        .accepted
        .retain(|g| g.role != AssemblyRole::Meeting);
    let error = validation::validate_prepared_layout(&scene).unwrap_err();
    assert!(
        matches!(error, CaptureError::Rejected { rejection, .. } if rejection.kind == SceneRejectionKind::MissingProgramAnchor)
    );
}

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
            for axis in 0..2 {
                for f in [-0.35, 0.0, 0.35] {
                    let along = f * size.xz()[1 - axis];
                    let (left, left_yaw) = path.on_side(axis, -1., along).unwrap();
                    let (right, right_yaw) = path.on_side(axis, 1., along).unwrap();
                    assert!(left[axis] < right[axis]);
                    for (point, yaw, sign) in [(left, left_yaw, -1.), (right, right_yaw, 1.)] {
                        assert!((point[1 - axis] - along).abs() < 1e-5);
                        assert!(sign * Vec2::new(yaw.sin(), yaw.cos())[axis] > 0.);
                    }
                }
            }
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
