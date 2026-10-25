use super::*;
use crate::{
    annotation::schematic::{Overlay, RenderOptions, Schematic},
    scene::procedural_indoor::{objects, validation},
};

#[test]
fn shared_table_rows_have_seats_support_routes_and_replay() {
    let mut records = Vec::new();
    let started = std::time::Instant::now();
    for layout in [IndoorLayout::LectureHall, IndoorLayout::Cafeteria] {
        let mut selected = 0;
        let mut realized = 0;
        let mut alternatives = 0;
        let mut samples = 0;
        for seed in 0..128 {
            let density = [0., 0.35, 0.65, 1.][seed as usize % 4];
            let generation = std::time::Instant::now();
            let scene = IndoorManifest::generate_with_humans(seed, layout, density, 0, 0.35)
                .unwrap_or_else(|e| panic!("{layout:?} seed={seed}: {e}"));
            let generation_ms = generation.elapsed().as_secs_f64() * 1000.;
            validation::validate_layout(&scene).unwrap();
            scene.validate_public_programs().unwrap();
            let p = &scene.functional_programs()[0];
            let requested = if layout == IndoorLayout::LectureHall {
                p.parameters.table_rows.lecture_tables
            } else {
                p.parameters.table_rows.cafeteria_rows
            };
            selected += usize::from(requested);
            alternatives += usize::from(!requested);
            let groups: Vec<_> = p
                .members
                .iter()
                .filter_map(|&id| {
                    let table = &scene.objects[id];
                    if !matches!(table.kind, ObjectKind::Table | ObjectKind::Desk) {
                        return None;
                    }
                    let seats: Vec<_> = scene
                        .objects
                        .iter()
                        .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(id))
                        .collect();
                    (seats.len()
                        >= if layout == IndoorLayout::Cafeteria {
                            4
                        } else {
                            2
                        })
                    .then_some((table, seats))
                })
                .collect();
            realized += usize::from(!groups.is_empty());
            if requested && layout == IndoorLayout::LectureHall {
                assert!(
                    !groups.is_empty(),
                    "lecture seed {seed} has no usable shared desk"
                );
            }
            for (table, seats) in &groups {
                let inv = table.transform().compute_affine().inverse();
                let mut sides = [0, 0];
                for chair in seats {
                    assert!((table.position.y - chair.position.y).abs() < 0.015);
                    assert!(!super::super::super::seating::furniture_overlap(
                        table, chair, 0.008
                    ));
                    let q = inv.transform_point3(chair.position);
                    sides[usize::from(q.z > 0.)] += 1;
                    let facing = inv.transform_vector3(chair.transform().rotation * -Vec3::Z);
                    assert!(facing.z * q.z.signum() < -0.85,
                        "{layout:?} seed={seed} chair={} table={} local={q:?} facing={facing:?}: seat faces away from its row",
                        chair.id, table.id);
                }
                if layout == IndoorLayout::Cafeteria {
                    assert!(sides.iter().all(|&n| n >= 2), "one-sided dining row");
                } else {
                    assert_eq!(sides[0], 0);
                }
                for o in std::iter::once(*table).chain(seats.iter().copied()) {
                    assert!((o.position.y - scene.floor_height(o.position.xz())).abs() < 0.015);
                    let mesh = objects::build_object(o);
                    objects::audit::check(o, &mesh).unwrap();
                }
            }
            if seed < 4 {
                assert_eq!(
                    scene,
                    IndoorManifest::generate_with_humans(seed, layout, density, 0, 0.35).unwrap()
                );
                let replay: IndoorManifest =
                    serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
                assert_eq!(scene, replay);
                validation::validate_geometry(&scene).unwrap();
            }
            if requested && !groups.is_empty() && samples < 3 {
                Schematic::from_manifest(&scene, 0.)
                    .unwrap()
                    .write(
                        format!("out/table_rows/{layout:?}-{seed}"),
                        RenderOptions::default(),
                        Overlay::default(),
                    )
                    .unwrap();
                samples += 1;
            }
            records.push(serde_json::json!({
                "layout":layout,"seed":seed,"density":density,"generation_ms":generation_ms,"requested_table_rows":requested,
                "shared_tables":groups.len(),"chairs":p.accepted_capacity,
                "table_seats":groups.iter().map(|(t,c)| serde_json::json!({"table":t.id,"seats":c.len(),"elevation_m":t.position.y,"size":t.size})).collect::<Vec<_>>(),
                "policy":p.parameters.table_rows,
                "objects":p.members.iter().map(|&i| &scene.objects[i]).collect::<Vec<_>>(),
                "reservations":p.reservations,
            }));
        }
        assert!(selected >= 45 && alternatives >= 30);
        assert!(
            realized >= selected * 3 / 4,
            "{layout:?}: only {realized}/{selected} table-row rooms realized"
        );
        eprintln!("{layout:?}: {selected}/128 table-row selections, {realized} realized; {alternatives} alternative layouts");
    }
    std::fs::create_dir_all("out/table_rows").unwrap();
    std::fs::write("out/table_rows/audit.json", serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"128 consecutive seeds per layout, four densities, static people; no GPU throughput claim",
        "elapsed_seconds_including_validation_and_schematics":started.elapsed().as_secs_f64(),
        "rooms":records
    })).unwrap()).unwrap();
}

#[test]
fn older_receipts_default_to_original_seating() {
    let parameters = ProgramParameters::sample(3, 0, 0.65);
    let mut wire = serde_json::to_value(&parameters).unwrap();
    wire.as_object_mut().unwrap().remove("table_rows");
    let decoded: ProgramParameters = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded.table_rows, TableRowSeating::default());
    assert_eq!(decoded.lecture_seating, parameters.lecture_seating);
}
