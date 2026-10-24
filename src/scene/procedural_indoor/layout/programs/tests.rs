use super::*;
use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};

#[test]
fn rejected_mixed_rig_seed_has_a_furnished_primary_room() {
    for density in [0., 0.65, 1.] {
        let scene =
            IndoorManifest::generate_with_humans(65_459_150, IndoorLayout::Mixed, density, 0, 0.)
                .unwrap();
        validate_layout(&scene).unwrap();
        validate_geometry(&scene).unwrap();
        let desk = scene
            .objects
            .iter()
            .find(|o| {
                o.kind == ObjectKind::Desk && !o.neighbor && scene.in_primary_room(o.position, 0.)
            })
            .expect("other partitions must not satisfy an empty primary work area");
        let chair = scene
            .objects
            .iter()
            .find(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(desk.id))
            .expect("the repair must include usable seating");
        for o in [desk, chair] {
            assert!(scene.in_primary_room(o.position, 0.));
            // Final chairs may be pushed in using the existing 14 mm seating
            // clearance; the initial 60 mm furnishing margin no longer applies.
            assert!(scene.placement_clear(o, 0.));
            assert!((scene.floor_height(o.position.xz()) - o.position.y).abs() < 1e-5);
        }
        assert!(!seating::furniture_overlap(desk, chair, 0.014));
        assert!((desk.position.y - chair.position.y).abs() < 1e-5);
        assert!(scene.assembly_diagnostics.proposed_workstations < 2000);
        let mut recovered = scene.clone();
        recovered.complete_empty_primary_work_area();
        assert_eq!(
            recovered, scene,
            "a complete primary group must be untouched"
        );
        let replay =
            IndoorManifest::generate_with_humans(65_459_150, IndoorLayout::Mixed, density, 0, 0.)
                .unwrap();
        assert_eq!(scene, replay);
    }
}

#[test]
fn impossible_primary_work_area_repair_leaves_no_partial_group() {
    use crate::scene::procedural_indoor::envelope::FloorPatch;
    let mut scene =
        IndoorManifest::generate_with_humans(65_459_150, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    scene.rug_supports.clear();
    scene.assembly_diagnostics = default();
    let (lo, hi) = scene.primary_room_bounds();
    // Every desk/chair footprint spans multiple risers. No supported pair can
    // fit, even though the primary zone's bounding rectangle is large enough.
    scene.envelope.as_mut().unwrap().floor_patches = (0..64)
        .filter(|i| i % 2 == 0)
        .map(|i| FloorPatch {
            min: Vec2::new(lo.x + (hi.x - lo.x) * i as f32 / 64., lo.y),
            max: Vec2::new(lo.x + (hi.x - lo.x) * (i + 1) as f32 / 64., hi.y),
            height: 0.12,
        })
        .collect();
    let before = scene.clone();
    for _ in 0..2 {
        scene.complete_empty_primary_work_area();
        assert_eq!(
            scene, before,
            "exhaustion must not leave furniture or receipts"
        );
    }
}

#[test]
fn constrained_activity_seed_20053_has_a_usable_primary_surface() {
    for density in [0., 0.35, 0.65, 1.] {
        for human_density in [0., 0.25, 1.] {
            let scene =
                IndoorManifest::office_regression_fixture(20_053, density, 3, human_density)
                    .unwrap();
            validate_layout(&scene)
                .unwrap_or_else(|e| panic!("density={density}, humans={human_density}: {e}"));
            validate_geometry(&scene).unwrap();
            let table = scene
                .objects
                .iter()
                .find(|o| {
                    !o.neighbor
                        && matches!(
                            o.kind,
                            ObjectKind::CoffeeTable | ObjectKind::Desk | ObjectKind::Table
                        )
                })
                .expect("a real activity surface must survive constrained placement");
            if density == 0.65 {
                // A new grammar can fit a workstation where the original
                // reported scene needed a sofa-side repair. Test the functional
                // relationship, not the old seed's exact furniture category.
                assert!(
                    scene.objects.iter().any(|o| {
                        !o.neighbor
                            && (o.position.y - table.position.y).abs() < 0.01
                            && (o.kind == ObjectKind::Chair
                                && o.interaction_target == Some(table.id)
                                || table.kind == ObjectKind::CoffeeTable
                                    && o.kind == ObjectKind::Sofa
                                    && o.position.distance(table.position) < 2.0)
                    }),
                    "the repaired surface must serve real seating"
                );
            }
            let mut recovered = scene.clone();
            recovered.complete_underfilled_activity_groups();
            assert_eq!(scene, recovered, "complete rooms must not be refurnished");
            assert_eq!(
                scene,
                IndoorManifest::office_regression_fixture(20_053, density, 3, human_density,)
                    .unwrap(),
                "repaired scenes must replay deterministically"
            );
        }
    }
}

#[test]
fn accepted_meetings_have_real_seats_and_replayable_receipts() {
    let mut meetings = 0;
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.65, 0, 0.)
                .unwrap();
        scene.validate_assemblies().unwrap();
        for group in &scene.assembly_diagnostics.accepted {
            if group.role == assemblies::AssemblyRole::Meeting {
                meetings += 1;
                assert!(
                    group.seats.len() >= (group.requested_capacity as f32 * 0.65).ceil() as usize
                );
                assert!(group.attempts <= 12);
            }
        }
        let replay =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Conference, 0.65, 0, 0.)
                .unwrap();
        assert_eq!(scene, replay);
    }
    assert!(
        meetings > 16,
        "conference activity should realize meeting assemblies: {meetings}"
    );
}

#[test]
fn constrained_mixed_activity_room_retains_a_complete_furniture_group() {
    let scene =
        IndoorManifest::generate_with_humans(43_218_880, IndoorLayout::Mixed, 0.65, 3, 0.25)
            .unwrap();
    validate_layout(&scene).unwrap();
    let table = scene
        .objects
        .iter()
        .find(|o| !o.neighbor && o.kind == ObjectKind::CoffeeTable)
        .expect("existing lounge seating must have a usable primary activity surface");
    assert!(scene.placement_clear(table, 0.06));
    assert!(scene.objects.iter().any(|o| {
        if o.neighbor || o.kind != ObjectKind::Sofa {
            return false;
        }
        let local = o
            .transform()
            .compute_affine()
            .inverse()
            .transform_point3(table.position);
        local.z < -o.size.z * 0.5 && local.x.abs() < o.size.x * 0.5
    }));
    assert!(validate_geometry(&scene).unwrap().semantic_triangles["coffee_table"] > 0);
    assert_eq!(
        scene,
        IndoorManifest::generate_with_humans(43_218_880, IndoorLayout::Mixed, 0.65, 3, 0.25)
            .unwrap()
    );
}

#[test]
fn constrained_activity_room_supports_density_and_occupancy_variation() {
    for density in [0.0, 0.35, 0.65, 1.0] {
        for human_density in [0.0, 0.25, 1.0] {
            let scene = IndoorManifest::generate_with_humans(
                43_218_880,
                IndoorLayout::Mixed,
                density,
                3,
                human_density,
            )
            .unwrap();
            validate_layout(&scene).unwrap_or_else(|e| {
                panic!("density={density}, human_density={human_density}: {e}")
            });
        }
    }
}

#[test]
fn complete_activity_rooms_are_unchanged_by_furnishing_recovery() {
    for seed in 0..16 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        validate_layout(&scene).unwrap();
        let before = scene.clone();
        scene.complete_lounge_groups();
        scene.complete_underfilled_activity_groups();
        assert!(
            scene == before,
            "seed {seed}: recovery added {:?}",
            &scene.objects[before.objects.len()..]
        );
    }
}

#[test]
fn constrained_lounge_seed_49309986_has_sufficient_primary_furniture() {
    let scene = IndoorManifest::office_regression_fixture(49_309_986, 0.65, 4, 0.25).unwrap();
    assert_eq!(scene.layout, IndoorLayout::Lounge);
    assert!(
        scene
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor)
            .count()
            >= scene.minimum_main_objects(),
        "required={}, objects={:?}",
        scene.minimum_main_objects(),
        scene
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor)
            .map(|o| (o.id, o.kind, o.position, o.size))
            .collect::<Vec<_>>()
    );
    validate_layout(&scene).unwrap();
    validate_geometry(&scene).unwrap();
    let mut recovered = scene.clone();
    recovered.complete_underfilled_activity_groups();
    assert_eq!(scene, recovered, "complete rooms must not be refurnished");
    assert_eq!(
        scene,
        IndoorManifest::office_regression_fixture(49_309_986, 0.65, 4, 0.25).unwrap()
    );
}

#[test]
fn constrained_lounge_seed_49309986_preserves_density_and_population_controls() {
    for density in [0., 0.35, 0.65, 1.] {
        for human_density in [0., 0.25, 1.] {
            let scene =
                IndoorManifest::office_regression_fixture(49_309_986, density, 3, human_density)
                    .unwrap();
            validate_layout(&scene)
                .unwrap_or_else(|e| panic!("density={density}, humans={human_density}: {e}"));
            validate_geometry(&scene).unwrap();
        }
    }
}

#[test]
fn constrained_library_seed_49981871_retains_sufficient_activity_furniture() {
    let scene = IndoorManifest::office_regression_fixture(49_981_871, 0.65, 0, 0.25).unwrap();
    assert_eq!(scene.layout, IndoorLayout::Library);
    validate_layout(&scene).unwrap();
    validate_geometry(&scene).unwrap();
    assert!(scene
        .objects
        .iter()
        .any(|o| !o.neighbor && o.kind == ObjectKind::Bookcase));
    let mut recovered = scene.clone();
    recovered.complete_underfilled_activity_groups();
    assert_eq!(scene, recovered, "recovery must be idempotent");
    assert_eq!(
        scene,
        IndoorManifest::office_regression_fixture(49_981_871, 0.65, 0, 0.25).unwrap()
    );
}

#[test]
fn constrained_library_seed_49981871_preserves_density_population_and_camera_controls() {
    for density in [0., 0.35, 0.65, 1.] {
        for human_density in [0., 0.25, 1.] {
            let mut scene =
                IndoorManifest::office_regression_fixture(49_981_871, density, 3, human_density)
                    .unwrap();
            validate_layout(&scene)
                .unwrap_or_else(|e| panic!("density={density}, humans={human_density}: {e}"));
            validate_geometry(&scene).unwrap();
            let objects = scene.objects.clone();
            scene.resample_cameras(4, default(), 1.6).unwrap();
            assert_eq!(
                scene.objects, objects,
                "camera controls must not refurnish the room"
            );
            validate_layout(&scene).unwrap();
        }
    }
}

#[test]
fn raised_kitchenette_65286812_completes_supported_waste_storage() {
    for density in [0., 0.35, 0.65, 1.] {
        for human_density in [0., 0.25, 1.] {
            let scene = IndoorManifest::generate_with_humans(
                65_286_812,
                IndoorLayout::Mixed,
                density,
                0,
                human_density,
            )
            .unwrap();
            assert_eq!(scene.layout, IndoorLayout::Kitchenette);
            validate_layout(&scene).unwrap_or_else(|e| panic!("{density} {human_density}: {e}"));
            validate_geometry(&scene).unwrap();
            let mut replay = scene.clone();
            replay.complete_underfilled_activity_groups();
            assert_eq!(scene, replay, "accepted rooms must not be refurnished");
            if density == 0.65 {
                let sink = scene
                    .objects
                    .iter()
                    .find(|o| !o.neighbor && o.kind == ObjectKind::KitchenSink)
                    .unwrap();
                assert!(sink.position.y > 0.1, "exercise a raised kitchen floor");
                assert!(
                    scene.objects.iter().any(|o| !o.neighbor
                        && o.kind == ObjectKind::TrashCan
                        && (o.position.y - sink.position.y).abs() < 0.001
                        && o.position.distance(sink.position) < 2.0),
                    "the completed storage must serve the kitchen on its supported floor"
                );
            }
        }
    }
    let a = IndoorManifest::generate(65_286_812, IndoorLayout::Mixed, 0.65, 0).unwrap();
    let b = IndoorManifest::generate(65_286_812, IndoorLayout::Mixed, 0.65, 0).unwrap();
    assert_eq!(a, b, "the repair must preserve deterministic replay");
}
