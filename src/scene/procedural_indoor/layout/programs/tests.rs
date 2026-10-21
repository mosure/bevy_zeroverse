use super::*;
use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};

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
