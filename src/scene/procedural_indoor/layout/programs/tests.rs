use super::*;
use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};

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
    assert!(validate_geometry(&scene).unwrap().semantic_triangles["table"] > 0);
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
fn complete_activity_rooms_are_unchanged_by_lounge_recovery() {
    for seed in 0..16 {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        validate_layout(&scene).unwrap();
        let before = scene.clone();
        scene.complete_lounge_groups();
        scene.complete_underfilled_lounge_groups();
        assert_eq!(scene, before);
    }
}

#[test]
fn constrained_lounge_seed_49309986_has_sufficient_primary_furniture() {
    let scene =
        IndoorManifest::generate_with_humans(49_309_986, IndoorLayout::Mixed, 0.65, 4, 0.25)
            .unwrap();
    assert_eq!(scene.layout, IndoorLayout::Lounge);
    assert!(
        scene
            .objects
            .iter()
            .filter(|o| o.solid && !o.neighbor)
            .count()
            >= scene.minimum_main_objects()
    );
    validate_layout(&scene).unwrap();
    validate_geometry(&scene).unwrap();
    let mut recovered = scene.clone();
    recovered.complete_underfilled_lounge_groups();
    assert_eq!(scene, recovered, "complete rooms must not be refurnished");
    assert_eq!(
        scene,
        IndoorManifest::generate_with_humans(49_309_986, IndoorLayout::Mixed, 0.65, 4, 0.25)
            .unwrap()
    );
}

#[test]
fn constrained_lounge_seed_49309986_preserves_density_and_population_controls() {
    for density in [0., 0.35, 0.65, 1.] {
        for human_density in [0., 0.25, 1.] {
            let scene = IndoorManifest::generate_with_humans(
                49_309_986,
                IndoorLayout::Mixed,
                density,
                3,
                human_density,
            )
            .unwrap();
            validate_layout(&scene)
                .unwrap_or_else(|e| panic!("density={density}, humans={human_density}: {e}"));
            validate_geometry(&scene).unwrap();
        }
    }
}
