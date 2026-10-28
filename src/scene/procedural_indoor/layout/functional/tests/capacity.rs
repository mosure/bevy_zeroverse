use super::*;
use crate::{
    sample::failure::{CaptureError, SceneRejectionKind},
    scene::procedural_indoor::{cameras::CameraSettings, validation},
};

#[test]
fn dining_capacity_counts_banquettes_but_preserves_legacy_chair_receipts() {
    let scene =
        IndoorManifest::generate_with_humans(66_369_950, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut program = scene.functional_programs()[0].clone();
    // Freeze the counting contract independently of future seed-to-layout changes.
    let mut objects = vec![scene.objects[0].clone(); 5];
    for (id, (object, kind)) in objects
        .iter_mut()
        .zip([
            ObjectKind::Chair,
            ObjectKind::Banquette,
            ObjectKind::Banquette,
            ObjectKind::Table,
            ObjectKind::Chair,
        ])
        .enumerate()
    {
        object.id = id;
        object.kind = kind;
    }
    program.members = vec![0, 1, 2, 3]; // The last chair belongs to another region.
    program.capacity_unit = CapacityUnit::SeatingStations;
    assert_eq!(program.realized_capacity(&objects), 3);
    program.members = vec![1];
    assert_eq!(
        program.realized_capacity(&objects),
        1,
        "one shared bench is one station"
    );
    program.members = vec![0, 1, 2, 3];
    program.capacity_unit = CapacityUnit::ChairStations;
    assert_eq!(program.realized_capacity(&objects), 1);
    let old_receipt = serde_json::to_vec(&program).unwrap();
    let old_receipt: ProgramAssembly = serde_json::from_slice(&old_receipt).unwrap();
    assert_eq!(old_receipt.capacity_unit, CapacityUnit::ChairStations);
    assert_eq!(old_receipt.realized_capacity(&objects), 1);
}

#[test]
fn reported_66369950_dining_capacity_geometry_and_training_rig() {
    let started = std::time::Instant::now();
    let mut scene =
        IndoorManifest::generate_with_humans(66_369_950, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    assert_eq!(scene.layout, IndoorLayout::Cafe);
    let program = &scene.functional_programs()[0];
    assert_eq!(program.capacity_unit, CapacityUnit::SeatingStations);
    assert!(program.accepted_capacity >= 2);
    assert!(program
        .members
        .iter()
        .any(|&id| scene.objects[id].kind == ObjectKind::Banquette));
    eprintln!(
        "seed {}: {:?}, accepted={} requested={}, rejected_candidates={}, objects={}, generation_ms={:.3}",
        scene.seed,
        program.capacity_unit,
        program.accepted_capacity,
        program.requested_capacity,
        program.rejected_candidates,
        scene.objects.len(),
        started.elapsed().as_secs_f64() * 1000.,
    );
    validation::validate_layout(&scene).unwrap();
    validation::validate_geometry(&scene).unwrap();
    let replay =
        IndoorManifest::generate_with_humans(scene.seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    assert_eq!(scene, replay);
    let policy = CameraSettings::parse(include_str!(
        "../../../cameras/ceiling/tests/mixed_training_camera_settings.json"
    ))
    .unwrap();
    scene.resample_cameras(8, policy, 1.).unwrap();
    validation::validate_layout(&scene).unwrap();
    assert!(scene.accepts_camera_geometry());
    let receipt: IndoorManifest =
        serde_json::from_slice(&serde_json::to_vec(&scene).unwrap()).unwrap();
    assert_eq!(scene, receipt);
}

#[test]
fn dining_capacity_still_rejects_insufficient_seating_and_corrupt_receipts() {
    let mut scene =
        IndoorManifest::generate_with_humans(66_369_950, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let program = &mut scene.assembly_diagnostics.programs[0];
    let mut kept_seat = false;
    program.members.retain(|&id| {
        if matches!(
            scene.objects[id].kind,
            ObjectKind::Chair | ObjectKind::Banquette
        ) {
            let keep = !kept_seat;
            kept_seat = true;
            keep
        } else {
            true
        }
    });
    program.accepted_capacity = program.realized_capacity(&scene.objects);
    assert_eq!(program.accepted_capacity, 1);
    assert!(matches!(
        scene.validate_public_programs_typed(),
        Err(CaptureError::Rejected { rejection, .. })
            if rejection.kind == SceneRejectionKind::InsufficientProgramCapacity
    ));
    scene.assembly_diagnostics.programs[0].accepted_capacity = 2;
    assert!(matches!(
        scene.validate_public_programs_typed(),
        Err(CaptureError::Failed(message)) if message.contains("capacity receipt differs")
    ));
}

#[test]
fn dining_capacity_neighbors_and_density_variants_preserve_geometry() {
    for seed in 66_369_942..66_369_958 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        validation::validate_geometry(&scene).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
    for density in [0., 0.65, 1.] {
        for people in [0., 0.35] {
            let scene = IndoorManifest::generate_with_humans(
                66_369_950,
                IndoorLayout::Mixed,
                density,
                0,
                people,
            )
            .unwrap();
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("density={density}, people={people}: {e}"));
            validation::validate_geometry(&scene).unwrap();
        }
    }
}
