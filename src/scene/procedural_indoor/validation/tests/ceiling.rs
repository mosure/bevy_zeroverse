use super::*;
use crate::sample::failure::{CaptureError, SceneRejectionKind};

fn overlapping_ceiling(seed: u64) -> IndoorManifest {
    let mut scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    // Reported seeds are repaired, valid fixtures. Inject a finite conflict to
    // test recovery without requiring those seeds to stay broken forever.
    validate_prepared_layout(&scene).unwrap();
    let ceiling = scene.ceiling_installation.as_mut().unwrap();
    ceiling.modules.push(ceiling.modules[0].clone());
    scene
}

#[test]
fn ceiling_conflicts_are_typed_only_after_all_integrity_checks() {
    for seed in [2, 65_991_960, 65_904_936] {
        let scene = overlapping_ceiling(seed);
        assert!(matches!(validate_prepared_layout(&scene),
            Err(CaptureError::Rejected { rejection, retired: false })
            if rejection.seed == seed && rejection.kind == SceneRejectionKind::PlacementClearance
                && rejection.reason.contains("ceiling module")));

        for fault in 0..12 {
            let mut corrupt = scene.clone();
            let module = corrupt
                .ceiling_installation
                .as_mut()
                .unwrap()
                .modules
                .last_mut()
                .unwrap();
            match fault {
                0 => corrupt.room_size.x = f32::NAN,
                1 => corrupt.objects.last_mut().unwrap().id = usize::MAX,
                2 => corrupt.objects.last_mut().unwrap().yaw = f32::INFINITY,
                3 => module.center.x = f32::NAN,
                4 => module.size.y = -1.,
                5 => module.support_top_m = f32::INFINITY,
                6 => module.pitch_m = f32::NAN,
                7 => module.thickness_m = 100.,
                8 => corrupt.generator_version = u32::MAX,
                9 => corrupt.camera_aspect_ratio = f32::NAN,
                10 => module.size.x = 0.,
                _ => module.detail.as_mut().unwrap().finish = u8::MAX,
            }
            assert!(
                matches!(
                    validate_prepared_layout(&corrupt),
                    Err(CaptureError::Failed(_))
                ),
                "seed {seed}: placement must not mask integrity fault {fault}"
            );
        }
    }
}

#[test]
fn later_module_corruption_overrides_an_earlier_ceiling_overlap() {
    let mut scene = overlapping_ceiling(2);
    let ceiling = scene.ceiling_installation.as_mut().unwrap();
    let mut malformed = ceiling.modules[0].clone();
    malformed.support_top_m = f32::NAN;
    ceiling.modules.push(malformed);
    assert!(matches!(
        validate_prepared_layout(&scene),
        Err(CaptureError::Failed(_))
    ));
}
