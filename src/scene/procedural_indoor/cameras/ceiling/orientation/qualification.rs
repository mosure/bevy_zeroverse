//! Opt-in matched CPU search measurement; no renderer/device/model initialization.
use super::super::*;
use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};

#[test]
#[ignore = "matched CPU ceiling-tilt placement qualification"]
fn ceiling_tilt_placement_qualification() {
    let mut rows = Vec::new();
    for seed in (0..8).chain(std::iter::once(65_573_956)) {
        let base = IndoorManifest::generate_with_humans(
            seed,
            IndoorLayout::Mixed,
            0.65,
            0,
            if seed == 65_573_956 { 0. } else { 0.25 },
        )
        .unwrap();
        let counts: &[usize] = if seed == 65_573_956 {
            &[8]
        } else {
            &[3, 8, 12]
        };
        for &count in counts {
            for tilt in if (seed + count as u64).is_multiple_of(2) {
                [0., 12.]
            } else {
                [12., 0.]
            } {
                let mut scene = base.clone();
                let mut policy = if seed == 65_573_956 {
                    CameraSettings::parse(include_str!(
                        "../tests/mixed_training_camera_settings.json"
                    ))
                    .unwrap()
                } else {
                    CameraSettings {
                        motion_fraction: 0.65,
                        lenses: Some(LensSampling {
                            probability: 0.5,
                            ..default()
                        }),
                        ceiling_mount: Some(CeilingCameraSettings {
                            probability: 0.5,
                            motion_fraction: 0.65,
                            lenses: LensSampling {
                                probability: 0.5,
                                ..CeilingCameraSettings::default().lenses
                            },
                            ..default()
                        }),
                        ..default()
                    }
                };
                policy.ceiling_mount.as_mut().unwrap().tilt_degrees = [0., tilt];
                let slots: Vec<_> = (0..count).map(|i| {
                    let mounted = policy.ceiling_mounted(seed, i);
                    serde_json::json!({"mounted": mounted, "fisheye": policy.lens_for(seed,i).is_some(), "moving": policy.camera_moves(seed,i)})
                }).collect();
                let start = std::time::Instant::now();
                let result =
                    scene.resample_cameras(count, policy, [1., 16. / 9.][seed as usize % 2]);
                let row = serde_json::json!({"seed": seed, "count": count, "tilt_max": tilt, "seconds": start.elapsed().as_secs_f64(), "error": result.err(), "slots":slots, "diagnostics": scene.camera_diagnostics});
                eprintln!("TILT_RUN {row}");
                rows.push(row);
            }
        }
    }
    let dir = std::path::Path::new("out/ceiling_tilt_20261109");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("matched-placement.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
    assert!(
        rows.iter().all(|row| row["error"].is_null()),
        "placement failures saved in matched-placement.json"
    );
}
