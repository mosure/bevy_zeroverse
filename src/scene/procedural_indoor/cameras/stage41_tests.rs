use super::{CameraSettings, IndoorManifest};
use crate::sample::failure::{CaptureError, SceneRejectionKind};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    policy: serde_json::Value,
    rooms: Vec<Receipt>,
}

#[derive(Deserialize)]
struct Receipt {
    room: IndoorManifest,
    rejection: Option<String>,
}

/// Frozen 0.75.0 training rooms include the original furniture, people and
/// optics. Regenerating just their seeds would test a different furnished room.
#[test]
#[ignore = "full 16-camera CPU replay of accepted rooms and expensive production rejections"]
fn stage41_replays_published_camera_decisions_and_proposals() {
    let bytes = include_bytes!("../../../../tests/fixtures/stage41_camera_admission_g293.json.zst");
    let reader = ruzstd::decoding::StreamingDecoder::new(&bytes[..]).unwrap();
    let fixture: Fixture = serde_json::from_reader(reader).unwrap();
    let settings = CameraSettings::parse(&fixture.policy.to_string()).unwrap();
    assert_eq!(
        fixture
            .rooms
            .iter()
            .map(|r| r.room.seed)
            .collect::<Vec<_>>(),
        [
            614_440_196,
            614_440_199,
            614_440_200,
            614_449_168,
            614_449_169
        ]
    );

    for Receipt {
        mut room,
        rejection,
    } in fixture.rooms
    {
        assert_eq!(room.generator_version, 293);
        let cameras = std::mem::take(&mut room.cameras);
        let diagnostics = std::mem::take(&mut room.camera_diagnostics);
        let result = room.resample_cameras_typed(16, settings.clone(), 1.0);
        match (result, rejection) {
            (Ok(()), None) => assert_eq!(room.cameras.len(), 16),
            (Err(CaptureError::Rejected { rejection, .. }), Some(expected)) => {
                assert_eq!(rejection.kind, SceneRejectionKind::CameraPlacement);
                assert_eq!(rejection.seed, room.seed);
                // Includes the number of placed views and each rejection counter.
                assert_eq!(rejection.reason, expected);
            }
            (actual, expected) => {
                panic!("room {} expected {expected:?}, got {actual:?}", room.seed)
            }
        }
        assert_eq!(
            room.cameras, cameras,
            "camera/optics replay for {}",
            room.seed
        );
        assert_eq!(
            room.camera_diagnostics, diagnostics,
            "proposal counters for {}",
            room.seed
        );
    }
}
