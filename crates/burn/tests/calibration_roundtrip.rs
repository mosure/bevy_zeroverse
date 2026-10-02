use bevy_zeroverse::{
    calibration::CameraCalibration,
    sample::{Sample, View},
};
use bevy_zeroverse_burn::{
    chunk::{ColorCodec, load_chunk, save_chunk_with_codec},
    compression::Compression,
    fs::{load_sample_dir, save_sample_to_fs},
};

fn sample() -> Sample {
    Sample {
        view_dim: 2,
        views: (0..4)
            .map(|i| View {
                color: bytemuck::cast_slice(&vec![0.3f32; 7 * 5 * 4]).to_vec(),
                calibration: Some(
                    CameraCalibration::centered_pinhole(7, 5, 0.8 + i as f32 * 0.1, 1.4).unwrap(),
                ),
                trajectory_progress: Some((i / 2) as f32),
                time: (i / 2) as f32,
                time_seconds: (i < 2).then_some(0.),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}
fn assert_labels(actual: &Sample, expected: &Sample) {
    assert_eq!(actual.views.len(), expected.views.len());
    for (a, b) in actual.views.iter().zip(&expected.views) {
        assert_eq!(a.calibration, b.calibration);
        assert_eq!(a.trajectory_progress, b.trajectory_progress);
        assert_eq!(a.time_seconds, b.time_seconds);
        assert_eq!(a.time, b.time);
    }
}
#[test]
fn complete_calibration_and_unknown_physical_time_roundtrip() {
    let original = sample();
    let dir = tempfile::tempdir().unwrap();
    for (i, codec) in [ColorCodec::Raw, ColorCodec::Jpeg].into_iter().enumerate() {
        let path = save_chunk_with_codec(
            std::slice::from_ref(&original),
            dir.path(),
            i,
            Compression::None,
            7,
            5,
            false,
            codec,
        )
        .unwrap();
        let decoded = load_chunk(path).unwrap();
        assert_labels(&decoded[0], &original);
    }
    let path = save_sample_to_fs(&original, dir.path(), 4, 7, 5, false).unwrap();
    assert_labels(&load_sample_dir(path).unwrap(), &original);
}
#[test]
fn reject_partial_mislabeled_or_nonfinite_calibration() {
    let dir = tempfile::tempdir().unwrap();
    for index in 0..5 {
        let mut sample = sample();
        let v = &mut sample.views[1];
        match index {
            0 => v.calibration = None,
            1 => v.calibration.as_mut().unwrap().image_size = [5, 7],
            2 => v.calibration.as_mut().unwrap().k[0][0] = f32::NAN,
            3 => v.time_seconds = Some(-1.),
            _ => v.trajectory_progress = None,
        }
        assert!(save_sample_to_fs(&sample, dir.path(), index, 7, 5, false).is_err());
        assert!(
            save_chunk_with_codec(
                &[sample],
                dir.path(),
                index,
                Compression::None,
                7,
                5,
                false,
                ColorCodec::Raw
            )
            .is_err()
        );
    }
}
