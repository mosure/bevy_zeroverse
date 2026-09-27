use bevy_zeroverse::sample::{Sample, View};
use bevy_zeroverse_burn::{
    chunk::{load_chunk, save_chunk},
    compression::Compression,
    fs::{load_sample_dir, save_sample_to_fs},
};

fn sample() -> Sample {
    let mut views = Vec::new();
    for t in 0..3 {
        for camera in 0..2 {
            let pixels: Vec<f32> = (0..35)
                .flat_map(|i| {
                    if t == 2 || i == 0 {
                        [0.0; 4]
                    } else {
                        [
                            -17.125 - camera as f32,
                            t as f32 + 0.000_123,
                            1.0,
                            (i % 3 != 0) as u8 as f32,
                        ]
                    }
                })
                .collect();
            let normalized: Vec<f32> = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| [p[0] / 7.0, p[1] / 5.0, p[2], p[3]])
                .collect();
            views.push(View {
                optical_flow: bytemuck::cast_slice(&pixels).to_vec(),
                motion_vectors: bytemuck::cast_slice(&normalized).to_vec(),
                time: t as f32 * 0.3,
                ..Default::default()
            });
        }
    }
    Sample {
        views,
        view_dim: 2,
        ..Default::default()
    }
}

#[test]
fn signed_subpixel_flow_and_masks_roundtrip_without_rgb_or_tonemapping() {
    let sample = sample();
    let dir = tempfile::tempdir().unwrap();
    for (index, compression) in [Compression::None, Compression::Lz4 { level: 0 }]
        .into_iter()
        .enumerate()
    {
        let path = save_chunk(
            std::slice::from_ref(&sample),
            dir.path(),
            index,
            compression,
            7,
            5,
            false,
        )
        .unwrap();
        let decoded = load_chunk(path).unwrap();
        for (a, b) in sample.views.iter().zip(&decoded[0].views) {
            assert_eq!(a.optical_flow, b.optical_flow);
            assert_eq!(a.motion_vectors, b.motion_vectors);
        }
    }
    let path = save_sample_to_fs(&sample, dir.path(), 5, 7, 5, false).unwrap();
    assert!(!path.join("optical_flow_000_00.jpg").exists());
    let decoded = load_sample_dir(path).unwrap();
    for (a, b) in sample.views.iter().zip(&decoded.views) {
        assert_eq!(a.optical_flow, b.optical_flow);
        assert_eq!(a.motion_vectors, b.motion_vectors);
    }
}

#[test]
fn flow_export_rejects_nonfinite_values_and_invalid_masks() {
    let mut sample = sample();
    let dir = tempfile::tempdir().unwrap();
    for bad in [
        [f32::NAN, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.4, 0.0],
    ] {
        sample.views[0].optical_flow[..16].copy_from_slice(bytemuck::bytes_of(&bad));
        assert!(
            save_chunk(
                std::slice::from_ref(&sample),
                dir.path(),
                0,
                Compression::None,
                7,
                5,
                false
            )
            .is_err()
        );
    }
}
