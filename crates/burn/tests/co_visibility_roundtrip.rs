use bevy_zeroverse::{
    render::co_visibility::{annotation_metadata, mask_from_color},
    sample::{Sample, View},
};
use bevy_zeroverse_burn::{
    chunk::{ColorCodec, load_chunk, save_chunk_with_codec},
    compression::Compression,
    fs::{load_sample_dir, save_sample_to_fs},
};

fn sample(count: usize) -> Sample {
    let views = (0..2)
        .flat_map(|t| {
            (0..count).map(move |camera| {
                let rgba: Vec<f32> = (0..35)
                    .flat_map(|p| {
                        let mask = if p < 2 {
                            0
                        } else {
                            ((1u32 << count) - 1) as u16 ^ (1 << camera)
                        };
                        [
                            mask as f32,
                            mask.count_ones() as f32,
                            (p != 0) as u8 as f32,
                            0.0,
                        ]
                    })
                    .collect();
                View {
                    co_visibility: bytemuck::cast_slice(&rgba).to_vec(),
                    time: t as f32 * 0.3,
                    ..Default::default()
                }
            })
        })
        .collect();
    Sample {
        views,
        view_dim: count as u32,
        co_visibility_metadata: Some(annotation_metadata(
            &(0..count).map(|i| i * 3 + 2).collect::<Vec<_>>(),
        )),
        ..Default::default()
    }
}

#[test]
fn all_sixteen_bits_roundtrip_with_validity_metadata_and_lossless_preview() {
    for count in [1, 3, 16] {
        let original = sample(count);
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
            assert_eq!(
                decoded[0].co_visibility_metadata,
                original.co_visibility_metadata
            );
            for (a, b) in original.views.iter().zip(&decoded[0].views) {
                assert_eq!(a.co_visibility, b.co_visibility);
            }
        }
        let path = save_sample_to_fs(&original, dir.path(), 5, 7, 5, false).unwrap();
        let decoded = load_sample_dir(&path).unwrap();
        assert_eq!(
            decoded.co_visibility_metadata,
            original.co_visibility_metadata
        );
        for (a, b) in original.views.iter().zip(&decoded.views) {
            assert_eq!(a.co_visibility, b.co_visibility);
        }
        let preview = image::open(path.join("co_visibility_000_00.png"))
            .unwrap()
            .into_rgb8();
        let expected = if count == 1 {
            0
        } else {
            ((1u32 << count) - 1) as u16 ^ 1
        };
        assert_eq!(
            mask_from_color(preview.get_pixel(2, 0).0, count),
            Some(expected)
        );
        std::fs::remove_file(path.join("co_visibility_000_00.npz")).unwrap();
        assert!(load_sample_dir(&path).is_err());
    }
}

#[test]
fn malformed_membership_and_incomplete_views_are_rejected() {
    for bad in [
        [f32::NAN, 0., 1., 0.],
        [1., 1., 1., 0.],
        [8., 1., 1., 0.],
        [2., 0., 1., 0.],
        [2., 1., 0., 0.],
        [2., 1., 1., 1.],
    ] {
        let mut item = sample(3);
        item.views[0].co_visibility[..16].copy_from_slice(bytemuck::bytes_of(&bad));
        let dir = tempfile::tempdir().unwrap();
        assert!(
            save_chunk_with_codec(
                &[item],
                dir.path(),
                0,
                Compression::None,
                7,
                5,
                false,
                ColorCodec::Raw
            )
            .is_err()
        );
    }
    let mut item = sample(3);
    item.views[1].co_visibility.clear();
    let dir = tempfile::tempdir().unwrap();
    assert!(
        save_chunk_with_codec(
            &[item],
            dir.path(),
            0,
            Compression::None,
            7,
            5,
            false,
            ColorCodec::Raw
        )
        .is_err()
    );
}
