use super::*;

#[test]
fn every_supported_rgb_sum_decodes_without_ambiguity() {
    for count in 1..=MAX_CAMERAS {
        for mask in 0..(1u32 << count) {
            let rgb = mask_color(mask as u16, count);
            assert_eq!(mask_from_color(rgb, count), Some(mask as u16));
        }
    }
    assert_eq!(mask_from_color([1, 0, 0], 2), None);
    assert_eq!(mask_from_color([0; 3], 0), None);
    assert_eq!(mask_from_color([0; 3], 17), None);
}

#[test]
fn legend_keeps_global_camera_identity_and_reserves_source_bit() {
    let metadata = annotation_metadata(&[2, 5, 9]);
    assert_eq!(metadata["legend"][1]["camera_index"], 5);
    assert_eq!(metadata["legend"][1]["mask"], 2);
    assert_eq!(mask_color(0b101, 3), [255, 0, 255]);
}

#[test]
fn invalid_configuration_planes_and_legends_are_rejected() {
    for count in [0, 17] {
        assert!(validate_config(&[RenderMode::CoVisibility], count).is_err());
        assert!(validate_config(&[RenderMode::Color], count).is_ok());
    }
    for pixel in [
        [f32::NAN, 0., 1., 0.],
        [0.5, 0., 1., 0.],
        [1., 1., 1., 0.],
        [8., 1., 1., 0.],
        [2., 0., 1., 0.],
        [2., 1., 0., 0.],
        [0., 0., 1., 1.],
    ] {
        assert!(validate_plane(bytemuck::bytes_of(&pixel), 1, 3, 0).is_err());
    }
    assert!(validate_plane(bytemuck::bytes_of(&[2f32, 1., 1., 0.]), 1, 3, 0).is_ok());
    assert!(validate_plane(bytemuck::bytes_of(&[0f32, 0., 1., 0.]), 1, 1, 0).is_ok());
    let mut meta = annotation_metadata(&[1, 4, 8]);
    validate_metadata(&meta, 3).unwrap();
    meta["legend"][1]["camera_index"] = serde_json::json!(1);
    assert!(validate_metadata(&meta, 3).is_err());
}
