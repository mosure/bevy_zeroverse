use bevy_zeroverse::{
    render::color::{ColorEncoding, linear_to_srgb},
    sample::{Sample, View},
    scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest},
};
use bevy_zeroverse_burn::{
    chunk::{load_chunk, save_chunk},
    compression::Compression,
    fs::{FsDataset, save_sample_to_fs},
};
use burn::data::dataset::Dataset;

fn sample() -> Sample {
    let rgba: Vec<f32> = (0..64).flat_map(|_| [0.18, 0.18, 0.18, 1.0]).collect();
    Sample {
        indoor: Some(IndoorManifest::generate(314, IndoorLayout::Conference, 0.8, 1).unwrap()),
        indoor_render_metadata: Some(
            serde_json::json!({"quality":"auto","gi":{"rays":32,"bounces":4},"precision":"float32_geometry"}),
        ),
        annotation_precision: bevy_zeroverse::sample::AnnotationPrecision::Float32Geometry,
        color_encoding: ColorEncoding::TonemappedLinear,
        views: vec![View {
            color: bytemuck::cast_slice(&rgba).to_vec(),
            semantic: bytemuck::cast_slice(&rgba).to_vec(),
            ..Default::default()
        }],
        view_dim: 1,
        ..Default::default()
    }
}

fn assert_roundtrip(original: &Sample, decoded: &Sample) {
    assert_eq!(original.indoor, decoded.indoor);
    assert_eq!(
        original.indoor_render_metadata,
        decoded.indoor_render_metadata
    );
    assert_eq!(original.annotation_precision, decoded.annotation_precision);
    assert_eq!(decoded.color_encoding, ColorEncoding::Srgb);
    let labels: &[f32] = bytemuck::cast_slice(&decoded.views[0].semantic);
    assert!(
        labels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[..3] == [0.18; 3])
    );
    let pixels: &[f32] = bytemuck::cast_slice(&decoded.views[0].color);
    for p in pixels.as_chunks::<4>().0 {
        assert!((p[0] - linear_to_srgb(0.18)).abs() < 0.015);
        assert!((p[0] - p[1]).abs() < 0.008 && (p[1] - p[2]).abs() < 0.008);
    }
}

#[test]
fn chunk_preserves_seed_manifest_and_does_not_renormalize_constant_rgb() {
    let original = sample();
    let dir = tempfile::tempdir().unwrap();
    let path = save_chunk(
        std::slice::from_ref(&original),
        dir.path(),
        0,
        Compression::None,
        8,
        8,
        false,
    )
    .unwrap();
    let decoded = load_chunk(path).unwrap();

    assert_roundtrip(&original, &decoded[0]);
    // Repacking already encoded RGB must not apply the transfer a second time.
    let path = save_chunk(&decoded, dir.path(), 1, Compression::None, 8, 8, false).unwrap();
    assert_roundtrip(&original, &load_chunk(path).unwrap()[0]);
}

#[test]
fn filesystem_preserves_seed_manifest_and_color_encoding() {
    let original = sample();
    let dir = tempfile::tempdir().unwrap();
    save_sample_to_fs(&original, dir.path(), 0, 8, 8, false).unwrap();
    let dataset = FsDataset::from_dir(dir.path()).unwrap();
    assert_roundtrip(&original, &dataset.get(0).unwrap());
}

#[test]
fn storage_preserves_primary_bounds_and_positions_outside_them() {
    let mut original = sample();
    original.aabb = [[-2., 0., -3.], [2., 4., 3.]];
    let positions = [[-0.25_f32, 1.5, 0.75, 1.]; 64];
    original.views[0].position = bytemuck::cast_slice(&positions).to_vec();
    let dir = tempfile::tempdir().unwrap();
    let path = save_chunk(
        std::slice::from_ref(&original),
        dir.path(),
        0,
        Compression::None,
        8,
        8,
        false,
    )
    .unwrap();
    save_sample_to_fs(&original, dir.path(), 0, 8, 8, false).unwrap();
    let chunk = load_chunk(path).unwrap();
    let filesystem = FsDataset::from_dir(dir.path()).unwrap().get(0).unwrap();
    for decoded in [&chunk[0], &filesystem] {
        assert_eq!(decoded.aabb, original.aabb);
        let decoded_positions: &[[f32; 4]] = bytemuck::cast_slice(&decoded.views[0].position);
        for (expected, actual) in positions.iter().zip(decoded_positions) {
            assert_eq!(expected[..3], actual[..3]);
        }
    }
}

#[test]
fn filesystem_supports_annotation_only_samples_without_lossy_palette_labels() {
    let mut original = sample();
    original.views[0].color.clear();
    let dir = tempfile::tempdir().unwrap();
    save_sample_to_fs(&original, dir.path(), 0, 8, 8, false).unwrap();
    let decoded = FsDataset::from_dir(dir.path()).unwrap().get(0).unwrap();
    assert!(decoded.views[0].color.is_empty());
    assert_eq!(decoded.views[0].semantic, original.views[0].semantic);
    assert_eq!(decoded.indoor, original.indoor);
}

#[test]
fn resume_counts_partial_chunks_and_rejects_gaps_and_overwrites() {
    use bevy_zeroverse_burn::generator::{WriteMode, resume_offsets};
    let sample = sample();
    let dir = tempfile::tempdir().unwrap();
    for index in 0..3 {
        save_chunk(
            std::slice::from_ref(&sample),
            dir.path(),
            index,
            Compression::None,
            8,
            8,
            false,
        )
        .unwrap();
    }
    assert_eq!(
        resume_offsets(dir.path(), WriteMode::Chunk, 256).unwrap(),
        (3, 3)
    );
    assert!(
        save_chunk(
            std::slice::from_ref(&sample),
            dir.path(),
            0,
            Compression::None,
            8,
            8,
            false
        )
        .is_err()
    );
    std::fs::remove_file(dir.path().join("000001.safetensors")).unwrap();
    assert!(resume_offsets(dir.path(), WriteMode::Chunk, 256).is_err());
}

#[test]
fn indoor_generation_validates_workers_trajectories_and_accepts_numeric_flow() {
    use bevy_zeroverse::{render::RenderMode, scene::ZeroverseSceneType};
    use bevy_zeroverse_burn::generator::{GenConfig, validate_gen_config};
    let mut config = GenConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..Default::default()
    };
    assert!(validate_gen_config(&config).is_ok());
    config.indoor_gi_rays = 1024;
    assert!(validate_gen_config(&config).is_ok());
    config.indoor_gi_rays = 63;
    assert!(validate_gen_config(&config).is_err());
    config.indoor_gi_rays = 16385;
    assert!(validate_gen_config(&config).is_err());
    config.indoor_gi_rays = 256;
    config.workers = 2;
    assert!(validate_gen_config(&config).is_err());
    config.workers = 1;
    config.playback_step = 1.0;
    assert!(validate_gen_config(&config).is_err());
    config.playback_step = 0.05;
    config
        .render_modes
        .extend([RenderMode::OpticalFlow, RenderMode::MotionVectors]);
    assert!(validate_gen_config(&config).is_ok());
}

#[test]
fn generation_ovoxel_requires_static_single_step_capture() {
    use bevy_zeroverse::app::OvoxelMode;
    use bevy_zeroverse_burn::generator::{GenConfig, validate_gen_config};
    let mut config = GenConfig::default();
    assert!(validate_gen_config(&config).is_ok()); // no export, five steps
    config.export_ovoxel = true;
    assert!(
        validate_gen_config(&config)
            .unwrap_err()
            .to_string()
            .contains("playback_steps=1")
    );
    config.playback_steps = 1;
    assert!(validate_gen_config(&config).is_ok());
    config.human_motion = Some("{}".into());
    assert!(
        validate_gen_config(&config)
            .unwrap_err()
            .to_string()
            .contains("human motion disabled")
    );
    config.human_motion = None;
    config.ov_mode = OvoxelMode::Disabled;
    assert!(validate_gen_config(&config).is_err());
    config.export_ovoxel = false;
    config.playback_steps = 3;
    assert!(validate_gen_config(&config).is_ok());
}

#[test]
fn raw_rgb_is_exact_srgb_once_in_both_export_formats() {
    use bevy_zeroverse_burn::chunk::{ColorCodec, save_chunk_with_codec};
    use bevy_zeroverse_burn::fs::save_sample_to_fs_with_codec;
    let original = sample();
    let directory = tempfile::tempdir().unwrap();
    let path = save_chunk_with_codec(
        std::slice::from_ref(&original),
        directory.path(),
        0,
        Compression::None,
        8,
        8,
        false,
        ColorCodec::Raw,
    )
    .unwrap();
    let decoded = load_chunk(path).unwrap();
    // Older Python raw chunks infer image dimensions directly from `color`.
    let encoded = std::fs::read(directory.path().join("000000.safetensors")).unwrap();
    let archive = safetensors::SafeTensors::deserialize(&encoded).unwrap();
    let tensors: Vec<_> = archive
        .tensors()
        .into_iter()
        .filter(|(name, _)| name != "color_shape")
        .collect();
    let python_style = safetensors::serialize(tensors, None).unwrap();
    let python_path = directory.path().join("python_raw.safetensors");
    std::fs::write(&python_path, python_style).unwrap();
    assert_eq!(
        load_chunk(python_path).unwrap()[0].views[0].color,
        decoded[0].views[0].color
    );
    let path =
        save_sample_to_fs_with_codec(&original, directory.path(), 0, 8, 8, false, ColorCodec::Raw)
            .unwrap();
    let folder = bevy_zeroverse_burn::fs::load_sample_dir(path).unwrap();
    for candidate in [&decoded[0], &folder] {
        let pixels: &[f32] = bytemuck::cast_slice(&candidate.views[0].color);
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[..3] == [linear_to_srgb(0.18); 3])
        );
        assert_eq!(candidate.color_encoding, ColorEncoding::Srgb);
    }
}

#[test]
fn resume_recognizes_compressed_chunk_indices() {
    use bevy_zeroverse_burn::generator::{WriteMode, resume_offsets};
    for codec in [
        Compression::Lz4 { level: 0 },
        Compression::Zstd { level: 0 },
    ] {
        let directory = tempfile::tempdir().unwrap();
        save_chunk(&[sample()], directory.path(), 0, codec, 8, 8, false).unwrap();
        save_chunk(&[sample()], directory.path(), 1, codec, 8, 8, false).unwrap();
        assert_eq!(
            resume_offsets(directory.path(), WriteMode::Chunk, 256).unwrap(),
            (2, 2)
        );
    }
}

#[test]
fn mixed_human_counts_roundtrip_without_inventing_padded_people() {
    use bevy_zeroverse::sample::HumanPoseSample;
    let mut populated = sample();
    populated.indoor = None;
    populated.human_instance_ids = vec![111, 222];
    populated.human_bone_names = vec!["pelvis".into()];
    populated.human_bone_parents = vec![-1];
    populated.human_pose_steps = vec![vec![
        HumanPoseSample {
            bone_positions: vec![[1., 2., 3.]],
            bone_rotations: vec![[0., 0., 0., 1.]],
        },
        HumanPoseSample {
            bone_positions: vec![[4., 5., 6.]],
            bone_rotations: vec![[0., 0., 0., 1.]],
        },
    ]];
    let mut empty = populated.clone();
    empty.human_pose_steps = vec![vec![]];
    empty.human_instance_ids.clear();
    let mut one = populated.clone();
    one.human_pose_steps[0].truncate(1);
    one.human_instance_ids.truncate(1);
    let directory = tempfile::tempdir().unwrap();
    let path = save_chunk(
        &[empty, populated, one],
        directory.path(),
        0,
        Compression::None,
        8,
        8,
        false,
    )
    .unwrap();
    let decoded = load_chunk(path).unwrap();
    assert_eq!(
        decoded
            .iter()
            .map(|sample| sample.human_pose_steps[0].len())
            .collect::<Vec<_>>(),
        vec![0, 2, 1]
    );
    assert_eq!(decoded[1].human_instance_ids, vec![111, 222]);
    assert_eq!(decoded[2].human_instance_ids, vec![111]);
    assert_eq!(
        decoded[1].human_pose_steps[0][1].bone_positions,
        vec![[4., 5., 6.]]
    );
}

#[test]
fn schematic_exports_survive_capture_archive_roundtrip() {
    use bevy_zeroverse::annotation::schematic::{Document, Overlay, RenderOptions};
    let mut original = sample();
    let camera = &original.indoor.as_ref().unwrap().cameras[0];
    original.views[0].world_from_view = camera.transform_at(0.).to_matrix().to_cols_array_2d();
    original.views[0].fovy = camera.fov_degrees.to_radians();
    original.views[0].calibration = Some(
        bevy_zeroverse::calibration::CameraCalibration::centered_pinhole(
            8,
            8,
            original.views[0].fovy,
            1.,
        )
        .unwrap(),
    );
    original.views[0].trajectory_progress = Some(0.);
    original.views[0].time_seconds = Some(0.);
    let mut next = original.views[0].clone();
    next.world_from_view = camera.transform_at(1.).to_matrix().to_cols_array_2d();
    next.time = 1.;
    next.trajectory_progress = Some(1.);
    next.time_seconds = Some(5.);
    original.views.push(next);
    let dir = tempfile::tempdir().unwrap();
    let path = save_chunk(
        std::slice::from_ref(&original),
        dir.path(),
        0,
        Compression::None,
        8,
        8,
        false,
    )
    .unwrap();
    let decoded = load_chunk(path).unwrap();
    let options = RenderOptions {
        width: 256,
        height: 256,
        ..Default::default()
    };
    decoded[0]
        .write_schematics(dir.path().join("plans"), options)
        .unwrap();
    for step in 0..2 {
        let plan = original.schematic(step).unwrap();
        assert_eq!(plan, decoded[0].schematic(step).unwrap());
        let document: Document = serde_json::from_slice(
            &std::fs::read(dir.path().join(format!("plans/{step:03}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(document.schematic, plan);
        let png = image::open(dir.path().join(format!("plans/{step:03}.png")))
            .unwrap()
            .to_rgba8();
        assert_eq!(png.dimensions(), (256, 256));
        assert_eq!(
            png.as_raw(),
            &plan.rgba(&options, &Overlay::default()).unwrap()
        );
    }
}
