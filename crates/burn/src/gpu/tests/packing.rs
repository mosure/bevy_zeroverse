use super::*;
use bevy_zeroverse::sample::Sample;
use burn::tensor::TensorData;

#[test]
#[ignore = "runs on the Bevy-registered WGPU device, including software Vulkan"]
fn room_offsets_and_forward_flow_match_individual_views() -> Result<()> {
    setup_test_assets();
    let dataset = GpuLiveDataset::indoor()?;
    let device = dataset.device();
    for (steps, modes) in [
        (3, vec![RenderMode::Color]),
        (1, vec![RenderMode::MotionVectors]),
        (3, vec![RenderMode::OpticalFlow]),
        (
            3,
            vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
                RenderMode::OpticalFlow,
                RenderMode::MotionVectors,
                RenderMode::CoVisibility,
            ],
        ),
    ] {
        let spec = CaptureSpec::from_scene(BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            width: 5.,
            height: 2.,
            num_cameras: 2,
            playback_steps: steps,
            playback_step: 0.25,
            z_depth: true,
            depth_format: DepthFormat::Linear,
            render_modes: modes,
            ..default()
        })
        .color_output(ColorOutput::SrgbRgbF32)
        .resolve()?;
        let planes = spec.plan.copied.len();
        let row = (spec.scene().width as usize).div_ceil(16) * 16;
        let raw = |view: usize, plane: usize, y: usize, x: usize, channel: usize| {
            if x >= 5 {
                return -999.;
            }
            (view * 128 + plane * 24 + y * 8 + x + channel) as f32 / 1024.
        };
        let data = (0..spec.views * planes * 2 * row * 4)
            .map(|i| {
                let p = i / (row * 2 * 4);
                raw(
                    p / planes,
                    p % planes,
                    i / (row * 4) % 2,
                    i / 4 % row,
                    i % 4,
                )
            })
            .collect::<Vec<_>>();
        let storage = Tensor::<4>::from_data(
            TensorData::new(data, [spec.views * planes, 2, row, 4]),
            device,
        );
        let metadata = Sample {
            views: vec![default(); spec.views],
            view_dim: 2,
            ..default()
        };
        let individual = (0..spec.views).map(|v| {
            (
                storage
                    .clone()
                    .slice([v * planes..(v + 1) * planes, 0..2, 0..row, 0..4]),
                0,
            )
        });
        let separate = pack_sample(&spec, device, metadata.clone(), individual, None)?;
        let batch = spec
            .plan
            .index(Plane::Color)
            .map(|p| color::convert_room(storage.clone(), p, 5, planes, spec.color_output()))
            .transpose()?;
        let grouped = pack_sample(
            &spec,
            device,
            metadata,
            (0..spec.views).map(|v| (&storage, v * planes)),
            batch,
        )?;
        assert_eq!(separate.metadata, grouped.metadata);
        for (v, result) in grouped.views.iter().enumerate() {
            for &plane in &spec.plan.copied {
                // Motion-only requests consume the flow plane internally.
                let (actual, expected) = (result.plane(plane), separate.views[v].plane(plane));
                assert_eq!(actual.is_some(), expected.is_some());
                if let (Some(actual), Some(expected)) = (actual, expected) {
                    equal_plane(
                        values(actual.clone()),
                        values(expected.clone()),
                        "room vs individual",
                    );
                }
            }
            for (output, normalized, mode) in [
                (&result.optical_flow, false, RenderMode::OpticalFlow),
                (&result.motion_vectors, true, RenderMode::MotionVectors),
            ] {
                assert_eq!(output.is_some(), spec.scene().render_modes.contains(&mode));
                if let Some(output) = output {
                    let flow_plane = spec.plan.index(Plane::OpticalFlow).unwrap();
                    let got = values(output.clone());
                    for (i, value) in got.into_iter().enumerate() {
                        let c = i % 4;
                        let next = v + 2;
                        let displacement = if next < spec.views {
                            raw(next, flow_plane, i / 20, i / 4 % 5, c)
                        } else {
                            0.
                        };
                        let scale = if normalized { [5., 2., 1., 1.][c] } else { 1. };
                        assert!((value - displacement / scale).abs() <= f32::EPSILON);
                    }
                }
            }
        }
    }
    Ok(())
}
