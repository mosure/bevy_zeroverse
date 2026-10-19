//! Fused depadding, channel packing and IEC sRGB transfer on the training device.
//! Runs after linear-light SSAA. No tonemapping, quantization, or host pixels.
use super::*;
use burn::tensor::Tensor;
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::layout::linear::{LinearView, LinearViewMut},
};

#[allow(clippy::manual_range_contains)] // CubeCL requires scalar comparisons here.
#[cube(launch_unchecked)]
fn pack_color(
    input: LinearView<'_, f32>,
    mut output: LinearViewMut<'_, f32>,
    #[comptime] width: usize,
    #[comptime] height: usize,
    #[comptime] padded_width: usize,
    #[comptime] plane: usize,
    #[comptime] planes_per_view: usize,
    #[comptime] channels: usize,
    #[comptime] srgb: bool,
) {
    let i = ABSOLUTE_POS;
    if !output.is_in_bounds(i) {
        terminate!();
    }
    let channel = i % channels;
    let full_pixel = i / channels;
    let view = full_pixel / (width * height);
    let pixel = full_pixel % (width * height);
    let source = (((view * planes_per_view + plane) * height + pixel / width) * padded_width
        + pixel % width)
        * 4
        + channel;
    let mut value = input.read(source);
    // Preserve nonfinite values for validation; never turn a failed render into
    // apparently valid white/black training data. Alpha is never gamma encoded.
    if srgb && channel < 3 && value <= f32::MAX && value >= -f32::MAX {
        value = f32::clamp(value, 0.0, 1.0);
        if value <= 0.0031308 {
            value *= 12.92;
        } else {
            value = 1.055 * f32::powf(value, 1.0 / 2.4) - 0.055;
        }
    }
    output.write(i, value);
}

pub(super) fn convert(
    input: Tensor<4>,
    plane: usize,
    width: usize,
    output: ColorOutput,
) -> Result<Tensor<3>> {
    let [planes, height, _, _] = input.dims();
    Ok(convert_room(input, plane, width, planes, output)?.reshape([
        height,
        width,
        output.channels(),
    ]))
}
/// Fuse every camera/time view in a registered room arena in a single launch.
pub(super) fn convert_room(
    input: Tensor<4>,
    plane: usize,
    width: usize,
    planes_per_view: usize,
    output: ColorOutput,
) -> Result<Tensor<4>> {
    let [planes, height, padded_width, input_channels] = input.dims();
    ensure!(
        input_channels == 4
            && planes_per_view > 0
            && planes > 0
            && height > 0
            && width > 0
            && planes.is_multiple_of(planes_per_view)
            && plane < planes_per_view
            && width <= padded_width,
        "invalid color plane extent"
    );
    let views = planes / planes_per_view;
    let channels = output.channels();
    if output == ColorOutput::TonemappedLinearRgbaF32 && views == 1 {
        return Ok(input.slice([plane..plane + 1, 0..height, 0..width, 0..4]));
    }
    if output == ColorOutput::TonemappedLinearRgbaF32
        && planes_per_view == 1
        && width == padded_width
    {
        return Ok(input);
    }
    let count = views * height * width * channels;
    let result =
        Tensor::<1>::empty([count], &input.device()).reshape([views, height, width, channels]);
    let source = primitive(input)?;
    let destination = primitive(result.clone())?;
    let client = destination.client.clone();
    let cube_dim = CubeDim::new(&client, count);
    let cube_count = calculate_cube_count_elemwise(&client, count, cube_dim);
    // SAFETY: Buffers are checked plain f32 storage. The output guard and
    // validated view/plane/stride arithmetic keep every access within bounds.
    unsafe {
        pack_color::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            source.into_linear_view(),
            destination.into_linear_view(),
            width,
            height,
            padded_width,
            plane,
            planes_per_view,
            channels,
            output.is_srgb(),
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    #[test]
    #[ignore = "requires GPU; analytic sRGB and padding oracle"]
    fn srgb_gpu_matches_scalar_oracle() {
        let device = Device::wgpu(Default::default());
        let threshold = 0.0031308_f32;
        let ramp = [
            -0.1,
            0.,
            f32::from_bits(threshold.to_bits() - 1),
            threshold,
            f32::from_bits(threshold.to_bits() + 1),
            0.18,
            0.5,
            1.,
            1.5,
        ];
        let mut values = vec![77.; 2 * 16 * 4];
        for y in 0..2 {
            for x in 0..9 {
                values[(y * 16 + x) * 4..(y * 16 + x) * 4 + 4].copy_from_slice(&[
                    ramp[x],
                    ramp[8 - x],
                    0.25,
                    0.37,
                ]);
            }
        }
        let tensor = Tensor::<4>::from_data(
            burn::tensor::TensorData::new(values, [1, 2, 16, 4]),
            &device,
        );
        for output in [ColorOutput::SrgbRgbF32, ColorOutput::SrgbRgbaF32] {
            let actual = convert(tensor.clone(), 0, 9, output)
                .unwrap()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            for y in 0..2 {
                for x in 0..9 {
                    for c in 0..output.channels() {
                        let expected = if c == 3 {
                            0.37
                        } else {
                            bevy_zeroverse::render::color::linear_to_srgb(
                                [ramp[x], ramp[8 - x], 0.25][c],
                            )
                        };
                        assert!(
                            (actual[(y * 9 + x) * output.channels() + c] - expected).abs() < 2e-6,
                            "{x} {c}"
                        );
                    }
                }
            }
        }
        // A padded, multi-plane room arena must preserve camera/time ordering.
        let mut room = vec![99.; 4 * 2 * 2 * 16 * 4];
        for view in 0..4 {
            for y in 0..2 {
                for x in 0..3 {
                    let offset = (((view * 2 + 1) * 2 + y) * 16 + x) * 4;
                    room[offset..offset + 4].copy_from_slice(&[
                        view as f32 / 4.,
                        x as f32 / 3.,
                        y as f32 / 2.,
                        0.37,
                    ]);
                }
            }
        }
        let room =
            Tensor::<4>::from_data(burn::tensor::TensorData::new(room, [8, 2, 16, 4]), &device);
        let result = convert_room(room, 1, 3, 2, ColorOutput::SrgbRgbF32).unwrap();
        assert_eq!(result.dims(), [4, 2, 3, 3]);
        let result = result.into_data().try_to_vec::<f32>().unwrap();
        for view in 0..4 {
            for y in 0..2 {
                for x in 0..3 {
                    for c in 0..3 {
                        let expected = bevy_zeroverse::render::color::linear_to_srgb(
                            [view as f32 / 4., x as f32 / 3., y as f32 / 2.][c],
                        );
                        assert!((result[((view * 2 + y) * 3 + x) * 3 + c] - expected).abs() < 2e-6);
                    }
                }
            }
        }
        let bad = Tensor::<4>::from_data(
            burn::tensor::TensorData::new(
                vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5],
                [1, 1, 1, 4],
            ),
            &device,
        );
        let bad = convert(bad, 0, 1, ColorOutput::SrgbRgbaF32)
            .unwrap()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(bad[..3].iter().all(|v| !v.is_finite()));
        assert_eq!(bad[3], 0.5);
    }
}
