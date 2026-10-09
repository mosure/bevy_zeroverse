use burn::tensor::Device;
use std::io::Cursor;

use burn::tensor::{Tensor, TensorData};
use image::{DynamicImage, ImageDecoder, ImageReader, Limits, RgbImage};

use crate::config::Siglip2Config;

/// Fixed preprocessing constants used by the published fixed-resolution SigLIP2 checkpoints.
pub const SIGLIP2_IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP2_IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP2_IMAGE_RESCALE_FACTOR: f32 = 1.0 / 255.0;

/// Maximum accepted size of an encoded image supplied to [`Siglip2ImageProcessor`].
///
/// This limits both native and browser callers to 64 MiB before a decoder is constructed.
pub const SIGLIP2_MAX_ENCODED_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum accepted width or height of a source image.
pub const SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION: u32 = 16_384;

/// Maximum accepted source image area (64 megapixels).
pub const SIGLIP2_MAX_SOURCE_IMAGE_PIXELS: u64 = 64 * 1024 * 1024;

/// Maximum aggregate allocation exposed to an `image` decoder (256 MiB).
///
/// The `image` crate treats allocation limits as best effort for decoders that cannot enforce
/// them, so dimensions and pixel count are independently preflighted before full decoding.
pub const SIGLIP2_MAX_DECODER_ALLOCATION_BYTES: u64 = 256 * 1024 * 1024;

/// CPU image processor matching the Hugging Face fixed-resolution SigLIP2 processor.
///
/// Encoded images are first transposed according to their EXIF orientation. Images are then
/// converted to RGB, resized directly to the configured square resolution with bilinear sampling
/// (`PIL.Image.Resampling.BILINEAR`, value `2`), rescaled by `1 / 255`, and normalized
/// channel-wise with mean/std `0.5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Siglip2ImageProcessor {
    image_size: usize,
}

impl Siglip2ImageProcessor {
    pub fn new(config: &Siglip2Config) -> Result<Self, String> {
        config.validate()?;
        if config.channels != 3 {
            return Err(format!(
                "fixed-resolution SigLIP2 preprocessing requires 3 RGB channels, got {}",
                config.channels
            ));
        }
        u32::try_from(config.image_size).map_err(|_| {
            format!(
                "SigLIP2 image_size {} exceeds the image decoder limit",
                config.image_size
            )
        })?;
        Ok(Self {
            image_size: config.image_size,
        })
    }

    pub const fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn preprocess_bytes(
        &self,
        encoded_image: &[u8],
        device: &Device,
    ) -> Result<Tensor<4>, String> {
        validate_encoded_image_len(encoded_image.len())?;

        // Inspect with a separate reader so declared dimensions and pixel count are rejected
        // before the decoder can allocate the full output image.
        let dimensions_reader = bounded_image_reader(encoded_image)?;
        let (width, height) = dimensions_reader
            .into_dimensions()
            .map_err(|err| format!("failed to inspect encoded image dimensions: {err}"))?;
        validate_source_image_dimensions(width, height, "encoded image")?;

        let mut decoder = bounded_image_reader(encoded_image)?
            .into_decoder()
            .map_err(|err| format!("failed to initialize image decoder: {err}"))?;
        let orientation = decoder
            .orientation()
            .map_err(|err| format!("failed to inspect encoded image orientation: {err}"))?;

        // `ImageReader::decode` reserves the output buffer from its allocation limit, but it
        // does not expose decoder metadata before consuming the decoder. Since orientation must
        // be read first, reproduce that reservation before handing the decoder to DynamicImage.
        let mut remaining_limits = bounded_image_limits();
        remaining_limits
            .reserve(decoder.total_bytes())
            .map_err(|err| format!("encoded image exceeds decoder allocation limits: {err}"))?;
        decoder
            .set_limits(remaining_limits)
            .map_err(|err| format!("failed to apply image decoder limits: {err}"))?;

        let mut image = DynamicImage::from_decoder(decoder)
            .map_err(|err| format!("failed to decode image bytes: {err}"))?;
        // Defend against a malformed decoder reporting different dimensions after decoding.
        validate_source_image_dimensions(image.width(), image.height(), "decoded image")?;
        image.apply_orientation(orientation);
        validate_source_image_dimensions(image.width(), image.height(), "oriented image")?;
        self.preprocess_image(&image, device)
    }

    pub fn preprocess_image(
        &self,
        image: &DynamicImage,
        device: &Device,
    ) -> Result<Tensor<4>, String> {
        self.preprocess_images(std::slice::from_ref(image), device)
    }

    pub fn preprocess_images(
        &self,
        images: &[DynamicImage],
        device: &Device,
    ) -> Result<Tensor<4>, String> {
        if images.is_empty() {
            return Err("SigLIP2 image batch must contain at least one image".to_string());
        }

        // Validate the complete batch before converting any element to RGB or allocating resize
        // intermediates. This keeps caller-provided DynamicImage values under the same policy as
        // encoded inputs.
        for (batch_index, image) in images.iter().enumerate() {
            validate_source_image_dimensions(
                image.width(),
                image.height(),
                &format!("image batch element {batch_index}"),
            )?;
        }

        let plane = self
            .image_size
            .checked_mul(self.image_size)
            .ok_or_else(|| "SigLIP2 image plane size overflow".to_string())?;
        let sample_len = 3usize
            .checked_mul(plane)
            .ok_or_else(|| "SigLIP2 image sample size overflow".to_string())?;
        let output_len = images
            .len()
            .checked_mul(sample_len)
            .ok_or_else(|| "SigLIP2 image batch size overflow".to_string())?;
        let side = u32::try_from(self.image_size)
            .map_err(|_| format!("invalid SigLIP2 image size {}", self.image_size))?;
        let mut output = vec![0.0f32; output_len];

        for (batch_index, image) in images.iter().enumerate() {
            let rgb = image.to_rgb8();
            let resized = pillow_bilinear_resize(&rgb, side, side)?;
            let batch_offset = batch_index * sample_len;
            for y in 0..self.image_size {
                for x in 0..self.image_size {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let spatial_index = y * self.image_size + x;
                    for channel in 0..3 {
                        let rescaled = f32::from(pixel[channel]) * SIGLIP2_IMAGE_RESCALE_FACTOR;
                        output[batch_offset + channel * plane + spatial_index] =
                            (rescaled - SIGLIP2_IMAGE_MEAN[channel]) / SIGLIP2_IMAGE_STD[channel];
                    }
                }
            }
        }

        Ok(Tensor::<4>::from_data(
            TensorData::new(output, [images.len(), 3, self.image_size, self.image_size]),
            device,
        ))
    }
}

fn validate_encoded_image_len(encoded_len: usize) -> Result<(), String> {
    if encoded_len == 0 {
        return Err("cannot preprocess empty image bytes".to_string());
    }
    if encoded_len > SIGLIP2_MAX_ENCODED_IMAGE_BYTES {
        return Err(format!(
            "encoded image is {encoded_len} bytes, exceeding the {}-byte limit",
            SIGLIP2_MAX_ENCODED_IMAGE_BYTES
        ));
    }
    Ok(())
}

fn bounded_image_reader(encoded_image: &[u8]) -> Result<ImageReader<Cursor<&[u8]>>, String> {
    let mut reader = ImageReader::new(Cursor::new(encoded_image))
        .with_guessed_format()
        .map_err(|err| format!("failed to inspect encoded image format: {err}"))?;
    reader.limits(bounded_image_limits());
    Ok(reader)
}

fn bounded_image_limits() -> Limits {
    let mut limits = Limits::default();
    limits.max_image_width = Some(SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION);
    limits.max_image_height = Some(SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION);
    limits.max_alloc = Some(SIGLIP2_MAX_DECODER_ALLOCATION_BYTES);
    limits
}

fn validate_source_image_dimensions(width: u32, height: u32, context: &str) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err(format!(
            "{context} dimensions must be non-zero, got {width}x{height}"
        ));
    }
    if width > SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION || height > SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION {
        return Err(format!(
            "{context} dimensions {width}x{height} exceed the {}-pixel per-dimension limit",
            SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION
        ));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > SIGLIP2_MAX_SOURCE_IMAGE_PIXELS {
        return Err(format!(
            "{context} has {pixels} pixels, exceeding the {}-pixel limit",
            SIGLIP2_MAX_SOURCE_IMAGE_PIXELS
        ));
    }
    Ok(())
}

/// Resize an 8-bit RGB image with Pillow's separable BILINEAR algorithm.
///
/// Hugging Face's slow SigLIP2 image processor resizes through Pillow. The `image` crate's
/// `Triangle` filter uses the same sampling geometry, but its coefficient quantization and
/// rounding differ by one byte for many pixels. Those one-byte differences measurably amplify
/// through a vision transformer, so inference preprocessing intentionally mirrors Pillow's
/// 22-bit fixed-point coefficient path.
fn pillow_bilinear_resize(
    input: &RgbImage,
    output_width: u32,
    output_height: u32,
) -> Result<RgbImage, String> {
    if output_width == 0 || output_height == 0 {
        return Err("SigLIP2 resize dimensions must be non-zero".to_string());
    }
    if input.width() == 0 || input.height() == 0 {
        return Err("SigLIP2 cannot resize an empty image".to_string());
    }
    validate_source_image_dimensions(input.width(), input.height(), "resize input")?;
    validate_source_image_dimensions(output_width, output_height, "resize output")?;
    if input.width() == output_width && input.height() == output_height {
        return Ok(input.clone());
    }

    let input_width = usize::try_from(input.width())
        .map_err(|_| "SigLIP2 input width exceeds host limits".to_string())?;
    let input_height = usize::try_from(input.height())
        .map_err(|_| "SigLIP2 input height exceeds host limits".to_string())?;
    let output_width = usize::try_from(output_width)
        .map_err(|_| "SigLIP2 output width exceeds host limits".to_string())?;
    let output_height = usize::try_from(output_height)
        .map_err(|_| "SigLIP2 output height exceeds host limits".to_string())?;
    let horizontal = pillow_bilinear_coefficients(input_width, output_width)?;
    let vertical = pillow_bilinear_coefficients(input_height, output_height)?;

    let horizontal_len = output_width
        .checked_mul(input_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "SigLIP2 horizontal resize buffer overflow".to_string())?;
    let mut intermediate = vec![0u8; horizontal_len];
    let input_bytes = input.as_raw();
    for y in 0..input_height {
        for (x, coefficients) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let mut sum = PILLOW_FIXED_POINT_HALF;
                for (offset, coefficient) in coefficients.weights.iter().enumerate() {
                    let source = input_bytes
                        [((y * input_width + coefficients.start + offset) * 3) + channel];
                    sum += i64::from(source) * i64::from(*coefficient);
                }
                intermediate[(y * output_width + x) * 3 + channel] = pillow_clip_u8(sum);
            }
        }
    }

    let output_len = output_width
        .checked_mul(output_height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "SigLIP2 output resize buffer overflow".to_string())?;
    let mut output = vec![0u8; output_len];
    for (y, coefficients) in vertical.iter().enumerate() {
        for x in 0..output_width {
            for channel in 0..3 {
                let mut sum = PILLOW_FIXED_POINT_HALF;
                for (offset, coefficient) in coefficients.weights.iter().enumerate() {
                    let source = intermediate
                        [((coefficients.start + offset) * output_width + x) * 3 + channel];
                    sum += i64::from(source) * i64::from(*coefficient);
                }
                output[(y * output_width + x) * 3 + channel] = pillow_clip_u8(sum);
            }
        }
    }

    RgbImage::from_raw(output_width as u32, output_height as u32, output)
        .ok_or_else(|| "failed to construct resized SigLIP2 image".to_string())
}

const PILLOW_PRECISION_BITS: u32 = 22;
const PILLOW_FIXED_POINT_SCALE: f64 = (1u64 << PILLOW_PRECISION_BITS) as f64;
const PILLOW_FIXED_POINT_HALF: i64 = 1i64 << (PILLOW_PRECISION_BITS - 1);

#[derive(Debug)]
struct PillowCoefficients {
    start: usize,
    weights: Vec<i32>,
}

fn pillow_bilinear_coefficients(
    input_size: usize,
    output_size: usize,
) -> Result<Vec<PillowCoefficients>, String> {
    if input_size == 0 || output_size == 0 {
        return Err("Pillow resize dimensions must be non-zero".to_string());
    }
    let scale = input_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = filter_scale;
    let mut output = Vec::with_capacity(output_size);

    for destination in 0..output_size {
        let center = (destination as f64 + 0.5) * scale;
        // Pillow uses a C cast here, which truncates toward zero rather than flooring.
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input_size);
        if end <= start {
            return Err(format!(
                "Pillow bilinear resize produced an empty source window at output {destination}"
            ));
        }

        let mut floating = Vec::with_capacity(end - start);
        let mut total = 0.0f64;
        for source in start..end {
            let distance = ((source as f64 - center + 0.5) / filter_scale).abs();
            let weight = (1.0 - distance).max(0.0);
            floating.push(weight);
            total += weight;
        }
        if total == 0.0 {
            return Err(format!(
                "Pillow bilinear resize produced zero filter weight at output {destination}"
            ));
        }
        let weights = floating
            .into_iter()
            .map(|weight| ((weight / total) * PILLOW_FIXED_POINT_SCALE + 0.5) as i32)
            .collect();
        output.push(PillowCoefficients { start, weights });
    }
    Ok(output)
}

#[inline]
fn pillow_clip_u8(value: i64) -> u8 {
    (value >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

pub fn preprocess_image_bytes(
    encoded_image: &[u8],
    config: &Siglip2Config,
    device: &Device,
) -> Result<Tensor<4>, String> {
    Siglip2ImageProcessor::new(config)?.preprocess_bytes(encoded_image, device)
}

pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    config: &Siglip2Config,
    device: &Device,
) -> Result<Tensor<4>, String> {
    Siglip2ImageProcessor::new(config)?.preprocess_image(image, device)
}

pub fn preprocess_dynamic_images(
    images: &[DynamicImage],
    config: &Siglip2Config,
    device: &Device,
) -> Result<Tensor<4>, String> {
    Siglip2ImageProcessor::new(config)?.preprocess_images(images, device)
}

#[cfg(all(test, any(feature = "ndarray", feature = "flex")))]
mod tests {
    use std::io::Cursor;

    use image::{
        DynamicImage, ExtendedColorType, GrayImage, ImageEncoder, ImageFormat, Luma, Rgb, RgbImage,
        codecs::{jpeg::JpegEncoder, webp::WebPEncoder},
    };

    use super::{
        SIGLIP2_IMAGE_RESCALE_FACTOR, SIGLIP2_MAX_ENCODED_IMAGE_BYTES,
        SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION, Siglip2ImageProcessor, validate_encoded_image_len,
        validate_source_image_dimensions,
    };
    use crate::Siglip2Config;

    fn config(image_size: usize) -> Siglip2Config {
        Siglip2Config {
            image_size,
            patch_size: 1,
            ..Siglip2Config::tiny_for_tests()
        }
    }

    fn bmp_header(width: u32, height: u32) -> [u8; 54] {
        let mut bytes = [0u8; 54];
        bytes[0..2].copy_from_slice(b"BM");
        bytes[2..6].copy_from_slice(&54u32.to_le_bytes());
        bytes[10..14].copy_from_slice(&54u32.to_le_bytes());
        bytes[14..18].copy_from_slice(&40u32.to_le_bytes());
        bytes[18..22].copy_from_slice(&(width as i32).to_le_bytes());
        bytes[22..26].copy_from_slice(&(height as i32).to_le_bytes());
        bytes[26..28].copy_from_slice(&1u16.to_le_bytes());
        bytes[28..30].copy_from_slice(&24u16.to_le_bytes());
        bytes
    }

    fn orientation_exif(orientation: u16) -> Vec<u8> {
        // Minimal little-endian TIFF payload with one IFD0 Orientation (0x0112) SHORT entry.
        let mut exif = Vec::with_capacity(26);
        exif.extend_from_slice(b"II\x2a\0");
        exif.extend_from_slice(&8u32.to_le_bytes());
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&0x0112u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&0u16.to_le_bytes());
        exif.extend_from_slice(&0u32.to_le_bytes());
        exif
    }

    fn encode_oriented_jpeg(image: &RgbImage, orientation: u16) -> Result<Vec<u8>, String> {
        let mut encoded = Vec::new();
        let mut encoder = JpegEncoder::new_with_quality(&mut encoded, 100);
        encoder
            .set_exif_metadata(orientation_exif(orientation))
            .map_err(|err| format!("failed to set JPEG EXIF fixture: {err}"))?;
        encoder
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                ExtendedColorType::Rgb8,
            )
            .map_err(|err| format!("failed to encode JPEG fixture: {err}"))?;
        Ok(encoded)
    }

    fn encode_oriented_webp(image: &RgbImage, orientation: u16) -> Result<Vec<u8>, String> {
        let mut encoded = Vec::new();
        let mut encoder = WebPEncoder::new_lossless(&mut encoded);
        encoder
            .set_exif_metadata(orientation_exif(orientation))
            .map_err(|err| format!("failed to set WebP EXIF fixture: {err}"))?;
        encoder
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                ExtendedColorType::Rgb8,
            )
            .map_err(|err| format!("failed to encode WebP fixture: {err}"))?;
        Ok(encoded)
    }

    fn tensor_values(tensor: burn::tensor::Tensor<4>) -> Result<Vec<f32>, String> {
        tensor
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read test tensor: {err:?}"))
    }

    #[test]
    fn preprocesses_rgb_as_normalized_nchw() -> Result<(), String> {
        let image = RgbImage::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgb([0, 128, 255]),
            (1, 0) => Rgb([255, 0, 128]),
            (0, 1) => Rgb([128, 255, 0]),
            _ => Rgb([64, 192, 32]),
        });
        let processor = Siglip2ImageProcessor::new(&config(2))?;
        let device = burn::tensor::Device::flex();
        let tensor = processor.preprocess_image(&DynamicImage::ImageRgb8(image), &device)?;
        assert_eq!(tensor.shape().dims::<4>(), [1, 3, 2, 2]);
        let values = tensor
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read test tensor: {err:?}"))?;

        let normalized = |value: u8| f32::from(value) * SIGLIP2_IMAGE_RESCALE_FACTOR * 2.0 - 1.0;
        let expected = vec![
            normalized(0),
            normalized(255),
            normalized(128),
            normalized(64),
            normalized(128),
            normalized(0),
            normalized(255),
            normalized(192),
            normalized(255),
            normalized(128),
            normalized(0),
            normalized(32),
        ];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn converts_grayscale_to_three_rgb_channels() -> Result<(), String> {
        let image = DynamicImage::ImageLuma8(GrayImage::from_pixel(1, 1, Luma([255])));
        let processor = Siglip2ImageProcessor::new(&config(1))?;
        let device = burn::tensor::Device::flex();
        let values = processor
            .preprocess_image(&image, &device)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read test tensor: {err:?}"))?;
        assert_eq!(values, vec![1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn bilinear_resize_matches_pillow_reference_pixels() -> Result<(), String> {
        let image = RgbImage::from_raw(
            3,
            2,
            vec![
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
            ],
        )
        .ok_or_else(|| "invalid RGB fixture".to_string())?;
        let processor = Siglip2ImageProcessor::new(&config(2))?;
        let device = burn::tensor::Device::flex();
        let values = processor
            .preprocess_image(&DynamicImage::ImageRgb8(image), &device)?
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|err| format!("failed to read test tensor: {err:?}"))?;

        // Pillow 12.1.1 `Image.resize((2, 2), Resampling.BILINEAR)` yields these RGB
        // pixels. This locks the resize coordinate convention as well as the filter selection.
        let pillow_pixels = [
            [11u8, 21, 31],
            [49, 59, 69],
            [101, 111, 121],
            [139, 149, 159],
        ];
        for channel in 0..3 {
            for (spatial, pixel) in pillow_pixels.iter().enumerate() {
                let expected = f32::from(pixel[channel]) * SIGLIP2_IMAGE_RESCALE_FACTOR * 2.0 - 1.0;
                let actual = values[channel * 4 + spatial];
                assert!(
                    (actual - expected).abs() <= 1.0e-6,
                    "{actual} != {expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn preprocesses_bounded_encoded_images() -> Result<(), String> {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([10, 20, 30])));
        let mut encoded = Cursor::new(Vec::new());
        image
            .write_to(&mut encoded, ImageFormat::Png)
            .map_err(|err| format!("failed to encode fixture: {err}"))?;

        let processor = Siglip2ImageProcessor::new(&config(2))?;
        let device = burn::tensor::Device::flex();
        let tensor = processor.preprocess_bytes(encoded.get_ref(), &device)?;
        assert_eq!(tensor.shape().dims::<4>(), [1, 3, 2, 2]);
        Ok(())
    }

    #[test]
    fn applies_encoded_exif_orientation_once_before_preprocessing() -> Result<(), String> {
        let source = RgbImage::from_fn(16, 8, |x, y| {
            Rgb([
                (x * 13 + y * 3) as u8,
                (x * 5 + y * 23) as u8,
                (x * 17 + y * 11) as u8,
            ])
        });
        let fixtures = [
            ("JPEG", encode_oriented_jpeg(&source, 6)?),
            ("WebP", encode_oriented_webp(&source, 6)?),
        ];
        let processor = Siglip2ImageProcessor::new(&config(8))?;
        let device = burn::tensor::Device::flex();

        for (format, encoded) in fixtures {
            // The ordinary image decode path intentionally leaves EXIF orientation unapplied,
            // providing the same decoded pixels for the manual one-rotation reference.
            let decoded = image::load_from_memory(&encoded)
                .map_err(|err| format!("failed to decode {format} fixture: {err}"))?;
            assert_eq!((decoded.width(), decoded.height()), (16, 8));
            let oriented_once = decoded.rotate90();
            assert_eq!((oriented_once.width(), oriented_once.height()), (8, 16));

            let actual = tensor_values(processor.preprocess_bytes(&encoded, &device)?)?;
            let expected = tensor_values(processor.preprocess_image(&oriented_once, &device)?)?;
            let not_oriented = tensor_values(processor.preprocess_image(&decoded, &device)?)?;
            let oriented_twice =
                tensor_values(processor.preprocess_image(&oriented_once.rotate90(), &device)?)?;

            assert_eq!(actual, expected, "{format} EXIF orientation parity");
            assert_ne!(actual, not_oriented, "{format} orientation was not applied");
            assert_ne!(
                actual, oriented_twice,
                "{format} orientation was applied more than once"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_oversized_encoded_length_without_allocating_a_fixture() {
        let error = validate_encoded_image_len(SIGLIP2_MAX_ENCODED_IMAGE_BYTES + 1)
            .expect_err("oversized encoded payload must fail");
        assert!(error.contains("exceeding"), "{error}");
    }

    #[test]
    fn rejects_dimension_and_pixel_bombs_from_tiny_headers() {
        let device = burn::tensor::Device::flex();
        let processor = Siglip2ImageProcessor::new(&config(2)).expect("processor");

        let dimension_bomb = bmp_header(SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION + 1, 1);
        let error = processor
            .preprocess_bytes(&dimension_bomb, &device)
            .expect_err("oversized dimension must fail during preflight");
        assert!(error.contains("dimensions"), "{error}");

        // 8192 * 8193 is just above the 64-megapixel area limit, while each dimension
        // remains below the strict per-axis limit. Only a 54-byte header is needed.
        let pixel_bomb = bmp_header(8192, 8193);
        let error = processor
            .preprocess_bytes(&pixel_bomb, &device)
            .expect_err("oversized pixel count must fail during preflight");
        assert!(
            error.contains("pixels") && error.contains("exceeding"),
            "{error}"
        );
    }

    #[test]
    fn rejects_extreme_dynamic_image_before_rgb_conversion() {
        // This image is only about 49 KiB despite its hostile aspect ratio.
        let image =
            DynamicImage::ImageRgb8(RgbImage::new(SIGLIP2_MAX_SOURCE_IMAGE_DIMENSION + 1, 1));
        let device = burn::tensor::Device::flex();
        let processor = Siglip2ImageProcessor::new(&config(2)).expect("processor");
        let error = processor
            .preprocess_image(&image, &device)
            .expect_err("extreme DynamicImage must fail before conversion");
        assert!(error.contains("per-dimension limit"), "{error}");

        let error = validate_source_image_dimensions(8192, 8193, "test image")
            .expect_err("pixel area limit must be enforced without allocating an image");
        assert!(error.contains("pixels"), "{error}");
    }

    #[test]
    fn rejects_empty_batches_and_non_rgb_configs() {
        let device = burn::tensor::Device::flex();
        let processor = Siglip2ImageProcessor::new(&config(2)).expect("processor");
        assert!(
            processor
                .preprocess_images(&[], &device)
                .expect_err("empty batch should fail")
                .contains("at least one")
        );

        let mut invalid = config(2);
        invalid.channels = 1;
        assert!(
            Siglip2ImageProcessor::new(&invalid)
                .expect_err("non-RGB config should fail")
                .contains("3 RGB channels")
        );
    }
}
