//! Lossless U16 camera membership and U8 source validity, independent of RGB codecs.
use crate::{chunk::TensorData, dataset::ZeroverseSample};
use anyhow::{Context, Result, ensure};
use bevy_zeroverse::{
    render::co_visibility::{mask_color, validate_metadata, validate_plane},
    sample::View,
};
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, path::Path};

pub(crate) fn encode(samples: &[ZeroverseSample], shape: [usize; 5]) -> Result<Vec<TensorData>> {
    if samples
        .iter()
        .flat_map(|s| &s.views)
        .all(|v| v.co_visibility.is_empty())
    {
        return Ok(vec![]);
    }
    let pixels = shape[3] * shape[4];
    let mut masks = vec![];
    let mut valid = vec![];
    let mut out = vec![];
    for (s, sample) in samples.iter().enumerate() {
        let metadata = sample
            .co_visibility_metadata
            .as_ref()
            .context("missing co-visibility metadata")?;
        validate_metadata(metadata, shape[2]).map_err(anyhow::Error::msg)?;
        let json = serde_json::to_vec(metadata)?;
        out.push(TensorData::new(
            format!("co_visibility_metadata_{s}"),
            Dtype::U8,
            vec![json.len()],
            json,
        ));
        for (i, view) in sample.views.iter().enumerate() {
            let (m, v) = split(view, pixels, shape[2], i % shape[2])?;
            masks.extend(m);
            valid.extend(v);
        }
    }
    let mut tensor_shape = shape.to_vec();
    tensor_shape.push(1);
    out.push(TensorData::new(
        "co_visibility",
        Dtype::U16,
        tensor_shape.clone(),
        bytemuck::cast_slice(&masks).to_vec(),
    ));
    out.push(TensorData::new(
        "co_visibility_valid",
        Dtype::U8,
        tensor_shape,
        valid,
    ));
    Ok(out)
}

pub(crate) fn decode(
    tensors: &SafeTensors<'_>,
    samples: &mut [ZeroverseSample],
    shape: [usize; 5],
) -> Result<()> {
    let Ok(masks) = tensors.tensor("co_visibility") else {
        ensure!(
            tensors.tensor("co_visibility_valid").is_err(),
            "co-visibility validity without membership"
        );
        return Ok(());
    };
    let valid = tensors.tensor("co_visibility_valid")?;
    let mut expected = shape.to_vec();
    expected.push(1);
    ensure!(
        masks.dtype() == Dtype::U16
            && masks.shape() == expected
            && valid.dtype() == Dtype::U8
            && valid.shape() == expected,
        "invalid co-visibility tensors"
    );
    let pixels = shape[3] * shape[4];
    for (s, sample) in samples.iter_mut().enumerate() {
        let metadata = tensors.tensor(&format!("co_visibility_metadata_{s}"))?;
        ensure!(
            metadata.dtype() == Dtype::U8 && metadata.shape().len() == 1,
            "co-visibility metadata must be a U8 JSON byte vector"
        );
        let metadata: serde_json::Value = serde_json::from_slice(metadata.data())?;
        validate_metadata(&metadata, shape[2]).map_err(anyhow::Error::msg)?;
        sample.co_visibility_metadata = Some(metadata);
        for (i, view) in sample.views.iter_mut().enumerate() {
            let start = (s * shape[1] * shape[2] + i) * pixels;
            let masks: Vec<_> = masks.data()[start * 2..(start + pixels) * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect();
            view.co_visibility = join(
                &masks,
                &valid.data()[start..start + pixels],
                shape[2],
                i % shape[2],
            )?;
        }
    }
    Ok(())
}

fn split(view: &View, pixels: usize, count: usize, source: usize) -> Result<(Vec<u16>, Vec<u8>)> {
    validate_plane(&view.co_visibility, pixels, count, source).map_err(anyhow::Error::msg)?;
    Ok(view
        .co_visibility
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            let p: [f32; 4] = bytemuck::pod_read_unaligned(b);
            (p[0] as u16, p[2] as u8)
        })
        .unzip())
}
fn join(masks: &[u16], valid: &[u8], count: usize, source: usize) -> Result<Vec<u8>> {
    ensure!(
        masks.len() == valid.len(),
        "co-visibility mask/validity size mismatch"
    );
    let rgba: Vec<f32> = masks
        .iter()
        .zip(valid)
        .flat_map(|(&m, &v)| [m as f32, m.count_ones() as f32, v as f32, 0.0])
        .collect();
    let bytes = bytemuck::cast_slice(&rgba).to_vec();
    validate_plane(&bytes, masks.len(), count, source).map_err(anyhow::Error::msg)?;
    Ok(bytes)
}

pub(crate) fn save_view(
    dir: &Path,
    view: &View,
    time: usize,
    source: usize,
    size: [usize; 2],
    count: usize,
) -> Result<()> {
    use ndarray::Array3;
    let [height, width] = size;
    let (masks, valid) = split(view, height * width, count, source)?;
    let rgb: Vec<_> = masks.iter().flat_map(|m| mask_color(*m, count)).collect();
    let prefix = format!("co_visibility_{time:03}_{source:02}");
    let mut npz =
        ndarray_npy::NpzWriter::new_compressed(File::create(dir.join(format!("{prefix}.npz")))?);
    npz.add_array(
        "co_visibility",
        &Array3::from_shape_vec((height, width, 1), masks)?,
    )?;
    npz.add_array(
        "co_visibility_valid",
        &Array3::from_shape_vec((height, width, 1), valid)?,
    )?;
    npz.finish()?;
    image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .context("invalid co-visibility preview size")?
        .save(dir.join(format!("{prefix}.png")))?;
    Ok(())
}
pub(crate) fn load_view(
    dir: &Path,
    time: usize,
    source: usize,
    size: [usize; 2],
    count: usize,
) -> Result<Option<Vec<u8>>> {
    use ndarray::Array3;
    let path = dir.join(format!("co_visibility_{time:03}_{source:02}.npz"));
    if !path.exists() {
        return Ok(None);
    }
    let mut npz = ndarray_npy::NpzReader::new(File::open(path)?)?;
    let masks: Array3<u16> = npz.by_name("co_visibility")?;
    let valid: Array3<u8> = npz.by_name("co_visibility_valid")?;
    ensure!(
        masks.shape() == [size[0], size[1], 1] && masks.shape() == valid.shape(),
        "invalid co-visibility NPZ shape"
    );
    Ok(Some(join(
        masks.as_slice().context("non-contiguous co-visibility")?,
        valid.as_slice().context("non-contiguous validity")?,
        count,
        source,
    )?))
}
