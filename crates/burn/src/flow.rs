//! Numeric temporal annotations never enter the image/color codec path.
use anyhow::{Result, ensure};
use bevy_zeroverse::sample::View;
use safetensors::{Dtype, SafeTensors};

use crate::{chunk::TensorData, dataset::ZeroverseSample};

pub(crate) fn validate(bytes: &[u8], pixels: usize) -> Result<()> {
    ensure!(
        bytes.len() == pixels * 16,
        "flow must contain RGBA float32 vectors/masks"
    );
    for pixel in bytes.as_chunks::<16>().0.iter() {
        let p: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
        ensure!(
            p.iter().all(|x| x.is_finite())
                && [0.0, 1.0].contains(&p[2])
                && [0.0, 1.0].contains(&p[3])
                && p[3] <= p[2]
                && (p[2] != 0.0 || (p[0] == 0.0 && p[1] == 0.0)),
            "invalid flow values or masks"
        );
    }
    Ok(())
}

fn plane<'a>(view: &'a View, name: &str) -> &'a [u8] {
    match name {
        "optical_flow" => &view.optical_flow,
        _ => &view.motion_vectors,
    }
}

pub(crate) fn encode(samples: &[ZeroverseSample], shape: [usize; 5]) -> Result<Vec<TensorData>> {
    let pixels = shape[3] * shape[4];
    let views: Vec<_> = samples.iter().flat_map(|s| &s.views).collect();
    let mut out = Vec::new();
    for name in ["optical_flow", "motion_vectors"] {
        let present = views.iter().any(|v| !plane(v, name).is_empty());
        if !present {
            continue;
        }
        let mut vectors = Vec::<f32>::with_capacity(views.len() * pixels * 2);
        let mut valid = Vec::with_capacity(views.len() * pixels);
        let mut visible = Vec::with_capacity(views.len() * pixels);
        for view in &views {
            let data = plane(view, name);
            validate(data, pixels)?;
            for pixel in data.as_chunks::<16>().0.iter() {
                let p: [f32; 4] = bytemuck::pod_read_unaligned(pixel);
                vectors.extend_from_slice(&p[..2]);
                valid.push(p[2] as u8);
                visible.push(p[3] as u8);
            }
        }
        let mut vector_shape = shape.to_vec();
        vector_shape.push(2);
        let mut mask_shape = shape.to_vec();
        mask_shape.push(1);
        out.push(TensorData::new(
            name,
            Dtype::F32,
            vector_shape,
            bytemuck::cast_slice(&vectors).to_vec(),
        ));
        out.push(TensorData::new(
            format!("{name}_valid"),
            Dtype::U8,
            mask_shape.clone(),
            valid,
        ));
        out.push(TensorData::new(
            format!("{name}_visible"),
            Dtype::U8,
            mask_shape,
            visible,
        ));
    }
    if !out.is_empty() {
        let metadata =
            serde_json::to_vec(&bevy_zeroverse::render::optical_flow::annotation_metadata())?;
        out.push(TensorData::new(
            "flow_metadata",
            Dtype::U8,
            vec![metadata.len()],
            metadata,
        ));
    }
    Ok(out)
}

pub(crate) fn decode(
    tensors: &SafeTensors<'_>,
    samples: &mut [ZeroverseSample],
    shape: [usize; 5],
) -> Result<()> {
    let pixels = shape[3] * shape[4];
    for name in ["optical_flow", "motion_vectors"] {
        let Ok(vectors) = tensors.tensor(name) else {
            continue;
        };
        let metadata = tensors.tensor("flow_metadata")?;
        let metadata: serde_json::Value = serde_json::from_slice(metadata.data())?;
        ensure!(
            metadata["schema_version"] == 1,
            "unknown numeric flow convention; legacy colored flow is not optical-flow ground truth"
        );
        let valid = tensors.tensor(&format!("{name}_valid"))?;
        let visible = tensors.tensor(&format!("{name}_visible"))?;
        let mut vector_shape = shape.to_vec();
        vector_shape.push(2);
        let mut mask_shape = shape.to_vec();
        mask_shape.push(1);
        ensure!(
            vectors.dtype() == Dtype::F32 && vectors.shape() == vector_shape,
            "{name} must be a two-channel f32 vector field; legacy RGB flow is a visualization"
        );
        ensure!(
            valid.dtype() == Dtype::U8
                && visible.dtype() == Dtype::U8
                && valid.shape() == mask_shape
                && visible.shape() == mask_shape,
            "flow masks must be one-channel u8 arrays"
        );
        for (i, view) in samples.iter_mut().flat_map(|s| &mut s.views).enumerate() {
            let mut rgba = Vec::with_capacity(pixels * 16);
            for p in i * pixels..(i + 1) * pixels {
                rgba.extend_from_slice(&vectors.data()[p * 8..p * 8 + 8]);
                rgba.extend_from_slice(bytemuck::bytes_of(&(valid.data()[p] as f32)));
                rgba.extend_from_slice(bytemuck::bytes_of(&(visible.data()[p] as f32)));
            }
            validate(&rgba, pixels)?;
            match name {
                "optical_flow" => view.optical_flow = rgba,
                _ => view.motion_vectors = rgba,
            }
        }
    }
    Ok(())
}
