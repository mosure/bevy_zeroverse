use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use bytemuck::cast_slice;
use safetensors::{Dtype, SafeTensors, serialize, tensor::TensorView};
use serde_json;

use crate::{compression::Compression, dataset::ZeroverseSample};


pub(crate) fn decode_rgba_bytes(bytes: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    let pixels = width as usize * height as usize;
    let expected_u8 = pixels * 4;
    let expected_f32 = expected_u8 * 4;

    if bytes.len() == expected_f32 {
        let floats: &[f32] = cast_slice(bytes);
        return Ok(floats.to_vec());
    }

    if bytes.len() == expected_u8 {
        return Ok(bytes.iter().map(|b| *b as f32 / 255.0).collect());
    }

    let height_usize = height as usize;
    if height_usize > 0 && bytes.len().checked_rem(height_usize) == Some(0) {
        let row_bytes = bytes.len() / height_usize;

        // Handle pitched float rows (e.g. GPU copies aligned to 256 bytes).
        let float_row = width as usize * 4 * 4;
        if row_bytes >= float_row && row_bytes.checked_rem(4) == Some(0) {
            let mut out = Vec::with_capacity(pixels * 4);
            for row in 0..height_usize {
                let start = row * row_bytes;
                let end = start + float_row;
                let slice = &bytes[start..end];
                let floats: &[f32] = cast_slice(slice);
                out.extend_from_slice(floats);
            }
            if out.len() == pixels * 4 {
                return Ok(out);
            }
        }

        // Handle pitched u8 rows.
        let u8_row = width as usize * 4;
        if row_bytes >= u8_row {
            let mut out = Vec::with_capacity(pixels * 4);
            for row in 0..height_usize {
                let start = row * row_bytes;
                let end = start + u8_row;
                let slice = &bytes[start..end];
                out.extend(slice.iter().map(|b| *b as f32 / 255.0));
            }
            if out.len() == pixels * 4 {
                return Ok(out);
            }
        }
    }

    Err(anyhow!(
        "unexpected rgba byte length {} for {}x{}",
        bytes.len(),
        width,
        height
    ))
}


pub(crate) fn encode_jpeg_from_rgba_f32(rgba: &[f32], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for chunk in rgba.chunks_exact(4) {
        for c in chunk.iter().take(3) {
            let v = c.clamp(0.0, 1.0);
            rgb.push((v * 255.0).round().clamp(0.0, 255.0) as u8);
        }
    }
    let image =
        image::RgbImage::from_raw(width, height, rgb).context("failed to build rgb image")?;
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 75);
    encoder.encode_image(&image)?;
    Ok(buf)
}


pub(crate) fn decode_jpeg_to_rgba_f32(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>)> {
    let dyn_img = image::load_from_memory(bytes)?;
    let rgb = dyn_img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for chunk in rgb.into_raw().chunks_exact(3) {
        for c in chunk {
            out.push(*c as f32 / 255.0);
        }
        out.push(1.0); // alpha
    }
    Ok((width, height, out))
}


pub(crate) fn leak_bytes(buf: Vec<u8>) -> &'static [u8] {
    Box::leak(buf.into_boxed_slice())
}


pub(crate) fn normalize_hdr_image_tonemap(
    data: &mut [f32],
    steps: usize,
    view_dim: usize,
    height: usize,
    width: usize,
    channels: usize,
) {
    if data.is_empty() || channels == 0 {
        return;
    }

    let stride_view = height * width * channels;
    let stride_step = view_dim * stride_view;

    for v in data.iter_mut() {
        *v = *v / (1.0 + *v);
    }

    let mut min_vals = vec![f32::INFINITY; steps * channels];
    let mut max_vals = vec![f32::NEG_INFINITY; steps * channels];

    for t in 0..steps {
        for v in 0..view_dim {
            let base = t * stride_step + v * stride_view;
            let slice = &data[base..base + stride_view];
            for i in 0..(height * width) {
                for c in 0..channels {
                    let val = slice[i * channels + c];
                    let idx = t * channels + c;
                    if val < min_vals[idx] {
                        min_vals[idx] = val;
                    }
                    if val > max_vals[idx] {
                        max_vals[idx] = val;
                    }
                }
            }
        }
    }

    for t in 0..steps {
        for v in 0..view_dim {
            let base = t * stride_step + v * stride_view;
            let slice = &mut data[base..base + stride_view];
            for i in 0..(height * width) {
                for c in 0..channels {
                    let idx = t * channels + c;
                    let range = (max_vals[idx] - min_vals[idx]).max(1e-8);
                    let val = slice[i * channels + c];
                    slice[i * channels + c] = (val - min_vals[idx]) / range;
                }
            }
        }
    }
}


pub fn save_chunk(
    samples: &[ZeroverseSample],
    output_dir: impl AsRef<Path>,
    chunk_index: usize,
    compression: Compression,
    width: u32,
    height: u32,
) -> Result<PathBuf> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let file_name = format!("{chunk_index:06}.{}", compression.extension());
    let path = output_dir.join(file_name);

    let b = samples.len();
    let view_dim = samples
        .first()
        .map(|s| s.view_dim as usize)
        .unwrap_or(0)
        .max(1);
    let steps = samples
        .first()
        .map(|s| s.views.len() / view_dim)
        .unwrap_or(0)
        .max(1);

    let color_shape: [i64; 6] = [b as i64, steps as i64, view_dim as i64, height as i64, width as i64, 3];

    let mut tensors: Vec<(&'static str, TensorView<'static>)> = Vec::new();
    let mut color_entries: Vec<(String, TensorView<'static>)> = Vec::new();

    let mut depth = Vec::new();
    let mut normal = Vec::new();
    let mut optical_flow = Vec::new();
    let mut position = Vec::new();
    let mut world_from_view: Vec<f32> = Vec::new();
    let mut fovy = Vec::new();
    let mut near = Vec::new();
    let mut far = Vec::new();
    let mut time = Vec::new();
    let mut aabb: Vec<f32> = Vec::new();

    // Object OBB accumulation
    let mut class_to_idx: HashMap<String, i64> = HashMap::new();
    let mut class_names: Vec<String> = Vec::new();
    let mut max_obbs = 0usize;
    for sample in samples.iter() {
        max_obbs = max_obbs.max(sample.object_obbs.len());
        for obb in &sample.object_obbs {
            if !class_to_idx.contains_key(&obb.class_name) {
                let idx = class_to_idx.len() as i64;
                class_to_idx.insert(obb.class_name.clone(), idx);
                class_names.push(obb.class_name.clone());
            }
        }
    }

    let mut obb_center: Vec<f32> = vec![0.0; b * max_obbs * 3];
    let mut obb_scale: Vec<f32> = vec![0.0; b * max_obbs * 3];
    let mut obb_rotation: Vec<f32> = vec![0.0; b * max_obbs * 4];
    let mut obb_class_idx: Vec<i64> = vec![-1; b * max_obbs];

    for (sample_idx, sample) in samples.iter().enumerate() {
        assert_eq!(
            sample.views.len(),
            view_dim * steps,
            "sample views len mismatch expected view_dim*steps"
        );

        let mut sample_color: Option<Vec<f32>> = None;
        let mut sample_optical_flow: Option<Vec<f32>> = None;

        for (local_idx, view) in sample.views.iter().enumerate() {
            let t = local_idx / view_dim;
            let v = local_idx % view_dim;

            if !view.color.is_empty() {
                let rgba_f32 = decode_rgba_bytes(&view.color, width, height)
                    .context("failed to parse color")?;
                let pixel_count = (height * width) as usize;
                let color_buf = sample_color
                    .get_or_insert_with(|| vec![0.0; steps * view_dim * pixel_count * 3]);
                let base = (t * view_dim + v) * pixel_count * 3;
                for i in 0..pixel_count {
                    let src = i * 4;
                    let dst = base + i * 3;
                    color_buf[dst] = rgba_f32[src];
                    color_buf[dst + 1] = rgba_f32[src + 1];
                    color_buf[dst + 2] = rgba_f32[src + 2];
                }
            }

            if !view.depth.is_empty() {
                let rgba = decode_rgba_bytes(&view.depth, width, height)
                    .context("failed to parse depth")?;
                depth.extend(rgba.chunks_exact(4).map(|c| c[0]));
            }

            if !view.normal.is_empty() {
                let rgba = decode_rgba_bytes(&view.normal, width, height)
                    .context("failed to parse normal")?;
                for chunk in rgba.chunks_exact(4) {
                    normal.extend_from_slice(&chunk[..3]);
                }
            }

            if !view.optical_flow.is_empty() {
                let rgba = decode_rgba_bytes(&view.optical_flow, width, height)
                    .context("failed to parse optical flow")?;
                let pixel_count = (height * width) as usize;
                let flow_buf = sample_optical_flow
                    .get_or_insert_with(|| vec![0.0; steps * view_dim * pixel_count * 3]);
                let base = (t * view_dim + v) * pixel_count * 3;
                for i in 0..pixel_count {
                    let src = i * 4;
                    let dst = base + i * 3;
                    flow_buf[dst] = rgba[src];
                    flow_buf[dst + 1] = rgba[src + 1];
                    flow_buf[dst + 2] = rgba[src + 2];
                }
            }

            if !view.position.is_empty() {
                let rgba = decode_rgba_bytes(&view.position, width, height)
                    .context("failed to parse position")?;
                for chunk in rgba.chunks_exact(4) {
                    position.extend_from_slice(&chunk[..3]);
                }
            }

            world_from_view.extend_from_slice(cast_slice(&view.world_from_view));
            fovy.push(view.fovy);
            near.push(view.near);
            far.push(view.far);
            time.push(view.time);
        }

        if let Some(mut color) = sample_color {
            normalize_hdr_image_tonemap(
                &mut color,
                steps,
                view_dim,
                height as usize,
                width as usize,
                3,
            );

            let pixel_count = (height * width) as usize;
            for local_idx in 0..(view_dim * steps) {
                let base = local_idx * pixel_count * 3;
                let mut rgba = Vec::with_capacity(pixel_count * 4);
                for i in 0..pixel_count {
                    let src = base + i * 3;
                    rgba.extend_from_slice(&[color[src], color[src + 1], color[src + 2], 1.0]);
                }

                let global_idx = sample_idx * view_dim * steps + local_idx;
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                let data_ref = leak_bytes(jpg);
                color_entries.push((
                    format!("color_jpg_{global_idx}"),
                    TensorView::new(Dtype::U8, vec![data_ref.len()], data_ref)?,
                ));
            }
        }

        if let Some(mut flow) = sample_optical_flow {
            normalize_hdr_image_tonemap(
                &mut flow,
                steps,
                view_dim,
                height as usize,
                width as usize,
                3,
            );
            optical_flow.extend_from_slice(&flow);
        }

        aabb.extend_from_slice(cast_slice(&sample.aabb));

        if max_obbs > 0 {
            for (local_idx, obb) in sample.object_obbs.iter().take(max_obbs).enumerate() {
                let base = sample_idx * max_obbs + local_idx;
                let c_base = base * 3;
                let r_base = base * 4;

                obb_center[c_base..c_base + 3].copy_from_slice(&obb.center);
                obb_scale[c_base..c_base + 3].copy_from_slice(&obb.scale);
                obb_rotation[r_base..r_base + 4].copy_from_slice(&obb.rotation);

                if let Some(idx) = class_to_idx.get(&obb.class_name) {
                    obb_class_idx[base] = *idx;
                }
            }
        }
    }

    let color_shape_ref = leak_bytes(cast_slice(&color_shape).to_vec());
    tensors.push((
        "color_shape",
        TensorView::new(Dtype::I64, vec![6], color_shape_ref)?,
    ));

    let push_tensor = |name: &'static str,
                       data: Vec<f32>,
                       channels: usize,
                       tensors: &mut Vec<(&'static str, TensorView<'static>)>|
     -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let shape = [
            b,
            steps,
            view_dim,
            height as usize,
            width as usize,
            channels,
        ]
        .to_vec();
        let data_ref = leak_bytes(cast_slice(&data).to_vec());
        tensors.push((name, TensorView::new(Dtype::F32, shape, data_ref)?));
        Ok(())
    };

    push_tensor("depth", depth, 1, &mut tensors)?;
    push_tensor("normal", normal, 3, &mut tensors)?;
    push_tensor("optical_flow", optical_flow, 3, &mut tensors)?;
    push_tensor("position", position, 3, &mut tensors)?;
    if !aabb.is_empty() {
        let data_ref = leak_bytes(cast_slice(&aabb).to_vec());
        tensors.push((
            "aabb",
            TensorView::new(Dtype::F32, vec![b, 2, 3], data_ref)?,
        ));
    }

    if max_obbs > 0 {
        let center_ref = leak_bytes(cast_slice(&obb_center).to_vec());
        let scale_ref = leak_bytes(cast_slice(&obb_scale).to_vec());
        let rotation_ref = leak_bytes(cast_slice(&obb_rotation).to_vec());
        let class_ref = leak_bytes(cast_slice(&obb_class_idx).to_vec());
        let class_bytes = serde_json::to_vec(&class_names)?;
        let class_bytes_ref = leak_bytes(class_bytes);

        tensors.push((
            "object_obb_center",
            TensorView::new(Dtype::F32, vec![b, max_obbs, 3], center_ref)?,
        ));
        tensors.push((
            "object_obb_scale",
            TensorView::new(Dtype::F32, vec![b, max_obbs, 3], scale_ref)?,
        ));
        tensors.push((
            "object_obb_rotation",
            TensorView::new(Dtype::F32, vec![b, max_obbs, 4], rotation_ref)?,
        ));
        tensors.push((
            "object_obb_class_idx",
            TensorView::new(Dtype::I64, vec![b, max_obbs], class_ref)?,
        ));
        tensors.push((
            "object_obb_class_names",
            TensorView::new(Dtype::U8, vec![class_bytes_ref.len()], class_bytes_ref)?,
        ));
    }

    if !world_from_view.is_empty() {
        let shape = [b, steps, view_dim, 4, 4].to_vec();
        let data_ref = leak_bytes(cast_slice(&world_from_view).to_vec());
        tensors.push((
            "world_from_view",
            TensorView::new(Dtype::F32, shape, data_ref)?,
        ));
    }

    for (name, data) in [
        ("fovy", &fovy),
        ("near", &near),
        ("far", &far),
        ("time", &time),
    ] {
        if data.is_empty() {
            continue;
        }
        let shape = [b, steps, view_dim, 1].to_vec();
        let data_ref = leak_bytes(cast_slice(data).to_vec());
        tensors.push((name, TensorView::new(Dtype::F32, shape, data_ref)?));
    }

    for (name, view) in color_entries {
        let leaked: &'static str = Box::leak(name.into_boxed_str());
        tensors.push((leaked, view));
    }

    let serialized = serialize(tensors, None)?;
    let compressed = compression.compress(&serialized)?;
    fs::write(&path, compressed)?;
    Ok(path)
}


pub fn load_chunk(path: impl AsRef<Path>) -> Result<Vec<ZeroverseSample>> {
    let path = path.as_ref();
    let data = fs::read(path).with_context(|| format!("failed to read chunk {:?}", path))?;
    let compression = path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| ext.split('.').next_back())
        .map(Compression::from_extension)
        .unwrap_or_default();

    let decompressed = compression
        .decompress(&data)
        .with_context(|| format!("failed to decompress chunk {:?}", path))?;

    let tensors = SafeTensors::deserialize(&decompressed)?;

    let mut class_names: Vec<String> = Vec::new();
    if let Ok(class_tensor) = tensors.tensor("object_obb_class_names") {
        let bytes: Vec<u8> = cast_slice(class_tensor.data()).to_vec();
        if let Ok(names) = serde_json::from_slice::<Vec<String>>(&bytes) {
            class_names = names;
        }
    }

    let color_shape_tensor = tensors.tensor("color_shape")?;
    let color_shape: [i64; 6] = {
        let mut out = [0i64; 6];
        let bytes = color_shape_tensor.data();
        let ints: &[i64] = cast_slice(bytes);
        out.copy_from_slice(ints);
        out
    };

    let b = color_shape[0] as usize;
    let steps = color_shape[1] as usize;
    let view_dim = color_shape[2] as usize;
    let height = color_shape[3] as usize;
    let width = color_shape[4] as usize;

    let mut samples = vec![
        ZeroverseSample {
            views: vec![bevy_zeroverse::sample::View::default(); view_dim * steps],
            view_dim: view_dim as u32,
            aabb: [[0.0; 3]; 2],
            object_obbs: Vec::new(),
        };
        b
    ];

    for key in tensors.names() {
        if key.starts_with("color_jpg_") {
            let idx: usize = key.trim_start_matches("color_jpg_").parse().unwrap_or(0);
            let tensor = tensors.tensor(key)?;
            let (_, _, rgba) = decode_jpeg_to_rgba_f32(tensor.data())?;
            let sample_idx = idx / (view_dim * steps);
            let local = idx % (view_dim * steps);
            samples[sample_idx].views[local].color = cast_slice(&rgba).to_vec();
            continue;
        }
    }

    let mut fill_tensor =
        |name: &str,
         channels: usize,
         setter: &mut dyn FnMut(&mut bevy_zeroverse::sample::View, &[f32])| {
            if let Ok(tensor) = tensors.tensor(name) {
                let data: &[f32] = cast_slice(tensor.data());
                for (b_idx, sample) in samples.iter_mut().enumerate().take(b) {
                    for t in 0..steps {
                        for v in 0..view_dim {
                            let idx =
                                (((b_idx * steps + t) * view_dim + v) * height * width * channels)
                                    as usize;
                            let slice = &data[idx..idx + height * width * channels];
                            let view_idx = t * view_dim + v;
                            setter(&mut sample.views[view_idx], slice);
                        }
                    }
                }
            }
        };

    fill_tensor("depth", 1, &mut |view, data| {
        // expand depth to RGBA (store in depth buffer)
        let mut rgba = Vec::with_capacity(data.len() * 4);
        for d in data {
            rgba.extend_from_slice(&[*d; 4]);
        }
        view.depth = cast_slice(&rgba).to_vec();
    });

    fill_tensor("normal", 3, &mut |view, data| {
        let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
        for chunk in data.chunks_exact(3) {
            let w = chunk[0];
            rgba.extend_from_slice(chunk);
            rgba.push(w);
        }
        view.normal = cast_slice(&rgba).to_vec();
    });

    fill_tensor("optical_flow", 3, &mut |view, data| {
        let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
        for chunk in data.chunks_exact(3) {
            let w = chunk[0];
            rgba.extend_from_slice(chunk);
            rgba.push(w);
        }
        view.optical_flow = cast_slice(&rgba).to_vec();
    });

    fill_tensor("position", 3, &mut |view, data| {
        let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
        for chunk in data.chunks_exact(3) {
            let w = chunk[0];
            rgba.extend_from_slice(chunk);
            rgba.push(w);
        }
        view.position = cast_slice(&rgba).to_vec();
    });

    if let Ok(tensor) = tensors.tensor("world_from_view") {
        let data: &[f32] = cast_slice(tensor.data());
        for (b_idx, sample) in samples.iter_mut().enumerate().take(b) {
            for t in 0..steps {
                for v in 0..view_dim {
                    let idx = ((b_idx * steps + t) * view_dim + v) * 16;
                    let slice = &data[idx..idx + 16];
                    let mut mat = [[0.0f32; 4]; 4];
                    for r in 0..4 {
                        for c in 0..4 {
                            mat[r][c] = slice[r * 4 + c];
                        }
                    }
                    let view_idx = t * view_dim + v;
                    sample.views[view_idx].world_from_view = mat;
                }
            }
        }
    }

    if let Ok(tensor) = tensors.tensor("aabb") {
        let data: &[f32] = cast_slice(tensor.data());
        for (b_idx, sample) in samples.iter_mut().enumerate().take(b) {
            let start = b_idx * 6;
            let slice = &data[start..start + 6];
            sample.aabb =
                [[slice[0], slice[1], slice[2]], [slice[3], slice[4], slice[5]]];
        }
    }

    if let (Ok(center), Ok(scale), Ok(rotation), Ok(class_idx)) = (
        tensors.tensor("object_obb_center"),
        tensors.tensor("object_obb_scale"),
        tensors.tensor("object_obb_rotation"),
        tensors.tensor("object_obb_class_idx"),
    ) {
        let centers: &[f32] = cast_slice(center.data());
        let scales: &[f32] = cast_slice(scale.data());
        let rotations: &[f32] = cast_slice(rotation.data());
        let class_ids: &[i64] = cast_slice(class_idx.data());

        let max_obbs = center.shape().get(1).copied().unwrap_or(0);

        for (b_idx, sample) in samples.iter_mut().enumerate().take(b) {
            let base_center = b_idx * max_obbs * 3;
            let base_rot = b_idx * max_obbs * 4;
            let base_class = b_idx * max_obbs;

            for i in 0..max_obbs {
                let cls = class_ids[base_class + i];
                if cls < 0 {
                    continue;
                }

                let c_idx = base_center + i * 3;
                let r_idx = base_rot + i * 4;

                let class_name = class_names
                    .get(cls as usize)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());

                sample.object_obbs.push(bevy_zeroverse::sample::ObjectObbSample {
                    center: [
                        centers[c_idx],
                        centers[c_idx + 1],
                        centers[c_idx + 2],
                    ],
                    scale: [scales[c_idx], scales[c_idx + 1], scales[c_idx + 2]],
                    rotation: [
                        rotations[r_idx],
                        rotations[r_idx + 1],
                        rotations[r_idx + 2],
                        rotations[r_idx + 3],
                    ],
                    class_name,
                });
            }
        }
    }

    fn set_fovy(v: &mut bevy_zeroverse::sample::View, x: f32) {
        v.fovy = x;
    }
    fn set_near(v: &mut bevy_zeroverse::sample::View, x: f32) {
        v.near = x;
    }
    fn set_far(v: &mut bevy_zeroverse::sample::View, x: f32) {
        v.far = x;
    }
    fn set_time(v: &mut bevy_zeroverse::sample::View, x: f32) {
        v.time = x;
    }

    type ViewSetter = fn(&mut bevy_zeroverse::sample::View, f32);
    let scalar_fields: [(&str, ViewSetter); 4] = [
        ("fovy", set_fovy),
        ("near", set_near),
        ("far", set_far),
        ("time", set_time),
    ];

    for (name, setter) in scalar_fields {
        if let Ok(tensor) = tensors.tensor(name) {
            let data: &[f32] = cast_slice(tensor.data());
            for (b_idx, sample) in samples.iter_mut().enumerate().take(b) {
                for t in 0..steps {
                    for v in 0..view_dim {
                        let idx = (b_idx * steps + t) * view_dim + v;
                        let view_idx = t * view_dim + v;
                        setter(&mut sample.views[view_idx], data[idx]);
                    }
                }
            }
        }
    }

    Ok(samples)
}

pub fn discover_chunks(root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut files: Vec<_> = fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                ext.ends_with("lz4") || ext.ends_with("zst") || ext.ends_with("safetensors")
            } else {
                false
            }
        })
        .collect();
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_zeroverse::sample::ObjectObbSample;
    use tempfile::tempdir;

    fn sample_with_obb(class_name: &str) -> ZeroverseSample {
        ZeroverseSample {
            views: vec![bevy_zeroverse::sample::View::default()],
            view_dim: 1,
            aabb: [[0.0; 3]; 2],
            object_obbs: vec![ObjectObbSample {
                center: [1.0, 2.0, 3.0],
                scale: [4.0, 5.0, 6.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                class_name: class_name.to_string(),
            }],
        }
    }

    #[test]
    fn chunk_roundtrips_object_obbs_without_metadata() {
        let tmp = tempdir().unwrap();
        let path = save_chunk(
            &[sample_with_obb("chair")],
            tmp.path(),
            0,
            Compression::None,
            1,
            1,
        )
        .unwrap();

        // Ensure class names are stored as a tensor
        let raw = std::fs::read(&path).unwrap();
        let tensors = SafeTensors::deserialize(&raw).unwrap();
        assert!(tensors.tensor("object_obb_class_names").is_ok());

        let loaded = load_chunk(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        let obb = loaded[0]
            .object_obbs
            .first()
            .expect("obb should roundtrip");
        assert_eq!(obb.class_name, "chair");
        assert_eq!(obb.center, [1.0, 2.0, 3.0]);
        assert_eq!(obb.scale, [4.0, 5.0, 6.0]);
        assert_eq!(obb.rotation, [0.0, 0.0, 0.0, 1.0]);
    }
}
