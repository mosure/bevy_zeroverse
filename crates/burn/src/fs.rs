use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use burn::data::dataset::Dataset;
use ndarray::{Array2, Array3, ArrayD};
use ndarray_npy::NpzReader;
use safetensors::{Dtype, SafeTensors, serialize, tensor::TensorView};

use crate::{
    chunk::{
        decode_jpeg_to_rgba_f32, decode_rgba_bytes, encode_jpeg_from_rgba_f32, leak_bytes,
        normalize_hdr_image_tonemap,
    },
    dataset::ZeroverseSample,
};

const META_FILE: &str = "meta.safetensors";

pub struct FsDataset {
    sample_dirs: Vec<PathBuf>,
}

impl FsDataset {
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let mut sample_dirs: Vec<PathBuf> = fs::read_dir(root.as_ref())?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.is_dir())
            .filter(|p| p.join(META_FILE).exists())
            .collect();

        sample_dirs.sort();
        Ok(Self { sample_dirs })
    }
}

impl Dataset<ZeroverseSample> for FsDataset {
    fn len(&self) -> usize {
        self.sample_dirs.len()
    }

    fn get(&self, index: usize) -> Option<ZeroverseSample> {
        let dir = self.sample_dirs.get(index)?.clone();
        match load_sample_dir(&dir) {
            Ok(sample) => Some(sample),
            Err(err) => {
                eprintln!("failed to load sample from {:?}: {err:?}", dir);
                None
            }
        }
    }
}

fn tone_map(data: &[f32], channels: usize) -> Vec<f32> {
    if data.is_empty() || channels == 0 {
        return Vec::new();
    }

    let channels = channels.min(4);
    let mut tonemapped = Vec::with_capacity(data.len());
    for v in data {
        tonemapped.push(v / (1.0 + v));
    }

    let mut min = vec![f32::INFINITY; channels];
    let mut max = vec![f32::NEG_INFINITY; channels];
    for chunk in tonemapped.chunks_exact(channels) {
        for c in 0..channels {
            min[c] = min[c].min(chunk[c]);
            max[c] = max[c].max(chunk[c]);
        }
    }

    let mut normalized = Vec::with_capacity(data.len());
    for chunk in tonemapped.chunks_exact(channels) {
        for c in 0..channels {
            let range = (max[c] - min[c]).max(1e-8);
            normalized.push((chunk[c] - min[c]) / range);
        }
    }
    normalized
}

fn rgba_from_plane(data: &[f32], channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / channels.max(1) * 4);
    match channels {
        1 => {
            for v in data {
                out.extend_from_slice(&[*v, *v, *v, 1.0]);
            }
        }
        3 => {
            for chunk in data.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 1.0]);
            }
        }
        4 => out.extend_from_slice(data),
        _ => {}
    }
    out
}

fn write_npz(
    path: &Path,
    key: &str,
    data: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }

    let file = fs::File::create(path)?;
    let mut npz = ndarray_npy::NpzWriter::new(file);

    match channels {
        1 => {
            let array = Array2::from_shape_vec((height, width), data.to_vec())
                .context("failed to build 2d array for npz")?;
            npz.add_array(key, &array)?;
        }
        3 => {
            let array = Array3::from_shape_vec((height, width, 3), data.to_vec())
                .context("failed to build 3d array for npz")?;
            npz.add_array(key, &array)?;
        }
        4 => {
            let array = Array3::from_shape_vec((height, width, 4), data.to_vec())
                .context("failed to build 4d array for npz")?;
            npz.add_array(key, &array)?;
        }
        _ => anyhow::bail!("unsupported channel count {channels} for npz write"),
    }

    npz.finish()?;
    Ok(())
}

fn parse_indices(stem: &str, prefix: &str) -> Option<(usize, usize)> {
    if !stem.starts_with(prefix) {
        return None;
    }
    let suffix = &stem[prefix.len()..];
    let mut parts = suffix.split('_');
    let timestep: usize = parts.next()?.parse().ok()?;
    let view: usize = parts.next()?.parse().ok()?;
    Some((timestep, view))
}

fn collect_color_files(dir: &Path) -> Result<Vec<(usize, usize, PathBuf)>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.eq_ignore_ascii_case("jpg") || s.eq_ignore_ascii_case("jpeg"))
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if let Some((t, v)) = parse_indices(stem, "color_") {
            entries.push((t, v, path));
        }
    }
    entries.sort_by_key(|(t, v, _)| (*t, *v));
    Ok(entries)
}

fn load_npz_array(path: &Path, key: &str) -> Result<Option<(Vec<f32>, Vec<usize>)>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = fs::File::open(path)?;
    let mut reader = NpzReader::new(file)?;
    let array: ArrayD<f32> = reader.by_name(key)?;
    let shape = array.shape().to_vec();
    let (data, _offset) = array.into_raw_vec_and_offset();
    Ok(Some((data, shape)))
}

fn reshape_plane(
    data: Vec<f32>,
    shape: &[usize],
    height: usize,
    width: usize,
    expected_channels: usize,
) -> Option<Vec<f32>> {
    let target_len = height * width * expected_channels.max(1);
    if expected_channels > 0 && data.len() == target_len {
        return Some(data);
    }

    match expected_channels {
        1 => {
            if data.len() == width {
                let mut out = Vec::with_capacity(target_len);
                for _ in 0..height {
                    out.extend_from_slice(&data);
                }
                return Some(out);
            }
            if data.len() == height * width {
                return Some(data);
            }
        }
        3 | 4 => {
            if data.len() == height * width {
                let mut out = Vec::with_capacity(target_len);
                for v in data {
                    for _ in 0..expected_channels {
                        out.push(v);
                    }
                }
                return Some(out);
            }
            if data.len() == width * expected_channels {
                let mut out = Vec::with_capacity(target_len);
                for _ in 0..height {
                    out.extend_from_slice(&data);
                }
                return Some(out);
            }
        }
        _ => {}
    }

    eprintln!(
        "could not coerce npz data (shape {:?}, len {}) to {}x{}x{}",
        shape,
        data.len(),
        height,
        width,
        expected_channels
    );
    None
}

fn fill_plane_from_npz(
    dir: &Path,
    key: &str,
    timestep: usize,
    view: usize,
    height: usize,
    width: usize,
    expected_channels: usize,
) -> Option<Vec<f32>> {
    let file = dir.join(format!("{key}_{timestep:03}_{view:02}.npz"));
    let (data, shape) = load_npz_array(&file, key).ok().flatten()?;
    reshape_plane(data, &shape, height, width, expected_channels)
}

fn decode_color(bytes: &[u8], expected_width: u32, expected_height: u32) -> Result<Vec<f32>> {
    let (w, h, rgba) = decode_jpeg_to_rgba_f32(bytes)?;
    if w != expected_width || h != expected_height {
        anyhow::bail!(
            "color image dimensions mismatch: expected {}x{}, got {}x{}",
            expected_width,
            expected_height,
            w,
            h
        );
    }
    Ok(rgba)
}

struct MetaFields {
    world_from_view: Vec<[[f32; 4]; 4]>,
    fovy: Vec<f32>,
    near: Vec<f32>,
    far: Vec<f32>,
    time: Vec<f32>,
    aabb: [[f32; 3]; 2],
}

fn load_meta(dir: &Path, steps: usize, view_dim: usize) -> Result<MetaFields> {
    let meta_path = dir.join(META_FILE);
    let data = fs::read(&meta_path).with_context(|| format!("failed to read {:?}", meta_path))?;
    let tensors = SafeTensors::deserialize(&data)?;

    let default_len = steps * view_dim;
    let mut world_from_view = vec![[[0.0; 4]; 4]; default_len];
    let mut fovy = vec![0.0; default_len];
    let mut near = vec![0.0; default_len];
    let mut far = vec![0.0; default_len];
    let mut time = vec![0.0; default_len];
    let mut aabb = [[0.0; 3]; 2];

    if let Ok(tensor) = tensors.tensor("world_from_view") {
        let data: &[f32] = bytemuck::cast_slice(tensor.data());
        let shape = tensor.shape();
        let shape = if !shape.is_empty() && shape[0] == 1 {
            &shape[1..]
        } else {
            shape
        };

        if shape.len() == 4 {
            let (s, v) = (shape[0], shape[1]);
            for t in 0..s.min(steps) {
                for vi in 0..v.min(view_dim) {
                    let idx = (t * view_dim + vi).min(world_from_view.len() - 1);
                    let start = (t * v + vi) * 16;
                    if start + 16 > data.len() {
                        continue;
                    }
                    let mut mat = [[0.0f32; 4]; 4];
                    for r in 0..4 {
                        for c in 0..4 {
                            mat[r][c] = data[start + r * 4 + c];
                        }
                    }
                    world_from_view[idx] = mat;
                }
            }
        }
    }

    for (name, target) in [
        ("fovy", &mut fovy),
        ("near", &mut near),
        ("far", &mut far),
        ("time", &mut time),
    ] {
        if let Ok(tensor) = tensors.tensor(name) {
            let data: &[f32] = bytemuck::cast_slice(tensor.data());
            let shape = tensor.shape();
            let shape = if !shape.is_empty() && shape[0] == 1 {
                &shape[1..]
            } else {
                shape
            };

            let (s, v) = if shape.len() >= 2 {
                (shape[0], shape[1])
            } else {
                (steps, view_dim)
            };

            for t in 0..s.min(steps) {
                for vi in 0..v.min(view_dim) {
                    let idx = (t * view_dim + vi).min(target.len() - 1);
                    let data_idx = if shape.len() == 3 {
                        (t * v + vi) * shape[2]
                    } else {
                        t * v + vi
                    };
                    target[idx] = data.get(data_idx).cloned().unwrap_or(0.0);
                }
            }
        }
    }

    if let Ok(tensor) = tensors.tensor("aabb") {
        let data: &[f32] = bytemuck::cast_slice(tensor.data());
        if data.len() >= 6 {
            aabb = [[data[0], data[1], data[2]], [data[3], data[4], data[5]]];
        }
    }

    Ok(MetaFields {
        world_from_view,
        fovy,
        near,
        far,
        time,
        aabb,
    })
}

pub fn load_sample_dir(dir: impl AsRef<Path>) -> Result<ZeroverseSample> {
    let dir = dir.as_ref();
    let color_files = collect_color_files(dir)?;
    if color_files.is_empty() {
        anyhow::bail!("no color files found in {:?}", dir);
    }

    let timesteps: BTreeSet<usize> = color_files.iter().map(|(t, _, _)| *t).collect();
    let views: BTreeSet<usize> = color_files.iter().map(|(_, v, _)| *v).collect();
    let steps = timesteps.len().max(1);
    let view_dim = views.len().max(1);

    let mut color_map: BTreeMap<(usize, usize), PathBuf> = BTreeMap::new();
    for (t, v, path) in color_files {
        color_map.insert((t, v), path);
    }

    let first_color = color_map
        .values()
        .next()
        .context("no color paths after map build")?;
    let first_bytes = fs::read(first_color)?;
    let (width, height, _) = decode_jpeg_to_rgba_f32(&first_bytes)?;

    let meta = load_meta(dir, steps, view_dim)?;

    let mut sample = ZeroverseSample {
        views: vec![Default::default(); steps * view_dim],
        view_dim: view_dim as u32,
        aabb: meta.aabb,
    };

    let timestep_list: Vec<usize> = timesteps.into_iter().collect();
    let view_list: Vec<usize> = views.into_iter().collect();

    for (t_idx, t_val) in timestep_list.iter().enumerate() {
        for (v_idx, v_val) in view_list.iter().enumerate() {
            let idx = t_idx * view_dim + v_idx;
            let view = sample
                .views
                .get_mut(idx)
                .context("sample views indexing failed")?;

            view.world_from_view = meta
                .world_from_view
                .get(idx)
                .cloned()
                .unwrap_or([[0.0; 4]; 4]);
            view.fovy = *meta.fovy.get(idx).unwrap_or(&0.0);
            view.near = *meta.near.get(idx).unwrap_or(&0.0);
            view.far = *meta.far.get(idx).unwrap_or(&0.0);
            view.time = *meta.time.get(idx).unwrap_or(&0.0);

            if let Some(path) = color_map.get(&(*t_val, *v_val)) {
                let bytes = fs::read(path)?;
                let rgba = decode_color(&bytes, width, height)?;
                view.color = bytemuck::cast_slice(&rgba).to_vec();
            }

            if let Some(depth_plane) = fill_plane_from_npz(
                dir,
                "depth",
                *t_val,
                *v_val,
                height as usize,
                width as usize,
                1,
            ) {
                let mut rgba = Vec::with_capacity(depth_plane.len() * 4);
                for d in depth_plane {
                    rgba.extend_from_slice(&[d, d, d, d]);
                }
                view.depth = bytemuck::cast_slice(&rgba).to_vec();
            }

            if let Some(normal_plane) = fill_plane_from_npz(
                dir,
                "normal",
                *t_val,
                *v_val,
                height as usize,
                width as usize,
                3,
            ) {
                let mut rgba = Vec::with_capacity(normal_plane.len() / 3 * 4);
                for chunk in normal_plane.chunks_exact(3) {
                    rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], chunk[0]]);
                }
                view.normal = bytemuck::cast_slice(&rgba).to_vec();
            }

            if let Some(position_plane) = fill_plane_from_npz(
                dir,
                "position",
                *t_val,
                *v_val,
                height as usize,
                width as usize,
                3,
            ) {
                let mut rgba = Vec::with_capacity(position_plane.len() / 3 * 4);
                for chunk in position_plane.chunks_exact(3) {
                    rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], chunk[0]]);
                }
                view.position = bytemuck::cast_slice(&rgba).to_vec();
            }

            let optical_flow_path = dir.join(format!("optical_flow_{t_val:03}_{v_val:02}.jpg"));
            if optical_flow_path.exists() {
                let bytes = fs::read(&optical_flow_path)?;
                let (w, h, rgba) = decode_jpeg_to_rgba_f32(&bytes)?;
                if w == width && h == height {
                    view.optical_flow = bytemuck::cast_slice(&rgba).to_vec();
                }
            }
        }
    }

    Ok(sample)
}

pub fn save_sample_to_fs(
    sample: &ZeroverseSample,
    output_dir: impl AsRef<Path>,
    sample_idx: usize,
    width: u32,
    height: u32,
) -> Result<PathBuf> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;
    let scene_dir = output_dir.join(format!("{sample_idx:06}"));
    fs::create_dir_all(&scene_dir)?;

    let view_dim = sample.view_dim as usize;
    anyhow::ensure!(view_dim > 0, "view_dim must be > 0");
    anyhow::ensure!(
        sample.views.len().checked_rem(view_dim) == Some(0),
        "view count must be a multiple of view_dim"
    );
    let steps = (sample.views.len() / view_dim).max(1);

    let pixel_count = (height * width) as usize;
    let mut color_tensor: Option<Vec<f32>> = sample
        .views
        .iter()
        .any(|v| !v.color.is_empty())
        .then(|| vec![0.0; steps * view_dim * pixel_count * 3]);
    let mut flow_tensor: Option<Vec<f32>> = sample
        .views
        .iter()
        .any(|v| !v.optical_flow.is_empty())
        .then(|| vec![0.0; steps * view_dim * pixel_count * 3]);

    for t in 0..steps {
        for v in 0..view_dim {
            let idx = t * view_dim + v;
            let view = &sample.views[idx];

            if let Some(color_buf) = color_tensor.as_mut()
                && !view.color.is_empty()
            {
                let rgba = decode_rgba_bytes(&view.color, width, height)?;
                let base = (t * view_dim + v) * pixel_count * 3;
                for i in 0..pixel_count {
                    let src = i * 4;
                    let dst = base + i * 3;
                    color_buf[dst] = rgba[src];
                    color_buf[dst + 1] = rgba[src + 1];
                    color_buf[dst + 2] = rgba[src + 2];
                }
            }

            if let Some(flow_buf) = flow_tensor.as_mut()
                && !view.optical_flow.is_empty()
            {
                let rgba = decode_rgba_bytes(&view.optical_flow, width, height)?;
                let base = (t * view_dim + v) * pixel_count * 3;
                for i in 0..pixel_count {
                    let src = i * 4;
                    let dst = base + i * 3;
                    flow_buf[dst] = rgba[src];
                    flow_buf[dst + 1] = rgba[src + 1];
                    flow_buf[dst + 2] = rgba[src + 2];
                }
            }

            if !view.depth.is_empty() {
                let depth_rgba = decode_rgba_bytes(&view.depth, width, height)?;
                let depth_plane: Vec<f32> = depth_rgba.chunks_exact(4).map(|c| c[0]).collect();
                write_npz(
                    &scene_dir.join(format!("depth_{t:03}_{v:02}.npz")),
                    "depth",
                    &depth_plane,
                    height as usize,
                    width as usize,
                    1,
                )?;

                let mapped = tone_map(&depth_plane, 1);
                let rgba = rgba_from_plane(&mapped, 1);
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                fs::write(scene_dir.join(format!("depth_{t:03}_{v:02}.jpg")), jpg)?;
            }

            if !view.normal.is_empty() {
                let normal_rgba = decode_rgba_bytes(&view.normal, width, height)?;
                let normal_plane: Vec<f32> = normal_rgba
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect();
                write_npz(
                    &scene_dir.join(format!("normal_{t:03}_{v:02}.npz")),
                    "normal",
                    &normal_plane,
                    height as usize,
                    width as usize,
                    3,
                )?;

                let mapped = tone_map(&normal_plane, 3);
                let rgba = rgba_from_plane(&mapped, 3);
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                fs::write(scene_dir.join(format!("normal_{t:03}_{v:02}.jpg")), jpg)?;
            }

            if !view.position.is_empty() {
                let pos_rgba = decode_rgba_bytes(&view.position, width, height)?;
                let position_plane: Vec<f32> = pos_rgba
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect();
                write_npz(
                    &scene_dir.join(format!("position_{t:03}_{v:02}.npz")),
                    "position",
                    &position_plane,
                    height as usize,
                    width as usize,
                    3,
                )?;

                let mapped = tone_map(&position_plane, 3);
                let rgba = rgba_from_plane(&mapped, 3);
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                fs::write(scene_dir.join(format!("position_{t:03}_{v:02}.jpg")), jpg)?;
            }
        }
    }

    if let Some(mut color_buf) = color_tensor {
        normalize_hdr_image_tonemap(
            &mut color_buf,
            steps,
            view_dim,
            height as usize,
            width as usize,
            3,
        );
        for t in 0..steps {
            for v in 0..view_dim {
                let base = (t * view_dim + v) * pixel_count * 3;
                let mut rgba = Vec::with_capacity(pixel_count * 4);
                for i in 0..pixel_count {
                    let src = base + i * 3;
                    rgba.extend_from_slice(&[
                        color_buf[src],
                        color_buf[src + 1],
                        color_buf[src + 2],
                        1.0,
                    ]);
                }
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                fs::write(scene_dir.join(format!("color_{t:03}_{v:02}.jpg")), jpg)?;
            }
        }
    }

    if let Some(mut flow_buf) = flow_tensor {
        normalize_hdr_image_tonemap(
            &mut flow_buf,
            steps,
            view_dim,
            height as usize,
            width as usize,
            3,
        );
        for t in 0..steps {
            for v in 0..view_dim {
                let base = (t * view_dim + v) * pixel_count * 3;
                let mut rgba = Vec::with_capacity(pixel_count * 4);
                for i in 0..pixel_count {
                    let src = base + i * 3;
                    rgba.extend_from_slice(&[
                        flow_buf[src],
                        flow_buf[src + 1],
                        flow_buf[src + 2],
                        1.0,
                    ]);
                }
                let jpg = encode_jpeg_from_rgba_f32(&rgba, width, height)?;
                fs::write(
                    scene_dir.join(format!("optical_flow_{t:03}_{v:02}.jpg")),
                    jpg,
                )?;
            }
        }
    }

    let mut tensors: Vec<(&str, TensorView<'static>)> = Vec::new();

    let mut world_from_view = Vec::with_capacity(steps * view_dim * 16);
    let mut fovy = Vec::with_capacity(steps * view_dim);
    let mut near = Vec::with_capacity(steps * view_dim);
    let mut far = Vec::with_capacity(steps * view_dim);
    let mut time = Vec::with_capacity(steps * view_dim);

    for t in 0..steps {
        for v in 0..view_dim {
            let idx = t * view_dim + v;
            let view = &sample.views[idx];
            for row in view.world_from_view {
                world_from_view.extend_from_slice(&row);
            }
            fovy.push(view.fovy);
            near.push(view.near);
            far.push(view.far);
            time.push(view.time);
        }
    }

    let leak_tensor = |name: &'static str,
                       shape: Vec<usize>,
                       data: Vec<f32>,
                       tensors: &mut Vec<(&'static str, TensorView<'static>)>|
     -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let buf = leak_bytes(bytemuck::cast_slice(&data).to_vec());
        tensors.push((name, TensorView::new(Dtype::F32, shape, buf)?));
        Ok(())
    };

    leak_tensor(
        "world_from_view",
        vec![steps, view_dim, 4, 4],
        world_from_view,
        &mut tensors,
    )?;
    leak_tensor("fovy", vec![steps, view_dim, 1], fovy, &mut tensors)?;
    leak_tensor("near", vec![steps, view_dim, 1], near, &mut tensors)?;
    leak_tensor("far", vec![steps, view_dim, 1], far, &mut tensors)?;
    leak_tensor("time", vec![steps, view_dim, 1], time, &mut tensors)?;

    let aabb_buf = leak_bytes(bytemuck::cast_slice(&sample.aabb).to_vec());
    tensors.push(("aabb", TensorView::new(Dtype::F32, vec![2, 3], aabb_buf)?));

    let meta = serialize(tensors, None)?;
    fs::write(scene_dir.join(META_FILE), meta)?;

    Ok(scene_dir)
}
