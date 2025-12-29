use std::{borrow::Cow, path::Path, sync::mpsc::RecvTimeoutError, time::Duration};

use bevy::prelude::*;
use bevy_args::{parse_args, Deserialize, Parser, Serialize};
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    headless::{create_app, setup_globals},
    io::channels,
    sample::Sample,
    scene::ZeroverseSceneType,
};
use bytemuck::cast_slice;
use ndarray::{s, Array1, Array2, Array3, Array4, Array5, ArrayBase, Axis, Dimension, OwnedRepr};
use safetensors::{serialize_to_file, Dtype, View};

pub struct StackedViews {
    pub color: Array5<f32>, // Shape: (batch_size, num_views, height, width, channels)
    pub depth: Array5<f32>, // Shape: (batch_size, num_views, height, width, channels)
    // pub normal: Array5<f32>,            // Shape: (batch_size, num_views, height, width, channels)
    pub world_from_view: Array4<f32>, // Shape: (batch_size, num_views, 4, 4)
    pub fovy: Array3<f32>,            // Shape: (batch_size, num_views, 1)
    pub near: Array3<f32>,            // Shape: (batch_size, num_views, 1)
    pub far: Array3<f32>,             // Shape: (batch_size, num_views, 1)
    pub aabb: Array3<f32>,            // Shape: (batch_size, 2, 3) - min and max
}

struct Wrapper<A, D>(ArrayBase<OwnedRepr<A>, D>);

impl<D: Dimension> Wrapper<f32, D> {
    fn buffer(&self) -> &[u8] {
        let slice = self.0.as_slice().expect("Non-contiguous tensors");
        let new_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        };
        new_slice
    }
}

impl<D: Dimension> View for Wrapper<f32, D> {
    fn dtype(&self) -> Dtype {
        Dtype::F32
    }
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
    fn data(&self) -> Cow<'_, [u8]> {
        self.buffer().into()
    }
    fn data_len(&self) -> usize {
        self.buffer().len()
    }
}

#[derive(Clone)]
struct RawTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for RawTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Owned(self.data.clone())
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn stack_samples(samples: &[Sample], zeroverse_config: &BevyZeroverseConfig) -> StackedViews {
    let _batch_size = samples.len();
    let _num_views = samples.first().map_or(0, |sample| sample.views.len());
    let height = zeroverse_config.height as usize;
    let width = zeroverse_config.width as usize;

    let mut color_stacks = Vec::new();
    let mut depth_stacks = Vec::new();
    // let mut normal_stacks = Vec::new();
    let mut world_from_view_stacks = Vec::new();
    let mut fovy_stacks = Vec::new();
    let mut near_stacks = Vec::new();
    let mut far_stacks = Vec::new();
    let mut aabb_stacks = Vec::new();

    for sample in samples {
        let mut color_views = Vec::new();
        let mut depth_views = Vec::new();
        // let mut normal_views = Vec::new();
        let mut world_from_view_views = Vec::new();
        let mut fovy_views = Vec::new();
        let mut near_views = Vec::new();
        let mut far_views = Vec::new();

        for view in &sample.views {
            let color_f32: &[f32] = cast_slice(view.color.as_slice());
            let depth_f32: &[f32] = cast_slice(view.depth.as_slice());
            // let normal_f32: &[f32] = cast_slice(view.normal.as_slice());

            let color = Array3::from_shape_vec((height, width, 4), color_f32.to_vec()).unwrap();
            let depth = Array3::from_shape_vec((height, width, 4), depth_f32.to_vec()).unwrap();
            // let normal = Array3::from_shape_vec((height, width, 4), normal_f32.to_vec()).unwrap();

            let world_from_view = Array2::from_shape_vec(
                (4, 4),
                view.world_from_view
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect(),
            )
            .unwrap();
            let fovy = Array1::from_elem(1, view.fovy);
            let near = Array1::from_elem(1, view.near);
            let far = Array1::from_elem(1, view.far);

            let color = color.slice(s![.., .., 0..3]).to_owned();
            let depth = depth.slice(s![.., .., 0..3]).to_owned();
            // let normal = normal.slice(s![.., .., 0..3]).to_owned();

            color_views.push(color);
            depth_views.push(depth);
            // normal_views.push(normal);
            world_from_view_views.push(world_from_view);
            fovy_views.push(fovy);
            near_views.push(near);
            far_views.push(far);
        }

        let color_views_stacked = ndarray::stack(
            Axis(0),
            &color_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let depth_views_stacked = ndarray::stack(
            Axis(0),
            &depth_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        // let normal_views_stacked = ndarray::stack(Axis(0), &normal_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let world_from_view_views_stacked = ndarray::stack(
            Axis(0),
            &world_from_view_views
                .iter()
                .map(|a| a.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let fovy_views_stacked = ndarray::stack(
            Axis(0),
            &fovy_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let near_views_stacked = ndarray::stack(
            Axis(0),
            &near_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();
        let far_views_stacked = ndarray::stack(
            Axis(0),
            &far_views.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        color_stacks.push(color_views_stacked);
        depth_stacks.push(depth_views_stacked);
        // normal_stacks.push(normal_views_stacked);
        world_from_view_stacks.push(world_from_view_views_stacked);
        fovy_stacks.push(fovy_views_stacked);
        near_stacks.push(near_views_stacked);
        far_stacks.push(far_views_stacked);

        let sample_aabb = Array2::from_shape_vec(
            (2, 3),
            sample
                .aabb
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect(),
        )
        .unwrap();
        aabb_stacks.push(sample_aabb);
    }

    let color_stacked = ndarray::stack(
        Axis(0),
        &color_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let depth_stacked = ndarray::stack(
        Axis(0),
        &depth_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    // let normal_stacked = ndarray::stack(Axis(0), &normal_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
    let world_from_view_stacked = ndarray::stack(
        Axis(0),
        &world_from_view_stacks
            .iter()
            .map(|a| a.view())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let fovy_stacked = ndarray::stack(
        Axis(0),
        &fovy_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let near_stacked = ndarray::stack(
        Axis(0),
        &near_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let far_stacked = ndarray::stack(
        Axis(0),
        &far_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();
    let aabb_stacked = ndarray::stack(
        Axis(0),
        &aabb_stacks.iter().map(|a| a.view()).collect::<Vec<_>>(),
    )
    .unwrap();

    StackedViews {
        color: color_stacked,
        depth: depth_stacked,
        // normal: normal_stacked,
        world_from_view: world_from_view_stacked,
        fovy: fovy_stacked,
        near: near_stacked,
        far: far_stacked,
        aabb: aabb_stacked,
    }
}

enum TensorView {
    Color(Wrapper<f32, ndarray::Ix5>),
    Depth(Wrapper<f32, ndarray::Ix5>),
    // Normal(Wrapper<f32, ndarray::Ix5>),
    WorldFromView(Wrapper<f32, ndarray::Ix4>),
    Aabb(Wrapper<f32, ndarray::Ix3>),
    Singular(Wrapper<f32, ndarray::Ix3>),
    Raw(RawTensor),
}

impl View for TensorView {
    fn dtype(&self) -> Dtype {
        match self {
            TensorView::Color(t) => t.dtype(),
            TensorView::Depth(t) => t.dtype(),
            // TensorView::Normal(t) => t.dtype(),
            TensorView::WorldFromView(t) => t.dtype(),
            TensorView::Aabb(t) => t.dtype(),
            TensorView::Singular(t) => t.dtype(),
            TensorView::Raw(t) => t.dtype(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            TensorView::Color(t) => t.shape(),
            TensorView::Depth(t) => t.shape(),
            // TensorView::Normal(t) => t.shape(),
            TensorView::WorldFromView(t) => t.shape(),
            TensorView::Aabb(t) => t.shape(),
            TensorView::Singular(t) => t.shape(),
            TensorView::Raw(t) => t.shape(),
        }
    }

    fn data(&self) -> Cow<'_, [u8]> {
        match self {
            TensorView::Color(t) => t.data(),
            TensorView::Depth(t) => t.data(),
            // TensorView::Normal(t) => t.data(),
            TensorView::WorldFromView(t) => t.data(),
            TensorView::Aabb(t) => t.data(),
            TensorView::Singular(t) => t.data(),
            TensorView::Raw(t) => t.data(),
        }
    }

    fn data_len(&self) -> usize {
        match self {
            TensorView::Color(t) => t.data_len(),
            TensorView::Depth(t) => t.data_len(),
            // TensorView::Normal(t) => t.data_len(),
            TensorView::WorldFromView(t) => t.data_len(),
            TensorView::Aabb(t) => t.data_len(),
            TensorView::Singular(t) => t.data_len(),
            TensorView::Raw(t) => t.data_len(),
        }
    }
}

fn save_stacked_views_to_safetensors(
    stacked_views: StackedViews,
    chunk_samples: &[Sample],
    output_path: &Path,
) -> Result<(), safetensors::SafeTensorError> {
    #[allow(clippy::type_complexity)]
    type OvoxelTuple = ([u32; 3], [u8; 3], u8, [u8; 4], u16);

    let data: Vec<(&str, TensorView)> = vec![
        ("color", TensorView::Color(Wrapper(stacked_views.color))),
        ("depth", TensorView::Depth(Wrapper(stacked_views.depth))),
        // ("normal", TensorView::Normal(Wrapper(stacked_views.normal))),
        (
            "world_from_view",
            TensorView::WorldFromView(Wrapper(stacked_views.world_from_view)),
        ),
        ("fovy", TensorView::Singular(Wrapper(stacked_views.fovy))),
        ("near", TensorView::Singular(Wrapper(stacked_views.near))),
        ("far", TensorView::Singular(Wrapper(stacked_views.far))),
        ("aabb", TensorView::Aabb(Wrapper(stacked_views.aabb))),
    ];

    let mut tensors = data;

    // Flatten O-Voxel fields across the batch, tracking per-sample offsets.
    let mut coords: Vec<u32> = Vec::new();
    let mut dual: Vec<u8> = Vec::new();
    let mut intersected: Vec<u8> = Vec::new();
    let mut base_color: Vec<u8> = Vec::new();
    let mut semantic: Vec<u16> = Vec::new();
    let mut semantic_label_offsets: Vec<i64> = Vec::with_capacity(chunk_samples.len() * 2);
    let mut semantic_labels_bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(chunk_samples.len() * 2);
    let mut resolution: Vec<u32> = Vec::with_capacity(chunk_samples.len());
    let mut aabb: Vec<f32> = Vec::with_capacity(chunk_samples.len() * 6);

    for sample in chunk_samples {
        if let Some(ref ov) = sample.ovoxel {
            // Ensure coords sorted for determinism.
            let mut zipped: Vec<OvoxelTuple> = ov
                .coords
                .iter()
                .zip(ov.dual_vertices.iter())
                .zip(ov.intersected.iter())
                .zip(ov.base_color.iter())
                .zip(ov.semantics.iter())
                .map(|((((c, d), i), bc), s)| (*c, *d, *i, *bc, *s))
                .collect();
            zipped.sort_by(|a, b| a.0.cmp(&b.0));

            let start = coords.len() as i64;
            let len = zipped.len() as i64;
            offsets.push(start);
            offsets.push(len);

            for (c, d, i, bc, s) in zipped {
                coords.extend_from_slice(&c);
                dual.extend_from_slice(&d);
                intersected.push(i);
                base_color.extend_from_slice(&bc);
                semantic.push(s);
            }

            resolution.push(ov.resolution);
            for row in ov.aabb {
                aabb.extend_from_slice(&row);
            }

            let label_bytes = serde_json::to_vec(&ov.semantic_labels).unwrap_or_default();
            let lbl_start = semantic_labels_bytes.len() as i64;
            let lbl_len = label_bytes.len() as i64;
            semantic_label_offsets.push(lbl_start);
            semantic_label_offsets.push(lbl_len);
            semantic_labels_bytes.extend(label_bytes);
        } else {
            // Still append placeholders to keep alignment.
            offsets.push(coords.len() as i64);
            offsets.push(0);
            resolution.push(0);
            aabb.extend_from_slice(&[0.0; 6]);
            semantic_label_offsets.push(semantic_labels_bytes.len() as i64);
            semantic_label_offsets.push(0);
        }
    }

    if !coords.is_empty() {
        tensors.push((
            "ovoxel_coords",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U32,
                shape: vec![coords.len() / 3, 3],
                data: bytemuck::cast_slice(&coords).to_vec(),
            }),
        ));
        tensors.push((
            "ovoxel_dual_vertices",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U8,
                shape: vec![dual.len() / 3, 3],
                data: dual,
            }),
        ));
        tensors.push((
            "ovoxel_intersected",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U8,
                shape: vec![intersected.len()],
                data: intersected,
            }),
        ));
        tensors.push((
            "ovoxel_base_color",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U8,
                shape: vec![base_color.len() / 4, 4],
                data: base_color,
            }),
        ));
        tensors.push((
            "ovoxel_semantic",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U16,
                shape: vec![semantic.len()],
                data: bytemuck::cast_slice(&semantic).to_vec(),
            }),
        ));
        tensors.push((
            "ovoxel_semantic_label_offsets",
            TensorView::Raw(RawTensor {
                dtype: Dtype::I64,
                shape: vec![chunk_samples.len(), 2],
                data: bytemuck::cast_slice(&semantic_label_offsets).to_vec(),
            }),
        ));
        tensors.push((
            "ovoxel_semantic_labels",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U8,
                shape: vec![semantic_labels_bytes.len()],
                data: semantic_labels_bytes,
            }),
        ));
        tensors.push((
            "ovoxel_offsets",
            TensorView::Raw(RawTensor {
                dtype: Dtype::I64,
                shape: vec![chunk_samples.len(), 2],
                data: bytemuck::cast_slice(&offsets).to_vec(),
            }),
        ));
        tensors.push((
            "ovoxel_resolution",
            TensorView::Raw(RawTensor {
                dtype: Dtype::U32,
                shape: vec![chunk_samples.len()],
                data: bytemuck::cast_slice(&resolution).to_vec(),
            }),
        ));
        tensors.push((
            "ovoxel_aabb",
            TensorView::Raw(RawTensor {
                dtype: Dtype::F32,
                shape: vec![chunk_samples.len(), 2, 3],
                data: bytemuck::cast_slice(&aabb).to_vec(),
            }),
        ));
    }

    serialize_to_file(tensors, None, output_path)
}

#[derive(Clone, Debug, Resource, Serialize, Deserialize, Parser, Reflect)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct GeneratorConfig {
    #[arg(long, default_value = "10")]
    pub num_samples: usize,

    #[arg(long, default_value = "268435456")] // 256 MB
    pub bytes_per_chunk: usize,

    #[arg(long)] // overrides bytes_per_chunk
    pub samples_per_chunk: Option<usize>,

    #[arg(long, default_value = "data/zeroverse/rust")]
    pub output_dir: String,
}

impl Default for GeneratorConfig {
    fn default() -> GeneratorConfig {
        GeneratorConfig {
            num_samples: 10,
            bytes_per_chunk: 268435456,
            samples_per_chunk: None,
            output_dir: "data/zeroverse/rust".to_string(),
        }
    }
}

fn receive_samples(generator_config: &GeneratorConfig, zeroverse_config: &BevyZeroverseConfig) {
    let receiver = channels::sample_receiver().expect("sample receiver not initialized");
    let receiver = receiver.lock().unwrap();

    let mut chunk_size = 0;
    let mut chunk_index = 0;
    let mut chunk_samples = Vec::new();

    for sample_index in 0..generator_config.num_samples {
        {
            let app_frame_sender = channels::app_frame_sender();
            app_frame_sender.send(()).unwrap();
        }

        let timeout = Duration::from_secs(30);
        match receiver.recv_timeout(timeout) {
            Ok(sample) => {
                chunk_samples.push(sample);

                let sample_size = estimate_sample_size(chunk_samples.last().unwrap());
                chunk_size += sample_size;

                info!(
                    "    added sample {} to chunk ({:.2} MB).",
                    sample_index,
                    sample_size as f64 / 1e6
                );

                if let Some(samples_per_chunk) = generator_config.samples_per_chunk {
                    if chunk_samples.len() >= samples_per_chunk {
                        save_chunk(
                            &chunk_samples,
                            chunk_index,
                            generator_config,
                            zeroverse_config,
                        );

                        chunk_samples.clear();
                        chunk_size = 0;
                        chunk_index += 1;
                    }
                    continue;
                }

                if chunk_size >= generator_config.bytes_per_chunk {
                    save_chunk(
                        &chunk_samples,
                        chunk_index,
                        generator_config,
                        zeroverse_config,
                    );

                    chunk_samples.clear();
                    chunk_size = 0;
                    chunk_index += 1;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                error!("receive operation timed out");
                std::process::exit(1);
            }
            Err(RecvTimeoutError::Disconnected) => {
                error!("channel disconnected");
                std::process::exit(1);
            }
        }
    }

    if !chunk_samples.is_empty() {
        save_chunk(
            &chunk_samples,
            chunk_index,
            generator_config,
            zeroverse_config,
        );
    }

    info!("finished generating samples");
    std::process::exit(0);
}

fn estimate_sample_size(sample: &Sample) -> usize {
    let mut size = 0;

    for view in &sample.views {
        size += view.color.len() * 3 / 4;
        size += view.depth.len() * 3 / 4;
        // size += view.normal.len() * 3 / 4;
        size += view.world_from_view.len();
        size += 1; // fovy
        size += 1; // near
        size += 1; // far
    }

    size += sample.aabb.len();

    if let Some(ref ov) = sample.ovoxel {
        size += ov.coords.len() * (3 * 4 + 3 + 1 + 4 * 4 + 2);
        size += 6; // aabb
        size += 1; // res
        size += ov.semantic_labels.iter().map(|s| s.len()).sum::<usize>(); // palette payload
        size += 16; // semantic label offsets
    }

    size
}

fn save_chunk(
    chunk_samples: &[Sample],
    chunk_index: usize,
    generator_config: &GeneratorConfig,
    zeroverse_config: &BevyZeroverseConfig,
) {
    let stacked_views = stack_samples(chunk_samples, zeroverse_config);

    let file_name = format!("{chunk_index:06}.safetensors");
    let output_dir = Path::new(generator_config.output_dir.as_str());

    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).unwrap();
    }

    let output_path = output_dir.join(file_name);

    let chunk_size = chunk_samples
        .iter()
        .map(estimate_sample_size)
        .sum::<usize>();
    info!(
        "saving chunk {} of {} ({:.2} MB).",
        chunk_index,
        generator_config.num_samples,
        chunk_size as f64 / 1e6
    );

    match save_stacked_views_to_safetensors(stacked_views, chunk_samples, &output_path) {
        Ok(_) => info!("successfully saved chunk {}", chunk_index),
        Err(e) => warn!("failed to save chunk {}: {:?}", chunk_index, e),
    }
}

fn main() {
    let generator_args = parse_args::<GeneratorConfig>();
    let mut zeroverse_args = parse_args::<BevyZeroverseConfig>();

    zeroverse_args.editor = false;
    zeroverse_args.headless = true;
    zeroverse_args.num_cameras = 4;
    zeroverse_args.width = 640.0;
    zeroverse_args.height = 480.0;
    zeroverse_args.regenerate_scene_material_shuffle_period = 256;
    zeroverse_args.scene_type = ZeroverseSceneType::Room;

    setup_globals(None);

    let mut app = create_app(None, zeroverse_args.clone().into(), true);

    std::thread::spawn(move || {
        receive_samples(&generator_args, &zeroverse_args);
    });

    app.run();
}

// TODO: add test comparing safetensors from python vs rust
