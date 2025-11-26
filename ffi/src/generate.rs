use std::{borrow::Cow, path::Path, sync::mpsc::RecvTimeoutError, time::Duration};

use bevy::prelude::*;
use bevy_args::{parse_args, Deserialize, Parser, Serialize};
use bevy_zeroverse::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
use bevy_zeroverse_ffi::{create_app, setup_globals, Sample, APP_FRAME_SENDER, SAMPLE_RECEIVER};
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
        }
    }
}

fn save_stacked_views_to_safetensors(
    stacked_views: StackedViews,
    output_path: &Path,
) -> Result<(), safetensors::SafeTensorError> {
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

    serialize_to_file(data, None, output_path)
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
    let receiver = SAMPLE_RECEIVER.get().unwrap();
    let receiver = receiver.lock().unwrap();

    let mut chunk_size = 0;
    let mut chunk_index = 0;
    let mut chunk_samples = Vec::new();

    for sample_index in 0..generator_config.num_samples {
        {
            let app_frame_sender = APP_FRAME_SENDER.get().unwrap();
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

    match save_stacked_views_to_safetensors(stacked_views, &output_path) {
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
