use std::{
    borrow::Cow,
    path::Path,
    sync::mpsc::RecvTimeoutError,
    time::Duration,
};

use bevy::prelude::*;
use bevy_args::{
    parse_args,
    Deserialize,
    Parser,
    Serialize,
};
use bevy_zeroverse::app::BevyZeroverseConfig;
use bevy_zeroverse_ffi::{
    create_app,
    setup_globals,
    SAMPLE_RECEIVER,
    APP_FRAME_SENDER,
    Sample,
};
use ndarray::{Array5, Array4, Array3, Array2, Array1, ArrayBase, OwnedRepr, Axis, Dimension, s};
use safetensors::{
    serialize_to_file,
    View,
    Dtype,
};



pub struct StackedViews {
    pub color: Array5<u8>,            // Shape: (batch_size, num_views, height, width, channels)
    pub depth: Array5<u8>,            // Shape: (batch_size, num_views, height, width, 1)
    pub normal: Array5<u8>,           // Shape: (batch_size, num_views, height, width, channels)
    pub world_from_view: Array4<f32>, // Shape: (batch_size, num_views, 4, 4)
    pub fovy: Array3<f32>,            // Shape: (batch_size, num_views)
}

struct Wrapper<A, D>(ArrayBase<OwnedRepr<A>, D>);

impl<D: Dimension> Wrapper<f32, D> {
    fn buffer(&self) -> &[u8] {
        let slice = self.0.as_slice().expect("Non-contiguous tensors");
        let num_bytes = std::mem::size_of::<f32>();
        let new_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * num_bytes)
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
    fn data(&self) -> Cow<[u8]> {
        self.buffer().into()
    }
    fn data_len(&self) -> usize {
        self.buffer().len()
    }
}

impl<D: Dimension> Wrapper<u8, D> {
    fn buffer(&self) -> &[u8] {
        let slice = self.0.as_slice().expect("Non-contiguous tensors");
        slice
    }
}

impl<D: Dimension> View for Wrapper<u8, D> {
    fn dtype(&self) -> Dtype {
        Dtype::U8
    }
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }
    fn data(&self) -> Cow<[u8]> {
        self.buffer().into()
    }
    fn data_len(&self) -> usize {
        self.buffer().len()
    }
}

fn stack_samples(samples: Vec<Sample>, height: usize, width: usize) -> StackedViews {
    let batch_size = samples.len();
    let num_views = samples.first().map_or(0, |sample| sample.views.len());

    let mut color_stacks = Vec::new();
    let mut depth_stacks = Vec::new();
    let mut normal_stacks = Vec::new();
    let mut world_from_view_stacks = Vec::new();
    let mut fovy_stacks = Vec::new();

    for sample in samples {
        let mut color_views = Vec::new();
        let mut depth_views = Vec::new();
        let mut normal_views = Vec::new();
        let mut world_from_view_views = Vec::new();
        let mut fovy_views = Vec::new();

        for view in sample.views {
            // Reshape raw vectors into the appropriate ndarray dimensions
            let color = Array3::from_shape_vec((height, width, 4), view.color).unwrap(); // Assuming color has 4 channels (RGBA)
            let depth = Array3::from_shape_vec((height, width, 1), view.depth).unwrap(); // Assuming depth has a single channel
            let normal = Array3::from_shape_vec((height, width, 4), view.normal).unwrap(); // Assuming normal has 4 channels (could be 3 for XYZ + 1 for padding)

            let world_from_view = Array2::from_shape_vec((4, 4), view.world_from_view.concat()).unwrap();
            let fovy = Array1::from_elem(1, view.fovy);

            color_views.push(color);
            depth_views.push(depth);
            normal_views.push(normal);
            world_from_view_views.push(world_from_view);
            fovy_views.push(fovy);
        }

        let color_views_stacked = ndarray::stack(Axis(0), &color_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let depth_views_stacked = ndarray::stack(Axis(0), &depth_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let normal_views_stacked = ndarray::stack(Axis(0), &normal_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let world_from_view_views_stacked = ndarray::stack(Axis(0), &world_from_view_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let fovy_views_stacked = ndarray::stack(Axis(0), &fovy_views.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();

        color_stacks.push(color_views_stacked);
        depth_stacks.push(depth_views_stacked);
        normal_stacks.push(normal_views_stacked);
        world_from_view_stacks.push(world_from_view_views_stacked);
        fovy_stacks.push(fovy_views_stacked);
    }

    let color_stacked = ndarray::stack(Axis(0), &color_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
    let depth_stacked = ndarray::stack(Axis(0), &depth_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
    let normal_stacked = ndarray::stack(Axis(0), &normal_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
    let world_from_view_stacked = ndarray::stack(Axis(0), &world_from_view_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
    let fovy_stacked = ndarray::stack(Axis(0), &fovy_stacks.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();

    StackedViews {
        color: color_stacked,            // Shape: (batch_size, num_views, height, width, channels)
        depth: depth_stacked,            // Shape: (batch_size, num_views, height, width, 1)
        normal: normal_stacked,          // Shape: (batch_size, num_views, height, width, channels)
        world_from_view: world_from_view_stacked,  // Shape: (batch_size, num_views, 4, 4)
        fovy: fovy_stacked,              // Shape: (batch_size, num_views)
    }
}


enum TensorView {
    Color(Wrapper<u8, ndarray::Ix5>),
    Depth(Wrapper<u8, ndarray::Ix5>),
    Normal(Wrapper<u8, ndarray::Ix5>),
    WorldFromView(Wrapper<f32, ndarray::Ix4>),
    Fovy(Wrapper<f32, ndarray::Ix3>),
}

impl View for TensorView {
    fn dtype(&self) -> Dtype {
        match self {
            TensorView::Color(t) => t.dtype(),
            TensorView::Depth(t) => t.dtype(),
            TensorView::Normal(t) => t.dtype(),
            TensorView::WorldFromView(t) => t.dtype(),
            TensorView::Fovy(t) => t.dtype(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            TensorView::Color(t) => t.shape(),
            TensorView::Depth(t) => t.shape(),
            TensorView::Normal(t) => t.shape(),
            TensorView::WorldFromView(t) => t.shape(),
            TensorView::Fovy(t) => t.shape(),
        }
    }

    fn data(&self) -> Cow<[u8]> {
        match self {
            TensorView::Color(t) => t.data(),
            TensorView::Depth(t) => t.data(),
            TensorView::Normal(t) => t.data(),
            TensorView::WorldFromView(t) => t.data(),
            TensorView::Fovy(t) => t.data(),
        }
    }

    fn data_len(&self) -> usize {
        match self {
            TensorView::Color(t) => t.data_len(),
            TensorView::Depth(t) => t.data_len(),
            TensorView::Normal(t) => t.data_len(),
            TensorView::WorldFromView(t) => t.data_len(),
            TensorView::Fovy(t) => t.data_len(),
        }
    }
}


fn save_stacked_views_to_safetensors(stacked_views: StackedViews, output_path: &Path) -> Result<(), safetensors::SafeTensorError> {
    let data: Vec<(&str, TensorView)> = vec![
        ("color", TensorView::Color(Wrapper(stacked_views.color))),
        ("depth", TensorView::Depth(Wrapper(stacked_views.depth))),
        ("normal", TensorView::Normal(Wrapper(stacked_views.normal))),
        ("world_from_view", TensorView::WorldFromView(Wrapper(stacked_views.world_from_view))),
        ("fovy", TensorView::Fovy(Wrapper(stacked_views.fovy))),
    ];

    serialize_to_file(data, &None, output_path)
}



#[derive(
    Clone,
    Debug,
    Resource,
    Serialize,
    Deserialize,
    Parser,
    Reflect,
)]
#[command(about = "bevy_zeroverse viewer", version, long_about = None)]
#[reflect(Resource)]
pub struct GeneratorConfig {
    #[arg(long, default_value = "100")]
    pub num_samples: usize,

    #[arg(long, default_value = "268435456")]  // 256 MB
    pub bytes_per_chunk: usize,
}

impl Default for GeneratorConfig {
    fn default() -> GeneratorConfig {
        GeneratorConfig {
            num_samples: 100,
            bytes_per_chunk: 268435456,
        }
    }
}


fn receive_samples(generator_config: GeneratorConfig) {
    let receiver = SAMPLE_RECEIVER.get().unwrap();
    let receiver = receiver.lock().unwrap();

    for _ in 0..generator_config.num_samples {
        {
            let app_frame_sender = APP_FRAME_SENDER.get().unwrap();
            app_frame_sender.send(()).unwrap();
        }

        let timeout = Duration::from_secs(30);
        match receiver.recv_timeout(timeout) {
            Ok(sample) => {
                println!("Received sample: {:?}", sample);
                // accumulate a chunk based on bytes_per_chunk, flush to safetensor when full
            },
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
}


fn main() {
    let generator_args = parse_args::<GeneratorConfig>();
    let zeroverse_args = parse_args::<BevyZeroverseConfig>();

    setup_globals(None);

    std::thread::spawn(|| {
        receive_samples(generator_args);
    });

    // TODO: set to headless mode
    let mut app = create_app(zeroverse_args.into());
    app.run();
}


// TODO: add test comparing safetensors from python vs rust
