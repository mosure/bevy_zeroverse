use std::{
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            Ordering,
        },
        mpsc::{
            self,
            Sender,
            Receiver,
            RecvTimeoutError,
        },
    },
    thread,
    time::Duration,
};

use bevy::prelude::*;
use image::{
    DynamicImage,
    ImageBuffer,
    Luma,
};
use ndarray::Array3;
use once_cell::sync::OnceCell;
use pyo3::{
    prelude::*,
    exceptions::PyTimeoutError,
    types::PyList,
};

use ::bevy_zeroverse::{
    app::{
        viewer_app,
        BevyZeroverseConfig,
    },
    io::image_copy::ImageCopier,
    render::RenderMode,
};


// TODO: move to src/sample.rs to support torch (python) and burn dataloaders
type ColorImage = Array3<f32>;
type DepthImage = Array3<f32>;
type NormalImage = Array3<f32>;


#[derive(Clone, Debug, Default)]
#[pyclass]
struct View {
    color: ColorImage,
    depth: DepthImage,
    normal: NormalImage,

    #[pyo3(get, set)]
    view_from_world: [[f32; 4]; 4],

    #[pyo3(get, set)]
    fovy: f32,
}

#[pymethods]
impl View {
    // TODO: upgrade rust-numpy to latest once pyo3 0.22 support is available
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let color_list: Vec<_> = self.color.iter().map(|&v| v.into_py(py)).collect();
        PyList::new_bound(py, color_list)
    }

    #[getter]
    fn depth<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let depth_list: Vec<_> = self.depth.iter().map(|&v| v.into_py(py)).collect();
        PyList::new_bound(py, depth_list)
    }

    #[getter]
    fn normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let normal_list: Vec<_> = self.normal.iter().map(|&v| v.into_py(py)).collect();
        PyList::new_bound(py, normal_list)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[derive(Debug, Default, Resource)]
#[pyclass]
struct Sample {
    views: Vec<View>,
}

#[pymethods]
impl Sample {
    #[getter]
    fn views<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let views_list: Vec<_> = self.views.iter().map(|v| v.clone().into_py(py)).collect();
        PyList::new_bound(py, views_list)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}


#[derive(Debug, Default, Resource, Reflect)]
struct SamplerState {
    frames: u32,
    render_modes: Vec<RenderMode>,
}


static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<Sample>>>> = OnceCell::new();
static SAMPLE_SENDER: OnceCell<Sender<Sample>> = OnceCell::new();


// TODO: create Dataloader torch class (or a render `n` frames and return capture fn, used within a python wrapper dataloader, wrapper requires setup.py to include the python module)


pub fn setup_and_run_app(
    new_thread: bool,
    override_args: Option<BevyZeroverseConfig>,
) {
    let ready = Arc::new(AtomicBool::new(false));

    let startup = {
        let ready = Arc::clone(&ready);

        move || {
            let mut app = viewer_app(override_args);

            app.init_resource::<Sample>();
            app.init_resource::<SamplerState>();
            app.add_systems(PreUpdate, sample_stream);

            ready.store(true, Ordering::Release);

            app.run();
        }
    };

    if new_thread {
        info!("starting bevy_zeroverse in a new thread");

        thread::spawn(startup);

        while !ready.load(Ordering::Acquire) {
            thread::yield_now();
        }
    } else {
        startup();
    }
}


#[pyfunction]
#[pyo3(signature = (override_args=None, asset_root=None))]
fn initialize(
    py: Python<'_>,
    override_args: Option<BevyZeroverseConfig>,
    asset_root: Option<String>,
) {
    if let Some(asset_root) = asset_root {
        std::env::set_var("BEVY_ASSET_ROOT", asset_root);
    } else {
        std::env::set_var("BEVY_ASSET_ROOT", std::env::current_dir().unwrap());
    }

    let (
        sender,
        receiver,
    ) = mpsc::channel();

    SAMPLE_RECEIVER.set(Arc::new(Mutex::new(receiver))).unwrap();
    SAMPLE_SENDER.set(sender).unwrap();

    let mut views = Vec::new();

    for _ in 0..9 {
        let color = Array3::<f32>::zeros((1080, 1920, 3));
        let depth = Array3::<f32>::zeros((1080, 1920, 1));
        let normal = Array3::<f32>::zeros((1080, 1920, 3));

        let view = View {
            color,
            depth,
            normal,
            view_from_world: [[0.0; 4]; 4],
            fovy: 60.0,
        };

        views.push(view);
    }

    let sample = Sample {
        views,
    };

    let sender = SAMPLE_SENDER.get().unwrap();
    sender.send(sample).unwrap();

    py.allow_threads(|| {
        setup_and_run_app(true, override_args);
    });
}


// TODO: keep a global sample before sending over mpsc (when sending, clear the resource to avoid copy)
// TODO: update the global sample depending on rendering mode
// TODO: update SamplerState -> SamplerState to include render_mode cycling

fn sample_stream(
    mut buffered_sample: ResMut<Sample>,
    mut state: ResMut<SamplerState>,
    cameras: Query<(
        &GlobalTransform,
        &Projection,
        &ImageCopier,
    )>,
    images: Res<Assets<Image>>,
    render_mode: Res<RenderMode>,
) {
    if state.frames > 0 {
        state.frames -= 1;
        return;
    }

    if buffered_sample.views.len() != cameras.iter().count() {
        buffered_sample.views.clear();

        for _ in cameras.iter() {
            buffered_sample.views.push(View::default());
        }
    }

    for (i, (camera_transform, projection, image_copier)) in cameras.iter().enumerate() {
        let view = &mut buffered_sample.views[i];

        view.fovy = match projection {
            Projection::Perspective(perspective) => perspective.fov,
            Projection::Orthographic(_) => panic!("orthographic projection not supported"),
        };

        let view_from_world = camera_transform.compute_matrix().to_cols_array_2d();
        view.view_from_world = view_from_world;

        let cpu_image = images.get(&image_copier.dst_image).unwrap();
        let size = cpu_image.size();

        let dynamic_cpu_image = cpu_image
            .clone()
            .try_into_dynamic()
            .unwrap();

        match *render_mode {
            RenderMode::Color => {
                let color_image = dynamic_cpu_image
                    .as_rgb32f()
                    .unwrap();

                view.color = Array3::<f32>::from_shape_vec(
                    (size.y as usize, size.x as usize, 3),
                    color_image.to_vec(),
                ).unwrap();
            },
            RenderMode::Depth => {
                fn rgba16f_to_r32f(image: &DynamicImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
                    let img = image.as_rgb16().unwrap();
                    let (width, height) = img.dimensions();
                    let buffer: Vec<f32> = img.pixels().map(|p| p.0[0] as f32).collect();
                    ImageBuffer::from_raw(width, height, buffer).unwrap()
                }

                let depth_image = rgba16f_to_r32f(&dynamic_cpu_image);

                view.depth = Array3::<f32>::from_shape_vec(
                    (size.y as usize, size.x as usize, 1),
                    depth_image.to_vec(),
                ).unwrap();
            },
            RenderMode::Normal => {
                let normal_image = dynamic_cpu_image
                    .as_rgb32f()
                    .unwrap();

                view.normal = Array3::<f32>::from_shape_vec(
                    (size.y as usize, size.x as usize, 3),
                    normal_image.to_vec(),
                ).unwrap();
            },
        }
    }


    // TODO: progress to next render_mode, reset frame counter (add a frame count reset to SamplerState)


    if state.render_modes.is_empty() {
        let views = std::mem::take(&mut buffered_sample.views);
        let sample = Sample {
            views,
        };

        let sender = SAMPLE_SENDER.get().unwrap();
        sender.send(sample).unwrap();
    }
}


// TODO: support batch dimension (e.g. single array allocation for multiple samples)
// TODO: add options to bevy_zeroverse.next (e.g. render_mode, scene parameters, etc.)
#[pyfunction]
fn next() -> PyResult<Sample> {
    let receiver = SAMPLE_RECEIVER.get().unwrap();
    let receiver = receiver.lock().unwrap();

    let timeout = Duration::from_secs(10);

    match receiver.recv_timeout(timeout) {
        Ok(sample) => Ok(sample),
        Err(RecvTimeoutError::Timeout) => {
            Err(PyTimeoutError::new_err("receive operation timed out"))
        }
        Err(RecvTimeoutError::Disconnected) => {
            Err(PyTimeoutError::new_err("channel disconnected"))
        }
    }
}


#[pymodule]
fn bevy_zeroverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BevyZeroverseConfig>()?;
    m.add_class::<Sample>()?;
    m.add_class::<View>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;
    Ok(())
}
