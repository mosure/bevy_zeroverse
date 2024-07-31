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
use ndarray::{Array2, Array3};
use once_cell::sync::OnceCell;
use pyo3::{
    prelude::*,
    exceptions::PyTimeoutError,
    types::PyList,
};

use ::bevy_zeroverse::app::{
    viewer_app,
    BevyZeroverseConfig,
};


type ColorImage = Array3<u8>;
type DepthImage = Array2<f32>;
type NormalImage = Array3<f32>;


#[derive(Debug)]
#[derive(Clone)]
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

#[derive(Debug)]
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
        let color = Array3::<u8>::zeros((1080, 1920, 4));
        let depth = Array2::<f32>::zeros((1080, 1920));
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


// TODO: support batch dimension (e.g. single array allocation for multiple samples)
// TODO: add systems for pushing and receiving camera outputs + metadata to python
// TODO: add options to bevy_zeroverse.next (e.g. render_modes, scene parameters, etc.)
#[pyfunction]
fn next() -> PyResult<Sample> {
    // TODO: advance 'n' frames - requires app reference
    // TODO: capture frame - requires app system registration to write to textures and readback, triggered by app event after 'n' frames
    // TODO: send frame

    let receiver = SAMPLE_RECEIVER.get().unwrap();
    let receiver = receiver.lock().unwrap();

    let timeout = Duration::from_secs(5);

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
