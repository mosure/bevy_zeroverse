use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::RecvTimeoutError,
    },
    time::Duration,
};

use bevy::prelude::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError},
    prelude::*,
    types::{PyBytes, PyList},
};

use ::bevy_zeroverse::{
    app::{BevyZeroverseConfig, OvoxelMode},
    camera::{Playback, PlaybackMode},
    headless::{setup_and_run_app, setup_globals},
    io::channels,
    render::RenderMode,
    sample as core_sample,
    sample::OvoxelSample,
    scene::ZeroverseSceneType,
};

static INITIALIZED: AtomicBool = AtomicBool::new(false);

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct View {
    pub color: Vec<u8>,
    pub depth: Vec<u8>,
    pub normal: Vec<u8>,
    pub optical_flow: Vec<u8>,
    pub position: Vec<u8>,

    #[pyo3(get, set)]
    pub world_from_view: [[f32; 4]; 4],

    #[pyo3(get, set)]
    pub fovy: f32,

    #[pyo3(get, set)]
    pub near: f32,

    #[pyo3(get, set)]
    pub far: f32,

    #[pyo3(get, set)]
    pub time: f32,
}

impl From<core_sample::View> for View {
    fn from(value: core_sample::View) -> Self {
        View {
            color: value.color,
            depth: value.depth,
            normal: value.normal,
            optical_flow: value.optical_flow,
            position: value.position,
            world_from_view: value.world_from_view,
            fovy: value.fovy,
            near: value.near,
            far: value.far,
            time: value.time,
        }
    }
}

#[pymethods]
impl View {
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.color)
    }

    #[getter]
    fn depth<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.depth)
    }

    #[getter]
    fn normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.normal)
    }

    #[getter]
    fn optical_flow<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.optical_flow)
    }

    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.position)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct Ovoxel {
    #[pyo3(get, set)]
    pub coords: Vec<[u32; 3]>,
    #[pyo3(get, set)]
    pub dual_vertices: Vec<[u8; 3]>,
    #[pyo3(get, set)]
    pub intersected: Vec<u8>,
    #[pyo3(get, set)]
    pub base_color: Vec<[u8; 4]>,
    #[pyo3(get, set)]
    pub semantics: Vec<u16>,
    #[pyo3(get, set)]
    pub semantic_labels: Vec<String>,
    #[pyo3(get, set)]
    pub resolution: u32,
    #[pyo3(get, set)]
    pub aabb: [[f32; 3]; 2],
}

impl From<OvoxelSample> for Ovoxel {
    fn from(value: OvoxelSample) -> Self {
        Ovoxel {
            coords: value.coords,
            dual_vertices: value.dual_vertices,
            intersected: value.intersected,
            base_color: value.base_color,
            semantics: value.semantics,
            semantic_labels: value.semantic_labels,
            resolution: value.resolution,
            aabb: value.aabb,
        }
    }
}

#[pymethods]
impl Ovoxel {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct ObjectObb {
    #[pyo3(get, set)]
    pub center: [f32; 3],
    #[pyo3(get, set)]
    pub scale: [f32; 3],
    #[pyo3(get, set)]
    pub rotation: [f32; 4],
    #[pyo3(get, set)]
    pub class_name: String,
}

impl From<core_sample::ObjectObbSample> for ObjectObb {
    fn from(value: core_sample::ObjectObbSample) -> Self {
        ObjectObb {
            center: value.center,
            scale: value.scale,
            rotation: value.rotation,
            class_name: value.class_name,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Default, Resource)]
pub struct Sample {
    pub views: Vec<View>,

    #[pyo3(get, set)]
    pub view_dim: u32,

    /// min and max corners of the axis-aligned bounding box
    #[pyo3(get, set)]
    pub aabb: [[f32; 3]; 2],

    #[pyo3(get, set)]
    pub object_obbs: Vec<ObjectObb>,

    #[pyo3(get, set)]
    pub ovoxel: Option<Ovoxel>,
}

impl From<core_sample::Sample> for Sample {
    fn from(value: core_sample::Sample) -> Self {
        let views = value.views.into_iter().map(View::from).collect();
        Sample {
            views,
            view_dim: value.view_dim,
            aabb: value.aabb,
            object_obbs: value.object_obbs.into_iter().map(ObjectObb::from).collect(),
            ovoxel: value.ovoxel.map(Ovoxel::from),
        }
    }
}

#[pymethods]
impl Sample {
    #[getter]
    fn views<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let views_list: Vec<_> = self
            .views
            .iter()
            .cloned()
            .map(|v| Py::new(py, v))
            .collect::<PyResult<_>>()?;
        PyList::new(py, views_list)
    }

    fn take_views<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let views = std::mem::take(&mut self.views);
        let py_views: Vec<_> = views
            .into_iter()
            .map(|view| Py::new(py, view))
            .collect::<PyResult<_>>()?;
        PyList::new(py, py_views)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}

#[pyfunction]
#[pyo3(signature = (override_args=None, asset_root=None))]
pub fn initialize(
    py: Python<'_>,
    override_args: Option<BevyZeroverseConfig>,
    asset_root: Option<String>,
) {
    if INITIALIZED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        warn!("bevy_zeroverse_ffi.initialize called more than once; ignoring duplicate request");
        return;
    }

    setup_globals(asset_root);

    py.detach(move || {
        setup_and_run_app(true, override_args);
    });
}

#[pyfunction]
pub fn next(py: Python<'_>) -> PyResult<Sample> {
    {
        let app_frame_sender = channels::app_frame_sender();

        app_frame_sender
            .send(())
            .map_err(|_| PyRuntimeError::new_err("failed to request next frame from app"))?;
    }

    py.detach(|| {
        let sample_receiver = channels::sample_receiver().ok_or_else(|| {
            PyRuntimeError::new_err("bevy_zeroverse_ffi not initialized; call initialize() first")
        })?;
        let sample_receiver = sample_receiver
            .lock()
            .map_err(|_| PyRuntimeError::new_err("sample receiver lock poisoned"))?;

        let timeout = Duration::from_secs(300);

        match sample_receiver.recv_timeout(timeout) {
            Ok(sample) => Ok(Sample::from(sample)),
            Err(RecvTimeoutError::Timeout) => {
                Err(PyTimeoutError::new_err("receive operation timed out"))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(PyRuntimeError::new_err("channel disconnected"))
            }
        }
    })
}

#[pymodule]
pub fn bevy_zeroverse_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BevyZeroverseConfig>()?;
    m.add_class::<Playback>()?;
    m.add_class::<PlaybackMode>()?;
    m.add_class::<OvoxelMode>()?;
    m.add_class::<RenderMode>()?;
    m.add_class::<ZeroverseSceneType>()?;

    m.add_class::<Ovoxel>()?;
    m.add_class::<Sample>()?;
    m.add_class::<View>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;
    Ok(())
}
