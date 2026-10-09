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
static INDOOR_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct View {
    #[pyo3(get, set)]
    pub calibration: Option<String>,
    #[pyo3(get, set)]
    pub trajectory_progress: Option<f32>,
    #[pyo3(get, set)]
    pub time_seconds: Option<f32>,
    pub color: Vec<u8>,
    pub depth: Vec<u8>,
    pub normal: Vec<u8>,
    pub semantic: Vec<u8>,
    pub optical_flow: Vec<u8>,
    pub motion_vectors: Vec<u8>,
    pub co_visibility: Vec<u8>,
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
            calibration: value
                .calibration
                .map(|c| serde_json::to_string(&c).expect("valid calibration")),
            trajectory_progress: value.trajectory_progress,
            time_seconds: value.time_seconds,
            color: value.color,
            depth: value.depth,
            normal: value.normal,
            semantic: value.semantic,
            optical_flow: value.optical_flow,
            motion_vectors: value.motion_vectors,
            co_visibility: value.co_visibility,
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
    fn semantic<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.semantic)
    }

    #[getter]
    fn optical_flow<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.optical_flow)
    }

    #[getter]
    fn motion_vectors<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.motion_vectors)
    }

    #[getter]
    fn co_visibility<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.co_visibility)
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
    pub instance_id: Option<i64>,
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
            instance_id: value.instance_id,
            center: value.center,
            scale: value.scale,
            rotation: value.rotation,
            class_name: value.class_name,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct HumanPose {
    #[pyo3(get, set)]
    pub bone_positions: Vec<[f32; 3]>,
    #[pyo3(get, set)]
    pub bone_rotations: Vec<[f32; 4]>,
}

impl From<core_sample::HumanPoseSample> for HumanPose {
    fn from(value: core_sample::HumanPoseSample) -> Self {
        HumanPose {
            bone_positions: value.bone_positions,
            bone_rotations: value.bone_rotations,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug, Default, Resource)]
pub struct Sample {
    #[pyo3(get)]
    pub indoor_manifest: Option<String>,
    #[pyo3(get)]
    pub indoor_render_metadata: Option<String>,
    #[pyo3(get)]
    pub co_visibility_metadata: Option<String>,
    #[pyo3(get)]
    pub color_encoding: String,
    #[pyo3(get)]
    pub annotation_precision: String,
    #[pyo3(get)]
    pub annotation_glass: String,
    pub views: Vec<View>,

    #[pyo3(get, set)]
    pub view_dim: u32,

    /// min and max corners of the axis-aligned bounding box
    #[pyo3(get, set)]
    pub aabb: [[f32; 3]; 2],

    #[pyo3(get, set)]
    pub object_obbs: Vec<ObjectObb>,

    #[pyo3(get, set)]
    pub human_poses: Vec<HumanPose>,

    #[pyo3(get, set)]
    pub human_instance_ids: Vec<i64>,

    #[pyo3(get, set)]
    pub human_pose_steps: Vec<Vec<HumanPose>>,

    #[pyo3(get, set)]
    pub human_bone_names: Vec<String>,

    #[pyo3(get, set)]
    pub human_bone_parents: Vec<i64>,

    #[pyo3(get, set)]
    pub ovoxel: Option<Ovoxel>,
}

impl From<core_sample::Sample> for Sample {
    fn from(value: core_sample::Sample) -> Self {
        let views = value.views.into_iter().map(View::from).collect();
        Sample {
            co_visibility_metadata: value
                .co_visibility_metadata
                .as_ref()
                .map(|m| serde_json::to_string(m).expect("valid co-visibility metadata")),
            indoor_render_metadata: value
                .indoor_render_metadata
                .as_ref()
                .map(|metadata| serde_json::to_string(metadata).expect("valid render provenance")),
            annotation_glass: match value.annotation_glass {
                bevy_zeroverse::render::glass::AnnotationGlass::Surface => "surface",
                bevy_zeroverse::render::glass::AnnotationGlass::Through => "through",
            }
            .into(),
            annotation_precision: match value.annotation_precision {
                core_sample::AnnotationPrecision::Float16Hdr => "float16_hdr",
                core_sample::AnnotationPrecision::Float32Geometry => "float32_geometry",
            }
            .into(),
            indoor_manifest: value
                .indoor
                .as_ref()
                .map(|m| serde_json::to_string(m).expect("valid indoor manifest")),
            color_encoding: match value.color_encoding {
                bevy_zeroverse::render::color::ColorEncoding::Legacy => "legacy",
                bevy_zeroverse::render::color::ColorEncoding::TonemappedLinear => {
                    "tonemapped_linear"
                }
                bevy_zeroverse::render::color::ColorEncoding::Srgb => "srgb",
            }
            .to_owned(),
            views,
            view_dim: value.view_dim,
            aabb: value.aabb,
            object_obbs: value.object_obbs.into_iter().map(ObjectObb::from).collect(),
            human_poses: value.human_poses.into_iter().map(HumanPose::from).collect(),
            human_instance_ids: value.human_instance_ids,
            human_pose_steps: value
                .human_pose_steps
                .into_iter()
                .map(|step| step.into_iter().map(HumanPose::from).collect())
                .collect(),
            human_bone_names: value.human_bone_names,
            human_bone_parents: value.human_bone_parents,
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
) -> PyResult<()> {
    if let Some(config) = &override_args {
        config.validate_ovoxel().map_err(PyRuntimeError::new_err)?;
    }
    if INITIALIZED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(PyRuntimeError::new_err(
            "one Bevy engine is supported per process; create a fresh process for a different dataset configuration",
        ));
    }

    INDOOR_INITIALIZED.store(
        override_args
            .as_ref()
            .is_some_and(|args| args.scene_type == ZeroverseSceneType::ProceduralIndoor),
        Ordering::Release,
    );

    setup_globals(asset_root);

    py.detach(move || {
        setup_and_run_app(true, override_args);
    });
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (indoor_seed=None))]
pub fn next(py: Python<'_>, indoor_seed: Option<u64>) -> PyResult<Sample> {
    if indoor_seed.is_some() && !INDOOR_INITIALIZED.load(Ordering::Acquire) {
        return Err(PyRuntimeError::new_err(
            "indoor_seed requests require a procedural_indoor configuration",
        ));
    }
    py.detach(|| {
        let sample_receiver = channels::sample_receiver().ok_or_else(|| {
            PyRuntimeError::new_err("bevy_zeroverse_ffi not initialized; call initialize() first")
        })?;
        let sample_receiver = sample_receiver
            .lock()
            .map_err(|_| PyRuntimeError::new_err("sample receiver lock poisoned"))?;

        channels::app_frame_sender()
            .send(channels::AppFrameRequest {
                indoor_seed,
                ..Default::default()
            })
            .map_err(|_| PyRuntimeError::new_err("failed to request next frame from app"))?;
        let started = std::time::Instant::now();
        loop {
            if let Some(error) = channels::take_capture_failure() {
                return Err(PyRuntimeError::new_err(error));
            }
            match sample_receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(sample) => return Ok(Sample::from(sample)),
                Err(RecvTimeoutError::Timeout) if started.elapsed() < Duration::from_secs(300) => {}
                Err(RecvTimeoutError::Timeout) => {
                    return Err(PyTimeoutError::new_err("receive operation timed out"));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(PyRuntimeError::new_err("channel disconnected"));
                }
            }
        }
    })
}

#[pymodule]
pub fn bevy_zeroverse_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add(
        "CAMERA_CALIBRATION_METADATA",
        bevy_zeroverse::calibration::TENSOR_METADATA,
    )?;

    m.add_class::<BevyZeroverseConfig>()?;
    m.add_class::<Playback>()?;
    m.add_class::<PlaybackMode>()?;
    m.add_class::<OvoxelMode>()?;
    m.add_class::<RenderMode>()?;
    m.add_class::<bevy_zeroverse::render::glass::AnnotationGlass>()?;
    m.add_class::<bevy_zeroverse::render::depth::DepthFormat>()?;
    m.add_class::<ZeroverseSceneType>()?;
    m.add_class::<bevy_zeroverse::scene::procedural_indoor::layout::IndoorLayout>()?;
    m.add_class::<bevy_zeroverse::scene::procedural_indoor::IndoorQuality>()?;

    m.add_class::<Ovoxel>()?;
    m.add_class::<Sample>()?;
    m.add_class::<View>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;
    Ok(())
}
