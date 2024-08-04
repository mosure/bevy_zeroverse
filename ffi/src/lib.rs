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
use once_cell::sync::OnceCell;
use pyo3::{
    prelude::*,
    exceptions::PyTimeoutError,
    types::{
        PyBytes,
        PyList,
    },
};

use ::bevy_zeroverse::{
    app::{
        viewer_app,
        BevyZeroverseConfig,
    },
    io::image_copy::ImageCopier,
    render::RenderMode,
    scene::{
        RegenerateSceneEvent,
        ZeroverseSceneType,
    },
};


// TODO: move to src/sample.rs (or src/dataloader.rs) to support torch (python) and burn dataloaders

#[derive(Clone, Debug, Default)]
#[pyclass]
struct View {
    color: Vec<u8>,
    depth: Vec<u8>,
    normal: Vec<u8>,

    #[pyo3(get, set)]
    view_from_world: [[f32; 4]; 4],

    #[pyo3(get, set)]
    fovy: f32,
}

#[pymethods]
impl View {
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.color)
    }

    #[getter]
    fn depth<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.depth)
    }

    #[getter]
    fn normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.normal)
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


#[derive(Debug, Resource, Reflect)]
struct SamplerState {
    enabled: bool,
    frames: u32,
    render_modes: Vec<RenderMode>,
    warmup_frames: u32,
}

impl Default for SamplerState {
    fn default() -> Self {
        SamplerState {
            enabled: true,
            frames: SamplerState::FRAME_DELAY,
            render_modes: vec![
                RenderMode::Color,
                RenderMode::Depth,
                RenderMode::Normal,
            ],
            warmup_frames: SamplerState::WARMUP_FRAME_DELAY,
        }
    }
}

impl SamplerState {
    const FRAME_DELAY: u32 = 3;
    const WARMUP_FRAME_DELAY: u32 = 5;

    fn cycle_render_mode(
        &mut self,
        mut render_mode: ResMut<RenderMode>,
    ) {
        self.render_modes.retain(|mode| *mode != *render_mode);

        if self.render_modes.is_empty() {
            *render_mode = RenderMode::Color;
            self.enabled = false;
        } else {
            *render_mode = self.render_modes.pop().unwrap();
            self.enabled = true;
        }

        self.frames = SamplerState::FRAME_DELAY;
    }

    fn is_complete(&self) -> bool {
        self.render_modes.is_empty()
    }
}


static APP_FRAME_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
static APP_FRAME_SENDER: OnceCell<Sender<()>> = OnceCell::new();

static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<Sample>>>> = OnceCell::new();
static SAMPLE_SENDER: OnceCell<Sender<Sample>> = OnceCell::new();


fn signaled_runner(mut app: App) -> AppExit {
    app.finish();
    app.cleanup();

    loop {
        if let Some(receiver) = APP_FRAME_RECEIVER.get() {
            let receiver = receiver.lock().unwrap();
            if receiver.try_recv().is_ok() {
                app.insert_resource(SamplerState::default());

                loop {
                    app.update();
                    if let Some(exit) = app.should_exit() {
                        return exit;
                    }

                    if !app.world().resource::<SamplerState>().enabled {
                        break;
                    }
                }
            }
        }
    }
}


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

            app.set_runner(signaled_runner);
            app.run();
        }
    };

    if new_thread {
        thread::spawn(startup);

        while !ready.load(Ordering::Acquire) {
            thread::yield_now();
        }
    } else {
        startup();
    }
}


// TODO: add process idx, disable logging on worker idx > 0
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
        app_sender,
        app_receiver,
    ) = mpsc::channel();

    APP_FRAME_RECEIVER.set(Arc::new(Mutex::new(app_receiver))).unwrap();
    APP_FRAME_SENDER.set(app_sender).unwrap();

    let (
        sample_sender,
        sample_receiver,
    ) = mpsc::channel();

    SAMPLE_RECEIVER.set(Arc::new(Mutex::new(sample_receiver))).unwrap();
    SAMPLE_SENDER.set(sample_sender).unwrap();

    py.allow_threads(|| {
        setup_and_run_app(true, override_args);
    });
}


fn sample_stream(
    mut buffered_sample: ResMut<Sample>,
    mut state: ResMut<SamplerState>,
    cameras: Query<(
        &GlobalTransform,
        &Projection,
        &ImageCopier,
    )>,
    images: Res<Assets<Image>>,
    render_mode: ResMut<RenderMode>,
    mut regenerate_event: EventWriter<RegenerateSceneEvent>,
) {
    if !state.enabled {
        return;
    }

    if cameras.iter().count() == 0 {
        return;
    }

    if state.warmup_frames > 0 {
        state.warmup_frames -= 1;
        return;
    }

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

        let image_data = images
            .get(&image_copier.dst_image)
            .unwrap()
            .clone()
            .data;

        match *render_mode {
            RenderMode::Color => view.color = image_data,
            RenderMode::Depth => view.depth = image_data,
            RenderMode::Normal => view.normal = image_data,
        }
    }

    if state.is_complete() {
        let views = std::mem::take(&mut buffered_sample.views);
        let sample = Sample {
            views,
        };

        let sender = SAMPLE_SENDER.get().unwrap();
        sender.send(sample).unwrap();

        regenerate_event.send(RegenerateSceneEvent);
    }

    state.cycle_render_mode(render_mode);
}


// TODO: add options to bevy_zeroverse.next (e.g. render_mode, scene parameters, etc.)
#[pyfunction]
fn next(
    py: Python<'_>,
) -> PyResult<Sample> {
    {
        let app_frame_sender = APP_FRAME_SENDER.get().unwrap();
        app_frame_sender.send(()).unwrap();
    }

    py.allow_threads(|| {
        let sample_receiver = SAMPLE_RECEIVER.get().unwrap();
        let sample_receiver = sample_receiver.lock().unwrap();

        let timeout = Duration::from_secs(30);

        match sample_receiver.recv_timeout(timeout) {
            Ok(sample) => Ok(sample),
            Err(RecvTimeoutError::Timeout) => {
                Err(PyTimeoutError::new_err("receive operation timed out"))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(PyTimeoutError::new_err("channel disconnected"))
            }
        }
    })
}


#[pymodule]
fn bevy_zeroverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BevyZeroverseConfig>()?;
    m.add_class::<ZeroverseSceneType>()?;

    m.add_class::<Sample>()?;
    m.add_class::<View>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;
    Ok(())
}
