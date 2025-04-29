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
    camera::{
        Playback,
        PlaybackMode,
    },
    io::image_copy::ImageCopier,
    render::RenderMode,
    scene::{
        RegenerateSceneEvent,
        SceneAabb,
        ZeroverseSceneRoot,
        ZeroverseSceneType,
    },
};


// TODO: move to src/sample.rs (or src/dataloader.rs) to support torch (python) and burn dataloaders

#[derive(Clone, Debug, Default)]
#[pyclass]
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

#[derive(Clone, Debug, Default, Resource)]
#[pyclass]
pub struct Sample {
    pub views: Vec<View>,

    #[pyo3(get, set)]
    pub view_dim: u32,

    /// min and max corners of the axis-aligned bounding box
    #[pyo3(get, set)]
    pub aabb: [[f32; 3]; 2],
}

#[pymethods]
impl Sample {
    #[getter]
    fn views<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        #[allow(deprecated)]
        let views_list: Vec<_> = self.views.iter().map(|v| v.clone().into_py(py)).collect();
        PyList::new(py, views_list).unwrap()
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{self:?}"))
    }
}


#[derive(Debug, Resource, Reflect)]
#[reflect(Resource)]
pub struct SamplerState {
    pub enabled: bool,
    pub regenerate_scene: bool,
    pub frames: u32,
    pub render_modes: Vec<RenderMode>,
    pub step: u32,
    pub timesteps: Vec<f32>,
    pub warmup_frames: u32,
}

impl Default for SamplerState {
    fn default() -> Self {
        SamplerState {
            enabled: true,
            regenerate_scene: true,
            frames: SamplerState::FRAME_DELAY,
            render_modes: vec![
                RenderMode::Color,
                // RenderMode::Depth,
                // RenderMode::Normal,
                // RenderMode::OpticalFlow,
                // RenderMode::Position,
            ],
            step: 0,
            timesteps: vec![0.125],
            warmup_frames: SamplerState::WARMUP_FRAME_DELAY,
        }
    }
}

impl SamplerState {
    const FRAME_DELAY: u32 = 3;
    const WARMUP_FRAME_DELAY: u32 = 2;

    pub fn inference_only() -> Self {
        SamplerState {
            enabled: true,
            regenerate_scene: false,
            frames: 1,
            render_modes: vec![
                RenderMode::Color,
            ],
            step: 0,
            timesteps: vec![0.125],
            warmup_frames: 0,
        }
    }

    pub fn reset(&mut self) {
        self.frames = SamplerState::FRAME_DELAY;
    }
}


pub static APP_FRAME_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
pub static APP_FRAME_SENDER: OnceCell<Sender<()>> = OnceCell::new();

pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<Sample>>>> = OnceCell::new();
pub static SAMPLE_SENDER: OnceCell<Sender<Sample>> = OnceCell::new();


fn signaled_runner(mut app: App) -> AppExit {
    app.finish();
    app.cleanup();

    // prime update schedule at initialization
    for _ in 0..4 {
        app.update();
    }

    if let Some(exit) = app.should_exit() {
        return exit;
    }

    loop {
        if let Some(receiver) = APP_FRAME_RECEIVER.get() {
            let receiver = receiver.lock().unwrap();
            if receiver.recv().is_ok() {
                let args = app.world().resource::<BevyZeroverseConfig>();

                let render_modes = args.render_modes.clone();

                let timesteps = (1..args.playback_steps)
                    .map(|i| {
                        let x = i as f32 * args.playback_step;
                        if x > 1.0 {
                            panic!("timestep value {x} has range [0.0, 1.0]");
                        }
                        x
                    })
                    .collect();

                app.insert_resource(SamplerState {
                    timesteps,
                    render_modes,
                    ..default()
                });

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


pub fn create_app(
    override_args: Option<BevyZeroverseConfig>,
    set_runner: bool,
) -> App {
    let mut app = viewer_app(override_args.clone());

    app.init_resource::<Sample>();

    // initialize to disabled state
    app.insert_resource(SamplerState {
        enabled: false,
        render_modes: override_args.unwrap_or_default().render_modes,
        ..Default::default()
    });
    app.register_type::<SamplerState>();

    app.add_systems(PreUpdate, sample_stream);

    if set_runner {
        app.set_runner(signaled_runner);
    }

    app
}


pub fn setup_and_run_app(
    new_thread: bool,
    override_args: Option<BevyZeroverseConfig>,
) {
    let ready = Arc::new(AtomicBool::new(false));

    let startup = {
        let ready = Arc::clone(&ready);

        move || {
            let mut app = create_app(override_args, true);
            ready.store(true, Ordering::Release);
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


#[allow(clippy::too_many_arguments)]
fn sample_stream(
    args: Res<BevyZeroverseConfig>,
    mut buffered_sample: ResMut<Sample>,
    mut state: ResMut<SamplerState>,
    cameras: Query<(
        &GlobalTransform,
        &Projection,
        &ImageCopier,
    )>,
    scene: Query<(&ZeroverseSceneRoot, &SceneAabb)>,
    images: Res<Assets<Image>>,
    mut render_mode: ResMut<RenderMode>,
    mut playback: ResMut<Playback>,
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

    let camera_count = cameras.iter().count();
    let view_count = camera_count * args.playback_steps as usize;
    if buffered_sample.views.len() != view_count {
        buffered_sample.views.clear();

        for _ in 0..view_count {
            buffered_sample.views.push(View::default());
        }
    }

    let scene_aabb = scene.single().unwrap().1;
    buffered_sample.aabb = [
        scene_aabb.min.into(),
        scene_aabb.max.into(),
    ];

    let write_to = state.render_modes.remove(0);

    for (
        i,
        (
            camera_transform,
            projection,
            image_copier
        )
    ) in cameras.iter().enumerate() {
        let view_idx = i + camera_count * state.step as usize;
        let view = &mut buffered_sample.views[view_idx];

        view.time = playback.progress;

        match projection {
            Projection::Perspective(perspective) => {
                view.fovy = perspective.fov;
                view.near = perspective.near;
                view.far = perspective.far;
            }
            Projection::Orthographic(_) => panic!("orthographic projection not supported"),
            Projection::Custom(_) => panic!("custom projection not supported"),
        };

        let world_from_view = camera_transform.compute_matrix().to_cols_array_2d();
        view.world_from_view = world_from_view;

        let image_data = images
            .get(&image_copier.dst_image)
            .unwrap()
            .clone()
            .data
            .unwrap();

        match write_to {
            RenderMode::Color => view.color = image_data,
            RenderMode::Depth => view.depth = image_data,
            RenderMode::MotionVectors => panic!("motion vector rendering not supported"),
            RenderMode::Normal => view.normal = image_data,
            RenderMode::OpticalFlow => view.optical_flow = image_data,
            RenderMode::Position => view.position = image_data,
            RenderMode::Semantic => panic!("semantic rendering not supported"),
        }
    }

    if !state.render_modes.is_empty() {
        *render_mode = state.render_modes[0].clone();
        state.reset();
        return;
    }
    *render_mode = args.render_modes[0].clone();

    if !state.timesteps.is_empty() {
        playback.progress = state.timesteps.remove(0);
        state.step += 1;
        state.render_modes = args.render_modes.clone();
        state.reset();
        // TODO: set fixed previous cameras for optical flow across timesteps
        return;
    }

    playback.progress = 0.0;

    let views = std::mem::take(&mut buffered_sample.views);
    let sample: Sample = Sample {
        views,
        view_dim: camera_count as u32,
        aabb: buffered_sample.aabb,
    };

    let sender = SAMPLE_SENDER.get().unwrap();
    sender.send(sample).unwrap();

    if state.regenerate_scene {
        regenerate_event.write(RegenerateSceneEvent);
    }

    state.enabled = false;
}


pub fn setup_globals(
    asset_root: Option<String>,
) {
    if APP_FRAME_RECEIVER.get().is_some() {
        return;
    }

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
}


// TODO: add process idx, disable logging on worker idx > 0
#[pyfunction]
#[pyo3(signature = (override_args=None, asset_root=None))]
pub fn initialize(
    py: Python<'_>,
    override_args: Option<BevyZeroverseConfig>,
    asset_root: Option<String>,
) {
    setup_globals(asset_root);

    py.allow_threads(|| {
        setup_and_run_app(true, override_args);
    });
}


// TODO: add options to bevy_zeroverse.next (e.g. render_mode, scene parameters, etc.)
#[pyfunction]
pub fn next(
    py: Python<'_>,
) -> PyResult<Sample> {
    {
        let app_frame_sender = APP_FRAME_SENDER.get().unwrap();
        app_frame_sender.send(()).unwrap();
    }

    py.allow_threads(|| {
        let sample_receiver = SAMPLE_RECEIVER.get().unwrap();
        let sample_receiver = sample_receiver.lock().unwrap();

        let timeout = Duration::from_secs(120);

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
pub fn bevy_zeroverse_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BevyZeroverseConfig>()?;
    m.add_class::<Playback>()?;
    m.add_class::<PlaybackMode>()?;
    m.add_class::<RenderMode>()?;
    m.add_class::<ZeroverseSceneType>()?;

    m.add_class::<Sample>()?;
    m.add_class::<View>()?;

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(next, m)?)?;
    Ok(())
}
