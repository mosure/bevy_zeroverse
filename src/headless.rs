use std::{
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use crate::{
    app::{viewer_app, BevyZeroverseConfig},
    io::channels,
    sample::{configure_sampler, SamplerState},
};
use bevy::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
mod preflight;

static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);
static APP_THREAD: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);

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

    #[cfg(not(target_arch = "wasm32"))]
    let mut preflight = preflight::Preflight::default();

    loop {
        if EXIT_REQUESTED.load(Ordering::Acquire) {
            return AppExit::Success;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let timeout = preflight.poll_interval();
        #[cfg(target_arch = "wasm32")]
        let timeout = Duration::from_millis(50);
        let recv_result = if let Some(receiver) = channels::app_frame_receiver() {
            let receiver = receiver.lock().unwrap();
            receiver.recv_timeout(timeout)
        } else {
            thread::sleep(Duration::from_millis(50));
            continue;
        };

        match recv_result {
            Ok(request) => {
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(error) = preflight.begin_request(request.indoor_seed.is_some()) {
                    channels::report_capture_failure(error);
                    continue;
                }
                app.world_mut()
                    .resource_mut::<crate::scene::procedural_indoor::preparation::IndoorPrefetch>()
                    .depth = request.prefetch_indoor;
                app.world_mut()
                    .resource_mut::<crate::sample::CaptureFailure>()
                    .0 = None;
                if let Some(seed) = request.indoor_seed {
                    let config = app.world().resource::<BevyZeroverseConfig>();
                    assert_eq!(
                        config.scene_type,
                        crate::scene::ZeroverseSceneType::ProceduralIndoor,
                        "an indexed indoor capture requires the procedural_indoor scene"
                    );
                    crate::scene::procedural_indoor::reset_indoor_sequence(app.world_mut(), seed);
                    app.world_mut()
                        .write_message(crate::scene::RegenerateSceneEvent);
                    // Apply regeneration while the sampler is disabled, before scheduling modalities.
                    app.update();
                }
                let args = app.world().resource::<BevyZeroverseConfig>();

                let mut state = SamplerState::from_config(args);
                state.regenerate_scene = request.indoor_seed.is_none();
                app.insert_resource(state);

                let mut capture_failed = false;
                loop {
                    if EXIT_REQUESTED.load(Ordering::Acquire) {
                        return AppExit::Success;
                    }

                    update_capture(&mut app);
                    if let Some(exit) = app.should_exit() {
                        return exit;
                    }

                    if !app.world().resource::<SamplerState>().enabled {
                        if let Some(error) = app
                            .world_mut()
                            .resource_mut::<crate::sample::CaptureFailure>()
                            .0
                            .take()
                        {
                            capture_failed = true;
                            channels::report_capture_failure(error);
                        }
                        break;
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                if !capture_failed {
                    preflight.arm(app.world(), request);
                }
                #[cfg(target_arch = "wasm32")]
                let _ = capture_failed;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                #[cfg(not(target_arch = "wasm32"))]
                preflight.step(&mut app);
                if let Some(exit) = app.should_exit() {
                    return exit;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if EXIT_REQUESTED.load(Ordering::Acquire) {
                    return AppExit::Success;
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Advance a native headless capture, servicing registered readback callbacks
/// without repeatedly extracting/rendering while its exact packets are pending.
/// Returns whether an application update ran. Interactive/Web apps update normally.
/// Callers retain their ordinary exit, failure and wall-time timeout checks.
pub fn update_capture(app: &mut App) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::sample::native_readback::update(app, || EXIT_REQUESTED.load(Ordering::Acquire))
    }
    #[cfg(target_arch = "wasm32")]
    {
        app.update();
        true
    }
}

pub fn create_app(
    app: Option<App>,
    override_args: Option<BevyZeroverseConfig>,
    set_runner: bool,
) -> App {
    let mut app = viewer_app(app, override_args.clone());

    configure_sampler(
        &mut app,
        SamplerState {
            enabled: false,
            render_modes: override_args.clone().unwrap_or_default().render_modes,
            ..Default::default()
        },
    );

    if set_runner {
        app.set_runner(signaled_runner);
    }

    app
}

pub fn setup_and_run_app(new_thread: bool, override_args: Option<BevyZeroverseConfig>) {
    EXIT_REQUESTED.store(false, Ordering::Release);
    let ready = Arc::new(AtomicBool::new(false));

    let startup = {
        let ready = Arc::clone(&ready);

        move || {
            let mut app = create_app(None, override_args, true);
            ready.store(true, Ordering::Release);
            app.run();
        }
    };

    if new_thread {
        let mut app_thread = APP_THREAD.lock().unwrap();
        assert!(
            app_thread.is_none(),
            "a background renderer is already registered"
        );
        *app_thread = Some(thread::spawn(startup));
        drop(app_thread);

        while !ready.load(Ordering::Acquire) {
            thread::yield_now();
        }
    } else {
        startup();
    }
}

pub fn run_app_on_current_thread(
    override_args: Option<BevyZeroverseConfig>,
    ready: Option<Arc<AtomicBool>>,
) {
    EXIT_REQUESTED.store(false, Ordering::Release);
    let mut app = create_app(None, override_args, true);
    if let Some(ready) = ready {
        ready.store(true, Ordering::Release);
    }
    app.run();
}

pub fn request_exit() {
    EXIT_REQUESTED.store(true, Ordering::Release);
}

/// Stop and join the process-owned background renderer after its consumers finish.
/// A main-thread runner receives the exit request without a background join.
pub fn shutdown_app() -> thread::Result<()> {
    request_exit();
    let worker = APP_THREAD.lock().unwrap().take();
    match worker {
        Some(worker) if worker.thread().id() == thread::current().id() => {
            *APP_THREAD.lock().unwrap() = Some(worker);
            Err(Box::new(
                "the renderer must be joined by its consumer thread",
            ))
        }
        Some(worker) => worker.join(),
        None => Ok(()),
    }
}

pub fn setup_globals(asset_root: Option<String>) {
    if channels::channels_initialized() {
        return;
    }

    if let Some(asset_root) = asset_root {
        env::set_var("BEVY_ASSET_ROOT", asset_root);
    } else {
        env::set_var("BEVY_ASSET_ROOT", env::current_dir().unwrap());
    }

    channels::init_channels();
}
