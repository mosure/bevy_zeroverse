use std::{
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
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

static EXIT_REQUESTED: AtomicBool = AtomicBool::new(false);

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
        if EXIT_REQUESTED.load(Ordering::Acquire) {
            return AppExit::Success;
        }

        let recv_result = if let Some(receiver) = channels::app_frame_receiver() {
            let receiver = receiver.lock().unwrap();
            receiver.recv_timeout(Duration::from_millis(50))
        } else {
            thread::sleep(Duration::from_millis(50));
            continue;
        };

        match recv_result {
            Ok(()) => {
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
                    if EXIT_REQUESTED.load(Ordering::Acquire) {
                        return AppExit::Success;
                    }

                    app.update();
                    if let Some(exit) = app.should_exit() {
                        return exit;
                    }

                    if !app.world().resource::<SamplerState>().enabled {
                        break;
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if EXIT_REQUESTED.load(Ordering::Acquire) {
                    return AppExit::Success;
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
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
        thread::spawn(startup);

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
