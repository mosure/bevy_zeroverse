use std::{
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

use crate::{
    app::{viewer_app, BevyZeroverseConfig},
    io::channels,
    sample::{configure_sampler, SamplerState},
};
use bevy::prelude::*;

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
        if let Some(receiver) = channels::app_frame_receiver() {
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
