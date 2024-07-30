use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
};

// use bevy::prelude::*;
use pyo3::prelude::*;

use ::bevy_zeroverse::app::viewer_app;


// TODO: create Dataloader torch class (or a render `n` frames and return capture fn, used within a python wrapper dataloader, wrapper requires setup.py to include the python module)


pub fn setup_and_run_app(
    new_thread: bool,
) {
    let ready = Arc::new(AtomicBool::new(false));

    let startup = {
        let ready = Arc::clone(&ready);

        move || {
            let mut app = viewer_app(None);

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


// TODO: add BevyZeroverseViewer struct parameter
#[pyfunction]
fn initialize(
    py: Python<'_>,
) {
    py.allow_threads(|| {
        setup_and_run_app(true);
    });
}


#[pymodule]
fn bevy_zeroverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    Ok(())
}
