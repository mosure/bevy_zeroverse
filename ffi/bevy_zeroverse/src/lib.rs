use bevy::prelude::*;
use pyo3::prelude::*;


// TODO: create Dataloader torch class (or a render `n` frames and return capture fn, used within a python wrapper dataloader, wrapper requires setup.py to include the python module)


pub fn setup_and_run_app(
    new_thread: bool,
) {
    let ready = Arc::new(AtomicBool::new(false));

    let mut startup = {
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

        return;
    }

    startup();
}


// TODO: add BevyZeroverseViewer struct parameter
#[pyfunction]
fn main(new_thread: bool) {
    setup_and_run_app(new_thread);
}


#[pymodule]
fn bevy_zeroverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(main, m)?)?;
    Ok(())
}
