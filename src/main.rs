#[cfg(not(target_arch = "wasm32"))]
fn main() -> bevy::app::AppExit {
    bevy_zeroverse::app::run_native_viewer()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // The Studio worker instantiates the same compiled module for CPU-only jobs.
    console_error_panic_hook::set_once();
    if web_sys::window().is_none() {
        return;
    }
    bevy_zeroverse::app::viewer_app(None, None).run();
}
