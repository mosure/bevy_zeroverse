use bevy_zeroverse::app::viewer_app;

pub fn main() {
    // The Studio worker instantiates the same compiled module for CPU-only jobs.
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        if web_sys::window().is_none() {
            return;
        }
    }
    viewer_app(None, None).run();
}
