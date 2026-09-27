//! Run the GPU regression directly without building every application binary.
#![recursion_limit = "256"]
#[cfg(not(target_arch = "wasm32"))]
#[path = "../tests/support/optical_flow.rs"]
mod reference;
fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    reference::run_validation();
}
