fn main() {
    use bevy_zeroverse_capture::provenance::{source_digest, source_inputs};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let inputs = source_inputs(root).expect("hash generator build inputs");
    for path in inputs.keys() {
        println!("cargo:rerun-if-changed={path}");
    }
    // Watch directories too, so additions/removals invalidate the identity.
    for path in [
        "src",
        "crates/capture/src",
        "assets/shaders",
        "assets/embedded",
        "third_party/wgpu-core/src",
        "third_party/wgpu-hal/src",
    ] {
        if root.join(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!(
        "cargo:rustc-env=ZEROVERSE_SOURCE_SHA256={}",
        source_digest(&inputs)
    );
}
