fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let inputs =
        bevy_zeroverse_capture::provenance::publisher_inputs(root).expect("publisher build inputs");
    for name in inputs.keys() {
        println!("cargo:rerun-if-changed={name}");
    }
    for name in ["src", "templates", "fonts"] {
        println!("cargo:rerun-if-changed={name}");
    }
    println!(
        "cargo:rustc-env=PUBLICATION_SOURCE_SHA256={}",
        bevy_zeroverse_capture::provenance::source_digest(&inputs)
    );
}
