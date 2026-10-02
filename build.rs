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
    let lock: toml::Value =
        toml::from_str(&std::fs::read_to_string(root.join("Cargo.lock")).unwrap()).unwrap();
    let mut packages = std::collections::BTreeMap::<String, Vec<String>>::new();
    for package in lock["package"].as_array().unwrap() {
        let name = package["name"].as_str().unwrap();
        if name == "bevy"
            || name == "burn"
            || name == "wgpu"
            || name.starts_with("burn_")
            || name.starts_with("bevy_zeroverse")
            || name == "bevy_burn_human"
        {
            packages
                .entry(name.into())
                .or_default()
                .push(package["version"].as_str().unwrap().into());
        }
    }
    let destination = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    std::fs::write(
        destination.join("capture-packages.json"),
        serde_json::to_vec(&packages).unwrap(),
    )
    .unwrap();
    println!(
        "cargo:rustc-env=ZEROVERSE_CAPTURE_TARGET={}",
        std::env::var("TARGET").unwrap()
    );
}
