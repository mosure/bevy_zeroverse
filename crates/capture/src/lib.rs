//! Wire contracts shared by the generator and its publication pipeline.
//! This crate has no renderer, GPU, model or browser initialization.
use serde::{Deserialize, Serialize};
pub mod calibration;

pub const GENERATOR_VERSION: u32 = 22;
pub const CAPTURE_SCHEMA_VERSION: u32 = 1;
pub const MAX_VISIBILITY_CAMERAS: usize = 16;
pub const PUBLICATION_MODES: [&str; 6] = [
    "color",
    "depth",
    "normal",
    "semantic",
    "position",
    "co_visibility",
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GeneratorIdentity {
    pub schema_version: u32,
    pub crate_version: String,
    pub generator_version: u32,
    /// SHA256 of the length-delimited, sorted build-input path/hash map.
    pub source_sha256: String,
}

/// The renderer and publisher use exactly the same camera color program.
pub fn camera_color(index: usize, count: usize) -> [u8; 3] {
    assert!((1..=MAX_VISIBILITY_CAMERAS).contains(&count) && index < count);
    let channel = index % 3;
    let bits = (count - channel).div_ceil(3);
    let scale = 255 / ((1 << bits) - 1);
    let mut rgb = [0; 3];
    rgb[channel] = (scale * (1 << (bits - 1 - index / 3))) as u8;
    rgb
}

pub fn mask_color(mask: u16, count: usize) -> [u8; 3] {
    assert!((1..=MAX_VISIBILITY_CAMERAS).contains(&count));
    let mut rgb = [0u8; 3];
    for i in 0..count {
        if mask & (1 << i) != 0 {
            let color = camera_color(i, count);
            for c in 0..3 {
                rgb[c] += color[c];
            }
        }
    }
    rgb
}

#[cfg(feature = "provenance")]
pub mod provenance {
    use sha2::{Digest, Sha256};
    use std::{collections::BTreeMap, fs, io, path::Path};

    pub fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    // Git checks text inputs out as LF. Hash their canonical text contents so
    // a pre-existing CRLF checkout has the same build identity. Binary assets
    // and the public sha() used for rendered artifacts remain byte-exact.
    fn input_sha(path: &Path, bytes: &[u8]) -> String {
        let text = matches!(
            path.extension().and_then(|value| value.to_str()),
            Some(
                "rs" | "wgsl"
                    | "glsl"
                    | "metal"
                    | "hlsl"
                    | "vert"
                    | "frag"
                    | "toml"
                    | "lock"
                    | "json"
                    | "html"
                    | "css"
                    | "tex"
                    | "sty"
                    | "bib"
                    | "md"
                    | "svg"
                    | "txt"
            )
        ) || path.file_name().is_some_and(|name| name == "LICENSE");
        if text {
            if let Ok(value) = std::str::from_utf8(bytes) {
                return sha(value.replace("\r\n", "\n").as_bytes());
            }
        }
        sha(bytes)
    }

    /// Explicit renderer inputs; documentation/tooling changes do not force GPU
    /// recapture. Registry dependencies are bound by Cargo.lock. Local WGPU
    /// patches, shaders, built-in assets and the wire contract are included.
    pub fn source_inputs(root: &Path) -> io::Result<BTreeMap<String, String>> {
        let mut files = BTreeMap::new();
        for name in [
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "crates/capture/Cargo.toml",
            "third_party/wgpu-core/Cargo.toml",
            "third_party/wgpu-hal/Cargo.toml",
        ] {
            let path = root.join(name);
            if path.is_file() {
                files.insert(name.into(), input_sha(&path, &fs::read(&path)?));
            }
        }
        for name in [
            "src",
            "crates/capture/src",
            "third_party/wgpu-core/src",
            "third_party/wgpu-hal/src",
        ] {
            collect(root, &root.join(name), &mut files)?;
        }
        // Avoid hashing downloaded models/textures that are deliberately not
        // part of the source release. This is a source/dependency identity, not
        // a claim of bit-identical GPU output or an external asset audit.
        for name in ["assets/shaders", "assets/embedded"] {
            collect(root, &root.join(name), &mut files)?;
        }
        Ok(files)
    }

    fn collect(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) -> io::Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_symlink() {
                return Err(io::Error::other(
                    "symlinks are not supported in generator source inputs",
                ));
            }
            if kind.is_dir() {
                collect(root, &entry.path(), out)?;
            }
            if kind.is_file() {
                let name = entry
                    .path()
                    .strip_prefix(root)
                    .map_err(io::Error::other)?
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(name, input_sha(&entry.path(), &fs::read(entry.path())?));
            }
        }
        Ok(())
    }

    pub fn source_digest(inputs: &BTreeMap<String, String>) -> String {
        let mut hash = Sha256::new();
        for (path, digest) in inputs {
            hash.update((path.len() as u64).to_le_bytes());
            hash.update(path.as_bytes());
            hash.update(digest.as_bytes());
        }
        format!("{hash:x}", hash = hash.finalize())
    }

    pub fn publisher_inputs(root: &Path) -> io::Result<BTreeMap<String, String>> {
        let mut inputs = BTreeMap::new();
        for name in ["Cargo.toml", "build.rs"] {
            let path = root.join(name);
            inputs.insert(name.into(), input_sha(&path, &fs::read(&path)?));
        }
        for name in ["src", "templates", "fonts"] {
            collect(root, &root.join(name), &mut inputs)?;
        }
        Ok(inputs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn text_build_inputs_have_identical_lf_and_crlf_identities() {
            for name in [
                "Cargo.toml",
                "Cargo.lock",
                "src/main.rs",
                "src/main.wgsl",
                "templates/main.html",
                "fonts/LICENSE",
            ] {
                let path = Path::new(name);
                assert_eq!(
                    input_sha(path, b"first\nsecond\n"),
                    input_sha(path, b"first\r\nsecond\r\n"),
                    "{name}"
                );
            }
        }

        #[test]
        fn binary_assets_and_artifact_hashes_remain_byte_exact() {
            let path = Path::new("assets/embedded/texture.png");
            let bytes = b"first\r\nsecond\r\n";
            assert_eq!(input_sha(path, bytes), sha(bytes));
            assert_ne!(input_sha(path, bytes), input_sha(path, b"first\nsecond\n"));
            assert_ne!(sha(bytes), sha(b"first\nsecond\n"));
            assert_eq!(
                input_sha(Path::new("src/invalid.rs"), b"\xff\r\n"),
                sha(b"\xff\r\n")
            );
        }
    }
}
