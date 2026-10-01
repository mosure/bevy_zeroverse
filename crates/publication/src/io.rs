use anyhow::{ensure, Context, Result};
use bevy_zeroverse_capture::provenance::sha;
use serde::{de::DeserializeOwned, Serialize};
use std::{collections::BTreeMap, fs, io::Write, path::Path};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| path.display().to_string())?)
        .with_context(|| format!("decode {}", path.display()))
}
pub fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut data = serde_json::to_vec_pretty(value)?;
    data.push(b'\n');
    fs::write(path, data)?;
    Ok(())
}
pub fn hash(path: &Path) -> Result<String> {
    Ok(sha(&fs::read(path)?))
}
pub fn files(root: &Path, dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    if !dir.exists() {
        return Ok(result);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        ensure!(
            !entry.file_type()?.is_symlink(),
            "symlink in publication assets"
        );
        if entry.file_type()?.is_dir() {
            result.extend(files(root, &entry.path())?);
        } else {
            result.insert(
                entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                hash(&entry.path())?,
            );
        }
    }
    Ok(result)
}
pub fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        ensure!(
            !entry.file_type()?.is_symlink(),
            "symlink in publication source"
        );
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &to.join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), to.join(entry.file_name()))?;
        }
    }
    Ok(())
}
/// Stable order, fixed timestamps and safe names make archives reproducible.
pub fn archive(path: &Path, entries: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    let mut archive = ZipWriter::new(fs::File::create(path)?);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .last_modified_time(zip::DateTime::default());
    for (name, bytes) in entries {
        super::config::safe_relative(Path::new(name))?;
        archive.start_file(name, options)?;
        archive.write_all(bytes)?;
    }
    archive.finish()?;
    Ok(())
}

pub fn html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
