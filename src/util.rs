use std::path::{Path, PathBuf};

pub fn strip_extended_length_prefix(path: &Path) -> PathBuf {
    if cfg!(windows) {
        let prefix = r"\\?\";
        if let Some(path_str) = path.to_str() {
            if let Some(stripped) = path_str.strip_prefix(prefix) {
                return PathBuf::from(stripped);
            }
        }
    }
    path.to_path_buf()
}
