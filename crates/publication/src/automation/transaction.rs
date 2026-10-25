//! Roll back prepared evidence if any later render, typeset or validation fails.
//! Backups are durable until commit; an interrupted run requires explicit
//! recovery before another writer starts (never bless a partial installation).
use crate::io;
use anyhow::{ensure, Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub struct Inputs {
    root: PathBuf,
    backup: PathBuf,
    installed: Vec<(PathBuf, bool)>,
}
impl Inputs {
    pub fn begin(root: &Path) -> Result<Self> {
        let backup = root.join("out/publication/input-transaction");
        ensure!(!backup.exists(), "interrupted publication input transaction: inspect {} and restore its journal before retrying", backup.display());
        fs::create_dir_all(&backup)?;
        Ok(Self {
            root: root.to_owned(),
            backup,
            installed: Vec::new(),
        })
    }
    pub fn replace(&mut self, source: &Path, destination: &Path) -> Result<()> {
        crate::config::safe_relative(destination)?;
        let target = self.root.join(destination);
        let backup = self.backup.join(destination);
        let existed = target.exists();
        self.installed.push((destination.to_owned(), existed));
        // Record intent before mutation, so a killed process leaves recoverable paths.
        io::write(&self.backup.join("journal.json"), &self.installed)?;
        if existed {
            fs::create_dir_all(backup.parent().context("backup parent")?)?;
            fs::rename(&target, &backup)?;
        }
        fs::create_dir_all(target.parent().context("destination parent")?)?;
        if source.is_dir() {
            io::copy_tree(source, &target)?;
        } else {
            fs::copy(source, &target)?;
        }
        Ok(())
    }
    pub fn commit(mut self) -> Result<()> {
        self.installed.clear();
        fs::remove_dir_all(&self.backup)?;
        Ok(())
    }
}
impl Drop for Inputs {
    fn drop(&mut self) {
        let mut failed = false;
        for (name, existed) in self.installed.iter().rev() {
            let target = self.root.join(name);
            let backup = self.backup.join(name);
            // A failed rename may leave the original in place. Do not delete it.
            if *existed && !backup.exists() {
                continue;
            }
            if target.exists() {
                let result = if target.is_dir() {
                    fs::remove_dir_all(&target)
                } else {
                    fs::remove_file(&target)
                };
                if result.is_err() {
                    failed = true;
                    continue;
                }
            }
            if *existed && fs::rename(backup, target).is_err() {
                failed = true;
            }
        }
        if !failed {
            let _ = fs::remove_dir_all(&self.backup);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_restores_original_inputs_and_removes_new_inputs() {
        let root = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        fs::write(source.path().join("new"), "new").unwrap();
        fs::create_dir_all(root.path().join("docs/evidence")).unwrap();
        fs::write(root.path().join("docs/evidence/old"), "old").unwrap();
        {
            let mut tx = Inputs::begin(root.path()).unwrap();
            tx.replace(&source.path().join("new"), Path::new("docs/evidence/old"))
                .unwrap();
            tx.replace(source.path(), Path::new("docs/evidence/new"))
                .unwrap();
        }
        assert_eq!(
            fs::read_to_string(root.path().join("docs/evidence/old")).unwrap(),
            "old"
        );
        assert!(!root.path().join("docs/evidence/new").exists());
        let mut tx = Inputs::begin(root.path()).unwrap();
        tx.replace(&source.path().join("new"), Path::new("docs/evidence/old"))
            .unwrap();
        tx.commit().unwrap();
        assert_eq!(
            fs::read_to_string(root.path().join("docs/evidence/old")).unwrap(),
            "new"
        );
    }
}
