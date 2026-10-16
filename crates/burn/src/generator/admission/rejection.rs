use bevy_zeroverse::sample::failure::SceneRejectionKind;
use serde::Serialize;
use std::path::PathBuf;

/// Opt-in recovery never changes whether a scene passes the acceptance checks.
/// Limits count attempts in this run, not archives; zero/unbounded limits are invalid.
#[derive(Clone, Debug, Default)]
pub enum RejectionPolicy {
    #[default]
    Abort,
    Continue {
        kinds: Vec<SceneRejectionKind>,
        max_total: u64,
        max_consecutive: u64,
    },
}
impl RejectionPolicy {
    pub(super) fn validate(&self) -> anyhow::Result<()> {
        if let Self::Continue {
            kinds,
            max_total,
            max_consecutive,
        } = self
        {
            anyhow::ensure!(
                !kinds.is_empty()
                    && *max_total > 0
                    && *max_consecutive > 0
                    && max_consecutive <= max_total,
                "rejection recovery needs explicit kinds and positive total/consecutive bounds"
            );
            anyhow::ensure!(
                kinds
                    .iter()
                    .enumerate()
                    .all(|(i, k)| !kinds[..i].contains(k)),
                "duplicate rejection categories"
            );
        }
        Ok(())
    }
    pub(super) fn permits(&self, kind: SceneRejectionKind, total: u64, consecutive: u64) -> bool {
        matches!(self, Self::Continue { kinds, max_total, max_consecutive }
            if kinds.contains(&kind) && total <= *max_total && consecutive <= *max_consecutive)
    }
    pub(super) fn enabled(&self) -> bool {
        !matches!(self, Self::Abort)
    }
}

/// Present only after the atomic archive/directory writer has closed successfully.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PublicationReceipt {
    pub index: usize,
    pub path: PathBuf,
}
