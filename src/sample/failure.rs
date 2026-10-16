//! Typed scene rejections are distinct from renderer/integrity failures.
//! Only construction decisions with valid inputs may return a rejection.
use bevy::prelude::Resource;
use serde::Serialize;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SceneRejectionKind {
    MissingProgramAnchor,
    InsufficientProgramCapacity,
    CameraPlacement,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SceneRejection {
    pub seed: u64,
    pub kind: SceneRejectionKind,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum CaptureError {
    Rejected {
        rejection: SceneRejection,
        /// True only after the headless runner has retired the failed scene.
        retired: bool,
    },
    Failed(String),
}

impl CaptureError {
    pub(crate) fn rejected(seed: u64, kind: SceneRejectionKind, reason: String) -> Self {
        Self::Rejected {
            rejection: SceneRejection { seed, kind, reason },
            retired: false,
        }
    }
}
impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected { rejection, .. } => f.write_str(&rejection.reason),
            Self::Failed(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for CaptureError {}
impl From<String> for CaptureError {
    fn from(value: String) -> Self {
        Self::Failed(value)
    }
}
impl From<&str> for CaptureError {
    fn from(value: &str) -> Self {
        Self::Failed(value.into())
    }
}

/// Kept separately from the existing string diagnostic for viewer/API compatibility.
#[derive(Resource, Default)]
pub(crate) struct SceneCaptureError(pub Option<CaptureError>);

impl SceneCaptureError {
    pub(crate) fn take_for_diagnostic(&mut self, diagnostic: String) -> CaptureError {
        // A later renderer/readback failure can overwrite the legacy diagnostic.
        // Retain a typed source only when it still owns that diagnostic. Never
        // infer a rejection category from an arbitrary error's wording.
        self.0
            .take()
            .filter(|source| match source {
                CaptureError::Rejected { rejection, .. } => rejection.reason == diagnostic,
                CaptureError::Failed(message) => *message == diagnostic,
            })
            .unwrap_or(CaptureError::Failed(diagnostic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_failures_override_an_earlier_scene_rejection() {
        let rejection = || {
            CaptureError::rejected(
                17,
                SceneRejectionKind::CameraPlacement,
                "bounded search".into(),
            )
        };
        let mut source = SceneCaptureError(Some(rejection()));
        assert_eq!(
            source.take_for_diagnostic("bounded search".into()),
            rejection()
        );
        assert!(source.0.is_none());
        let mut source = SceneCaptureError(Some(rejection()));
        assert_eq!(
            source.take_for_diagnostic("GPU device lost".into()),
            CaptureError::Failed("GPU device lost".into())
        );
        assert_eq!(
            source.take_for_diagnostic("bounded search".into()),
            CaptureError::Failed("bounded search".into())
        );
    }
}
