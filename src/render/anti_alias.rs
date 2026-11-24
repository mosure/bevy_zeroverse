//! Interactive RGB anti-aliasing. Calibrated capture uses its existing RGB
//! treatment and the geometry-aware resolve in `ssaa`.
use bevy::{
    anti_alias::fxaa::{Fxaa, Sensitivity},
    prelude::*,
};
use bevy_args::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Reflect, ValueEnum, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum PreviewAa {
    /// Temporal on full pinhole editor/kiosk views, spatial on inset or fisheye views.
    /// Ordinary multi-camera grids retain the scene's normal RGB treatment.
    #[default]
    Auto,
    Off,
    Fxaa,
    Smaa,
    SmaaUltra,
    Taa,
    #[serde(rename = "msaa-2")]
    #[value(name = "msaa-2")]
    Msaa2,
    #[serde(rename = "msaa-4")]
    #[value(name = "msaa-4")]
    Msaa4,
    #[serde(rename = "msaa-8")]
    #[value(name = "msaa-8")]
    Msaa8,
    /// Temporal accumulation followed by a spatial edge pass on full pinhole views.
    TaaFxaa,
}

/// Integrate the warped RGB pixel footprint on interactive fisheye views.
/// Its absence keeps the calibrated renderer on the original center ray.
#[derive(Component)]
pub(crate) struct LensPreviewAa;

pub(crate) fn indoor_fxaa() -> Fxaa {
    Fxaa {
        enabled: true,
        edge_threshold: Sensitivity::High,
        edge_threshold_min: Sensitivity::High,
    }
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ConfigurePreviewAa;

#[cfg(feature = "viewer")]
mod runtime;
#[cfg(feature = "viewer")]
pub(crate) use runtime::{install, ManagedPreviewAa};
#[cfg(not(feature = "viewer"))]
#[derive(Component)]
pub(crate) struct ManagedPreviewAa;
