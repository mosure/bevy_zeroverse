//! Camera-model contract and optional nonlinear rendering. A logical camera has
//! exactly one calibration, timeline and capture index, regardless of its render
//! views. The default pinhole path allocates no nonlinear resources.
mod activity;
pub(crate) mod clusters;
#[cfg(feature = "viewer")]
mod preview;
mod raster;
mod resolve;
mod runtime;
mod sampling;
mod settings;
pub use bevy_zeroverse_capture::calibration::{
    CameraCalibration, Fisheye, FisheyeFit, FisheyeSensor,
};
pub(crate) use resolve::LensResolveLabel;
pub use runtime::{FisheyeCameraPlugin, LensCamera, LensFace};
pub use sampling::LensSampling;

pub fn calibration(
    projection: &bevy::prelude::Projection,
    fisheye: Option<FisheyeSensor>,
    offset: bevy::prelude::Vec2,
    size: [u32; 2],
) -> Result<CameraCalibration, String> {
    match fisheye {
        Some(lens) => lens.calibration(size, offset.to_array()),
        None => super::pinhole::calibration(projection, size[0], size[1]),
    }
}

/// Bevy 0.20 passes WGSL directly to Naga. Compose shared definitions explicitly
/// rather than relying on the removed legacy `#import` preprocessor.
pub(crate) fn shader(source: &str, extra: &str) -> String {
    [include_str!("lens.wgsl"), extra, source]
        .join("\n")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}
