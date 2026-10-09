//! Explicit GPU capture sink. CPU exports never receive an empty image sample.
use super::*;

#[derive(Resource, Default)]
pub struct GpuSampleSink {
    pub metadata: Option<Sample>,
    /// View indices use the same timestep-major ordering as Sample::views.
    pub images: Vec<(usize, crate::io::image_copy::GpuCapturedImages)>,
}
