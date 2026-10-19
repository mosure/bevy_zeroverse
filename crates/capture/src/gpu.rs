//! GPU capture semantics. OS handles and device implementations live elsewhere.
use serde::{Deserialize, Serialize};
pub mod wire;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Plane {
    Color,
    /// World XYZ, metric camera-Z in alpha (zero is invalid).
    WorldDepth,
    /// View normal encoded in [0,1], integral semantic ID in alpha.
    NormalSemantic,
    /// Forward pixel dx/dy, source validity, target visibility.
    OpticalFlow,
    /// Camera membership bits, peer count, source validity, zero.
    CoVisibility,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorOutput {
    /// Full precision tone-mapped linear RGBA, no transfer function applied.
    #[default]
    TonemappedLinearRgbaF32,
    /// IEC sRGB transfer applied once to RGB; alpha is dropped. HWC, float32.
    SrgbRgbF32,
    /// IEC sRGB transfer applied once to RGB; alpha is preserved linearly.
    SrgbRgbaF32,
}
impl ColorOutput {
    pub fn channels(self) -> usize {
        if self == Self::SrgbRgbF32 {
            3
        } else {
            4
        }
    }
    pub fn is_srgb(self) -> bool {
        self != Self::TonemappedLinearRgbaF32
    }
}
