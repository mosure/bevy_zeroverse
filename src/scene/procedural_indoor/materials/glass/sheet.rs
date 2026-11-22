//! Shared cut stock. Gauge numbers are stable PBR binding slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(into = "f32")]
pub enum SheetGauge {
    Four,
    Five,
    Six,
    Eight,
}

impl SheetGauge {
    pub const fn metres(self) -> f32 {
        match self {
            Self::Four => 0.004,
            Self::Five => 0.005,
            Self::Six => 0.006,
            Self::Eight => 0.008,
        }
    }
    pub(crate) fn slot(self) -> usize {
        64 + self as usize
    }
}
impl From<SheetGauge> for f32 {
    fn from(value: SheetGauge) -> Self {
        value.metres()
    }
}
