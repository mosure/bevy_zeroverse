//! Roller blind hardware, in metres. The housing and controls are ordered as
//! independent options within a bank of matching window treatments.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollerHead {
    Wrapped,
    SquareCassette,
    RoundCassette,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RollerDrive {
    LeftLoop,
    RightLoop,
    Cordless,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RollerStock {
    pub head: RollerHead,
    pub drive: RollerDrive,
    pub radius_m: f32,
    pub wheel_radius_m: f32,
    pub loop_fraction: f32,
    pub max_drop_m: f32,
    pub hem_height_m: f32,
    pub hem_depth_m: f32,
    pub metal_hem: bool,
}

impl RollerStock {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"RollGear"));
        Self {
            head: [
                RollerHead::Wrapped,
                RollerHead::SquareCassette,
                RollerHead::RoundCassette,
            ][rng.random_range(0..3)],
            drive: [
                RollerDrive::LeftLoop,
                RollerDrive::RightLoop,
                RollerDrive::Cordless,
            ][rng.random_range(0..3)],
            radius_m: rng.random_range(0.031..0.046),
            wheel_radius_m: rng.random_range(0.010..0.016),
            loop_fraction: rng.random_range(0.42..0.77),
            max_drop_m: rng.random_range(0.65..1.3),
            hem_height_m: rng.random_range(0.010..0.023),
            hem_depth_m: rng.random_range(0.013..0.021),
            metal_hem: rng.random_bool(0.58),
        }
    }

    pub(crate) fn loop_drop(self, available: f32) -> Option<f32> {
        (self.drive != RollerDrive::Cordless)
            .then(|| (available * self.loop_fraction).min(self.max_drop_m))
    }

    pub(super) fn valid(self) -> bool {
        (0.031..=0.046).contains(&self.radius_m)
            && (0.010..=0.016).contains(&self.wheel_radius_m)
            && (0.42..=0.77).contains(&self.loop_fraction)
            && (0.65..=1.3).contains(&self.max_drop_m)
            && (0.010..=0.023).contains(&self.hem_height_m)
            && (0.013..=0.021).contains(&self.hem_depth_m)
    }
}
