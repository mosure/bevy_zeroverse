//! Slot-stable mounting misalignment, independent of placement, lens and roll.
use super::*;
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;

/// A circularly symmetric small-angle error: the radius is Rayleigh distributed,
/// truncated at three scale lengths, then mapped into the requested interval.
/// This is a designed mounting prior, not a fit to a measured camera population.
/// Sampling happens once per slot; retries cannot prefer an easier zero tilt.
pub(super) fn sample(policy: &CeilingCameraSettings, seed: u64, index: usize) -> Quat {
    let [lo, hi] = policy.tilt_degrees;
    if hi == 0. {
        return Quat::IDENTITY;
    }
    let mut rng = stream(seed, 0x4345_494c_5449_4c54_u64.wrapping_add(index as u64));
    let u: f64 = rng.random();
    let radius = (-2. * (-u * (1. - (-4.5_f64).exp())).ln_1p()).sqrt() / 3.;
    let angle = (lo + (hi - lo) * radius as f32).to_radians();
    let azimuth = rng.random_range(0.0..std::f32::consts::TAU);
    Quat::from_axis_angle(Vec3::new(azimuth.cos(), azimuth.sin(), 0.), angle)
}

#[cfg(test)]
mod qualification;
#[cfg(test)]
mod tests;
