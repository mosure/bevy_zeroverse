//! The geometry and placement paths consume one shared sequence of chair dimensions.
use super::*;
use rand_chacha::ChaCha8Rng;

pub(super) struct Fit {
    pub(super) initial_back: u8,
    pub(super) armrests: bool,
    pub(super) curvature: f32,
    pub(super) taper: f32,
    pub(super) shell_thickness: f32,
    pub(super) recline: f32,
    pub(super) arm_height: f32,
    pub(super) spindle_pitch: f32,
    pub(super) headrest: bool,
    pub(super) lumbar: f32,
    pub(super) shoulder_flare: f32,
    pub(super) seat_roundness: f32,
    pub(super) seat_width_fraction: f32,
    pub(super) seat_depth_fraction: f32,
    pub(super) back_width_fraction: f32,
    pub(super) arm_pad_length: f32,
}

impl Fit {
    pub(super) fn sample(o: &IndoorObject) -> (Self, ChaCha8Rng) {
        let mut rng = stream(o.seed, 73);
        let fit = Self {
            initial_back: rng.random_range(0..4),
            armrests: rng.random_bool(0.62),
            curvature: rng.random_range(0.018..0.095),
            taper: rng.random_range(-0.06..0.20),
            shell_thickness: rng.random_range(0.018..0.045),
            recline: rng.random_range(-0.04..0.10),
            arm_height: rng.random_range(0.61..0.70),
            spindle_pitch: rng.random_range(0.045..0.09),
            headrest: o.size.y > 1.12
                && matches!(o.variant % FAMILIES, 0 | 1 | 5 | 9)
                && rng.random_bool(0.62),
            lumbar: rng.random_range(0.012..0.050),
            shoulder_flare: rng.random_range(-0.12..0.10),
            seat_roundness: rng.random_range(2.5..7.0),
            seat_width_fraction: rng.random_range(0.63..0.80),
            seat_depth_fraction: rng.random_range(0.54..0.79),
            back_width_fraction: rng.random_range(0.68..1.08),
            arm_pad_length: rng.random_range(0.16..0.33),
        };
        (fit, rng)
    }
}

/// Camera placement needs the headrest and arm heights only. Fittings,
/// upholstery and frame construction use independent streams.
pub(crate) fn view_supports(o: &IndoorObject) -> (bool, Option<f32>) {
    let (fit, _) = Fit::sample(o);
    (fit.headrest, fit.armrests.then_some(fit.arm_height))
}
