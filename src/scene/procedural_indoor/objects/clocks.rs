//! Wall-clock movement, local time and housing. The print is registered to the recessed dial.
mod dial;
mod legacy;
use super::*;
pub use dial::{DialStyle, PrintedDial};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClockProgram {
    pub seconds: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub printed: Option<PrintedDial>,
    pub digital: bool,
    pub square: bool,
    pub dark_dial: bool,
    pub second_hand: bool,
    pub bezel_fraction: f32,
    pub casing: Surface,
    pub roundness: f32,
    pub case_taper: f32,
    pub minor_ticks: bool,
    pub tick_radius: f32,
    pub hand_lengths: [f32; 3],
    pub hand_widths: [f32; 3],
    pub digit_width_fraction: f32,
    pub digit_height_fraction: f32,
    pub digit_stroke_fraction: f32,
}
pub fn parameters(o: &IndoorObject) -> ClockProgram {
    let mut rng = stream(o.seed, 871);
    let mut detail = stream(o.seed, 0x434c4f434b444554);
    let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 0x434c4f434b46494e);
    let digital = rng.random_bool(if o.size.x / o.size.y > 1.45 {
        0.90
    } else {
        0.30
    });
    let square = rng.random_bool(0.4) || digital;
    // Preserve the draw order (including the old time) on archived and current stock.
    let old_time = rng.random_range(0..86400);
    let installed = o.variant & INSTALLED != 0;
    ClockProgram {
        seconds: if installed {
            ((o.variant >> 3) & 0x1ffff) % 86400
        } else {
            old_time
        },
        printed: installed.then(|| PrintedDial::sample(o, digital)),
        digital,
        square,
        dark_dial: finish.random_bool(0.35),
        second_hand: rng.random_bool(0.65),
        bezel_fraction: rng.random_range(0.035..0.13),
        casing: [
            Surface::Metal,
            Surface::WoodEdge,
            Surface::Chrome,
            Surface::Plastic,
        ][finish.random_range(0..4)],
        roundness: if square {
            detail.random_range(3.5..12.)
        } else {
            2.
        },
        case_taper: detail.random_range(0.82..0.995),
        minor_ticks: detail.random_bool(0.72),
        tick_radius: detail.random_range(0.78..0.87),
        hand_lengths: [
            detail.random_range(0.40..0.57),
            detail.random_range(0.65..0.77),
            detail.random_range(0.70..0.82),
        ],
        hand_widths: [
            detail.random_range(0.014..0.032),
            detail.random_range(0.009..0.020),
            detail.random_range(0.003..0.008),
        ],
        digit_width_fraction: detail.random_range(0.135..0.18),
        digit_height_fraction: detail.random_range(0.46..0.72),
        digit_stroke_fraction: detail.random_range(0.055..0.12),
    }
}

// Three low bits retain the legacy case variation. Store seconds in bits 3..20
// so a standalone archived object replays without access to the room manifest.
const INSTALLED: u32 = 1 << 24;

pub(crate) fn install(
    o: &mut IndoorObject,
    room_seed: u64,
    mood: super::super::layout::LightingMood,
) {
    use super::super::layout::LightingMood;
    let mut local = stream(room_seed, u64::from_le_bytes(*b"WallTime"));
    let hours = match mood {
        LightingMood::Daylight | LightingMood::Overcast => 6..20,
        LightingMood::Evening => 17..26,
    };
    let reference: i32 = local.random_range(hours.start * 3600..hours.end * 3600);
    let mut movement = stream(o.seed, u64::from_le_bytes(*b"Quartz!!"));
    let offset = if movement.random_bool(0.32) {
        movement.random_range(-8..=8)
    } else {
        0
    };
    let seconds = (reference + offset).rem_euclid(86400) as u32;
    o.variant = INSTALLED | (seconds << 3) | (o.variant & 7);
}

/// Metric dimensions are sampled before mounting/clearance admission.
pub(crate) fn dimensions(r: &mut ChaCha8Rng) -> Vec3 {
    let height = r.random_range(0.20..0.46);
    let aspect = if r.random_bool(0.55) {
        1.
    } else {
        r.random_range(0.80..2.35)
    };
    Vec3::new(height * aspect, height, r.random_range(0.032..0.080))
}
pub fn hand_angles(seconds: u32) -> [f32; 3] {
    let t = seconds as f32;
    [
        TAU * (t % 43200.) / 43200.,
        TAU * (t % 3600.) / 3600.,
        TAU * (t % 60.) / 60.,
    ]
}
/// Local back and bezel planes. Keep the admitted wall mount and use the
/// reserved forward volume for the movement housing on a recessed dial.
pub(crate) fn case_planes(o: &IndoorObject) -> (f32, f32) {
    let mut rng = stream(o.seed, 0x434c4f434b444550);
    let back_offset = (o.size.z - 0.026).max(o.size.z * 0.12) * rng.random_range(0.72..1.0) * 0.5;
    let front = if o.variant & INSTALLED != 0 {
        // Allow both low-profile and deep housings at the same footprint. A
        // 16 mm movement fits whenever the admitted depth can accommodate it.
        let limit = o.size.z * 0.5;
        let minimum = (0.016 - back_offset).max(back_offset).min(limit);
        minimum + (limit - minimum) * rng.random_range(0.12..1.0)
    } else {
        back_offset
    };
    (-back_offset, front)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    match p.printed {
        Some(dial) => dial.build(a, o, &p),
        None => legacy::build(a, o, &p),
    }
}

#[cfg(test)]
mod tests;
