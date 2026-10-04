//! Continuously shaped, hollow planters with fitted soil and separate saucers.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PotProfile {
    pub belly: f32,
    pub neck: f32,
    pub wall_fraction: f32,
    pub flute_depth: f32,
    pub flute_count: u32,
    pub lip_height_fraction: f32,
}
impl PotProfile {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 6220);
        Self {
            belly: rng.random_range(0.82..1.035),
            neck: rng.random_range(0.82..1.02),
            wall_fraction: rng.random_range(0.055..0.095),
            flute_depth: rng.random_range(0.0..0.055),
            flute_count: rng.random_range(4..=8),
            lip_height_fraction: rng.random_range(0.025..0.065),
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    material: Surface,
    r: f32,
    h: f32,
    taper: f32,
    p: &PotProfile,
) -> (f32, f32) {
    let foot = h * 0.035;
    let wall = p.wall_fraction;
    let lip = p.lip_height_fraction;
    let profile = [
        (0., foot),
        (r * taper, foot),
        (r * (taper + 0.02), h * 0.15),
        (r * (taper + p.belly) * 0.5, h * 0.42),
        (r * p.belly, h * 0.65),
        (r * p.neck, h * (1. - lip)),
        (r * (p.neck + 0.025), h * (1. - lip * 0.7)),
        (r * (p.neck + 0.025), h),
        (r * (p.neck - wall), h),
        (r * (p.neck - wall), h * (1. - lip)),
        (r * (p.belly - wall), h * 0.65),
        (r * (taper - wall), h * 0.14),
        (0., h * 0.14),
    ];
    a.part(material, "other_prop").fluted_lathe(
        &profile,
        40,
        p.flute_depth,
        p.flute_count,
        Transform::IDENTITY,
    );
    a.part(material, "other_prop").lathe(
        &[
            (0., 0.),
            (r * 1.10, 0.),
            (r * 1.10, foot),
            (r * 1.025, foot * 1.6),
            (r * 0.96, foot),
            (0., foot),
        ],
        32,
        Transform::IDENTITY,
    );
    let soil = h * (1. - lip - 0.065);
    // Fit the inscribed disc at its actual height, including inward flutes.
    let t = ((soil / h - 0.65) / (1. - lip - 0.65)).clamp(0., 1.);
    let radius =
        r * ((p.belly - wall) * (1. - t) + (p.neck - wall) * t) * (1. - p.flute_depth) - r * 0.008;
    a.part(Surface::Soil, "other_prop").cylinder(
        radius,
        h * 0.025,
        Transform::from_xyz(0., soil - h * 0.0125, 0.),
    );
    (soil, radius)
}
