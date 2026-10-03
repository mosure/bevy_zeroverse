//! Substrate-specific, periodic structure. Heights are metres, not painted shadows.
use super::{periodic_noise, program::MaterialRecipe, Surface};

pub(super) fn value(r: &MaterialRecipe, uv: [f32; 2], noise: [f32; 3], floor: u32) -> f32 {
    let [u, v] = uv;
    let [macro_n, meso, micro] = noise;
    let sample = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
    match r.surface {
        Surface::Wood | Surface::WoodEdge | Surface::Bark => super::timber::grain(r, u, v),
        Surface::Floor if floor == 0 => super::timber::grain(r, u, v),
        Surface::Chrome => sample(103, 5, 44) * 0.78 + micro * 0.22,
        Surface::Metal => micro * 0.80 + meso * 0.20,
        Surface::Plastic | Surface::Rubber | Surface::Paper => micro * 0.72 + meso * 0.28,
        Surface::Paint | Surface::Accent | Surface::Ceiling | Surface::Ceramic => {
            // Roller stipple / glaze orange peel. No marble veins in wall paint
            // or random colour speckles masquerading as ceramic highlights.
            0.70 * micro + 0.26 * meso + 0.04 * macro_n
        }
        _ => {
            // Aggregate spanning three scales, without the old sine-wave ridges.
            (0.55 * macro_n + 0.30 * meso + 0.15 * micro) * (1. - r.mineral_mix)
                + r.mineral_mix * (0.70 * meso + 0.30 * micro)
        }
    }
}
