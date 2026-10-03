//! Periodic substrate fields shared by mineral and coating programs.
use super::{hash, periodic_noise};

pub(super) fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
pub(super) struct Cell {
    pub radius: f32,
    pub edge: f32,
    pub dye: f32,
}
/// Jittered, wrapped cell centres. Colour and relief use the same cell identity.
pub(super) fn cellular([u, v]: [f32; 2], count: [u32; 2], seed: u64) -> Cell {
    let x = u.rem_euclid(1.) * count[0] as f32;
    let y = v.rem_euclid(1.) * count[1] as f32;
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let mut nearest = [f32::INFINITY; 2];
    let mut dye = 0.;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let hx = cx.rem_euclid(count[0] as i32) as u32;
            let hy = cy.rem_euclid(count[1] as i32) as u32;
            let px = cx as f32 + 0.05 + 0.90 * hash(hx, hy, seed);
            let py = cy as f32 + 0.05 + 0.90 * hash(hx, hy, seed.wrapping_add(31));
            let d = (px - x).powi(2) + (py - y).powi(2);
            if d < nearest[0] {
                nearest = [d, nearest[0]];
                dye = hash(hx, hy, seed.wrapping_add(71));
            } else if d < nearest[1] {
                nearest[1] = d;
            }
        }
    }
    Cell {
        radius: nearest[0].sqrt(),
        edge: nearest[1].sqrt() - nearest[0].sqrt(),
        dye,
    }
}
pub(super) fn deposit(uv: [f32; 2], count: [u32; 2], warp: f32, seed: u64) -> f32 {
    let [u, v] = uv;
    let x = u + (periodic_noise(u, v, 3, 5, seed) - 0.5) * warp;
    let y = v + (periodic_noise(u, v, 5, 3, seed.wrapping_add(17)) - 0.5) * warp;
    0.64 * periodic_noise(x, y, count[0], count[1], seed.wrapping_add(47))
        + 0.25 * periodic_noise(x, y, count[0] * 2, count[1] * 2, seed.wrapping_add(61))
        + 0.11 * periodic_noise(x, y, count[0] * 4, count[1] * 4, seed.wrapping_add(83))
}
pub(super) fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    // Pigmentation mixtures are interpolated in linear reflectance, not sRGB.
    [0, 1, 2].map(|i| {
        super::linear_to_srgb(
            super::srgb_to_linear(a[i]) * (1. - t) + super::srgb_to_linear(b[i]) * t,
        )
    })
}
pub(super) fn tint(c: [f32; 3], gain: f32) -> [f32; 3] {
    c.map(|c| super::linear_to_srgb((super::srgb_to_linear(c) * gain).clamp(0., 1.)))
}
