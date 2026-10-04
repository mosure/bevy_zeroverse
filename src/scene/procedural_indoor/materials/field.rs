//! Periodic substrate fields shared by mineral and coating programs.
use super::{hash, periodic_noise};

pub(super) fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
/// Integer, periodic feature counts derived from metres, not atlas resolution.
pub(super) fn frequency(period_m: f32, spacing_m: f32) -> u32 {
    (period_m / spacing_m).round().clamp(1., 192.) as u32
}

/// Approximate footprint integration for sparse round inclusions/air voids.
/// Features smaller than a texel preserve area instead of flashing at full contrast.
pub(super) fn disk(distance: f32, radius: f32, footprint: f32) -> f32 {
    let aa = footprint.max(0.002);
    let filtered = radius.max(aa);
    // The smooth radial edge integrates to pi * (filtered² + aa²/5).
    // Normalize it so even sub-texel inclusions keep their physical area.
    smooth((filtered + aa - distance) / (2. * aa)) * radius.powi(2)
        / (filtered.powi(2) + aa.powi(2) * 0.20)
}

pub(super) fn band(distance: f32, width: f32, footprint: f32) -> f32 {
    let aa = footprint.max(0.0001);
    let filtered = width.max(aa);
    smooth((filtered + aa - distance) / (2. * aa)) * width / filtered
}
pub(super) struct Cell {
    pub radius: f32,
    pub edge: f32,
    pub dye: f32,
    pub offset: [f32; 2],
    pub shape: f32,
}
/// Jittered, wrapped cell centres. Colour and relief use the same cell identity.
pub(super) fn cellular([u, v]: [f32; 2], count: [u32; 2], seed: u64) -> Cell {
    let x = u.rem_euclid(1.) * count[0] as f32;
    let y = v.rem_euclid(1.) * count[1] as f32;
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let mut nearest = [f32::INFINITY; 2];
    let mut dye = 0.;
    let mut offset = [0.; 2];
    let mut shape = 0.;
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
                offset = [px - x, py - y];
                shape = hash(hx, hy, seed.wrapping_add(113));
            } else if d < nearest[1] {
                nearest[1] = d;
            }
        }
    }
    Cell {
        radius: nearest[0].sqrt(),
        edge: nearest[1].sqrt() - nearest[0].sqrt(),
        dye,
        offset,
        shape,
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

#[cfg(test)]
mod tests {
    #[test]
    fn filtered_inclusions_preserve_area_across_pixel_footprints() {
        let n = 256;
        let pitch = 2. / n as f32;
        for radius in [0.04_f32, 0.18, 0.32] {
            for footprint in [0.005, 0.05, 0.25] {
                let mut coverage = 0.;
                for y in 0..n {
                    for x in 0..n {
                        let u = -1. + (x as f32 + 0.5) * pitch;
                        let v = -1. + (y as f32 + 0.5) * pitch;
                        coverage += super::disk((u * u + v * v).sqrt(), radius, footprint);
                    }
                }
                let area = coverage * pitch * pitch;
                let expected = std::f32::consts::PI * radius * radius;
                assert!((area / expected - 1.).abs() < 0.03,
                    "inclusion coverage changed with filtering: r={radius} footprint={footprint} area={area}");
            }
        }
    }
}
