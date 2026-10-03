//! Uneven growth contours, cathedral cuts, knots and elongated open pores.
use super::{hash, periodic_noise, program::MaterialRecipe, Surface};
use std::f32::consts::TAU;

pub(super) fn grain(r: &MaterialRecipe, mut u: f32, mut v: f32) -> f32 {
    let mut seed = r.seed;
    if r.surface == Surface::Floor {
        let [nx, ny] = r.floor_repetitions(0);
        let x = (u * nx).floor() as u32;
        let stagger = (x % 2) as f32 * 0.5;
        let y = ((v * ny + stagger).floor() as u32) % ny as u32;
        // Different cuts of one species share a spectrum but never continue
        // the very same growth ring across a joint into the next plank.
        u = (u * nx).fract() + hash(x, y, seed.wrapping_add(79));
        v = (v * ny + stagger).fract() + hash(x, y, seed.wrapping_add(83));
        seed = seed
            .wrapping_add(u64::from(x) * 0x9e3779b9)
            .wrapping_add(u64::from(y) * 0x85ebca6b);
    }
    let noise = |u, v, x, y, salt| periodic_noise(u, v, x, y, seed.wrapping_add(salt));
    let rings = r.grain_frequency * 0.30;
    // Intersect cylindrical growth with a continuously offset/tapered log cut.
    // Near the pith this exposes nested cathedral contours; deeper/rift cuts
    // expose elongated rings. Periodic coordinates make the volume tileable.
    let cut = hash(7, 19, seed);
    let bow = (TAU * v).cos() + 0.23 * (TAU * (2. * v + cut)).cos();
    let growth = noise(u, v, 3, 2, 13) - 0.5;
    let x = 0.55 * (TAU * u).sin() + cut * 0.85 + growth * r.warp;
    let depth = 0.08 + cut * cut * 1.4 + (1. - cut) * (0.32 + 0.22 * bow);
    let phase = (x * x + depth * depth).sqrt() * rings + growth * (0.30 + r.warp * 6.);
    let ring_id = phase.floor().max(0.) as u32;
    let width = 0.07 + 0.14 * hash(ring_id, 5, seed);
    // Earlywood gradually darkens into latewood, then starts a new growth year.
    // Ring-specific width/pigmentation avoids uniform barcode-like intervals.
    let annual = phase.rem_euclid(1.);
    let late = smooth((annual - (1. - width)) / width);
    let pigmentation = 0.7 + 0.3 * hash(ring_id, 31, seed);
    let fibre_u = u + bow * (1. - cut) * 0.04 + growth * 0.06;
    let fibres = noise(fibre_u, v, 79, 7, 47);
    let pores = smooth((noise(fibre_u, v, 109, 13, 91) - 0.62) * 5.);
    let mut value = 0.70 + 0.16 * growth + 0.09 * (fibres - 0.5)
        - (0.18 * annual * annual + 0.32 * late) * pigmentation
        - 0.12 * pores;
    // Sparse elliptical branch knots bend the surrounding fibres. Wrapped
    // distances keep both colour and its relief derivative continuous at seams.
    let knot_amount = smooth((hash(43, 9, seed) - 0.38) * 2.);
    let dx = wrapped(u - hash(17, 23, seed));
    let dy = wrapped(v - hash(37, 41, seed));
    let radius = (dx * dx * 100. + dy * dy * 22.).sqrt();
    let influence = (1. - smooth((radius - 0.30) / 0.70)) * knot_amount;
    let knot = 0.30 + 0.13 * (radius * 37. + growth * 4.).sin() - 0.22 * (-radius * 10.).exp();
    value += (knot - value) * influence;
    value.clamp(0., 1.)
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0., 1.);
    t * t * (3. - 2. * t)
}
fn wrapped(t: f32) -> f32 {
    (t + 0.5).rem_euclid(1.) - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staggered_plank_has_no_colour_seam_at_the_texture_wrap() {
        let mut r = super::super::program::sample(91).remove(Surface::Floor as usize);
        r.phase = [0.0, 0.0];
        r.layers.as_mut().unwrap().quarter_turn = 0;
        r.period_m = 2.3;
        r.panel_count = [5, 3];
        let [nx, _] = r.floor_repetitions(0);
        let a = r.evaluate(1.5 / nx, 0.000001, 0);
        let b = r.evaluate(1.5 / nx, 0.999999, 0);
        assert!(
            (a.0 - b.0).abs() < 0.001 && (a.1 - b.1).abs() < 0.000001 && (a.2 - b.2).abs() < 0.001,
            "{a:?} vs {b:?}"
        );
    }
}
