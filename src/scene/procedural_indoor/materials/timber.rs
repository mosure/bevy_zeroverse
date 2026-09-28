//! Correlated timber pores and growth grain, cut independently per floor plank.
use super::{hash, periodic_noise, program::MaterialRecipe, Surface};

pub(super) fn grain(r: &MaterialRecipe, mut u: f32, mut v: f32) -> f32 {
    let mut seed = r.seed;
    if r.surface == Surface::Floor {
        let [nx, ny] = r.floor_repetitions(0);
        let x = (u * nx).floor() as u32;
        let stagger = (x % 2) as f32 * 0.5;
        let y = ((v * ny + stagger).floor() as u32) % ny as u32;
        // Different cuts of one species share a spectrum but never continue
        // the very same growth ring across a joint into the next plank.
        u += hash(x, y, seed.wrapping_add(79));
        v += hash(x, y, seed.wrapping_add(83));
        seed = seed
            .wrapping_add(u64::from(x) * 0x9e3779b9)
            .wrapping_add(u64::from(y) * 0x85ebca6b);
    }
    let warp = (periodic_noise(u, v, 3, 4, seed.wrapping_add(4)) - 0.5) * r.warp;
    // Multiple elongated frequency bands, with a continuous mixture instead of
    // a uniform set of sine stripes. Fine open pores are correlated with relief.
    let frequency = r.grain_frequency;
    let sample = |factor: f32, salt: u64| {
        let f = (frequency * factor).clamp(4.0, 120.0);
        let n = f.floor() as u32;
        let cross = (r.cross_frequency * factor.sqrt()).round().max(1.0) as u32;
        let noise = |n| periodic_noise(u + warp, v, n, cross, seed.wrapping_add(salt));
        noise(n) * (1.0 - f.fract()) + noise(n + 1) * f.fract()
    };
    let growth = sample(0.42, 13);
    let fibres = sample(1.0, 0);
    let pores = ((sample(2.0, 47) - 0.56) * 3.0).clamp(0.0, 1.0);
    (0.36 * growth + 0.64 * fibres - 0.20 * pores).clamp(0.0, 1.0)
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
