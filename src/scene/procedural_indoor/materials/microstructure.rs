//! Substrate-specific, periodic structure. Heights are metres, not painted shadows.
use super::{hash, periodic_noise, program::MaterialRecipe, Surface};
use std::f32::consts::TAU;

pub(super) fn value(r: &MaterialRecipe, uv: [f32; 2], noise: [f32; 3], floor: u32) -> f32 {
    let [u, v] = uv;
    let [macro_n, meso, micro] = noise;
    let sample = |nx, ny, salt| periodic_noise(u, v, nx, ny, r.seed.wrapping_add(salt));
    match r.surface {
        Surface::Wood | Surface::WoodEdge | Surface::Bark => super::timber::grain(r, u, v),
        Surface::Floor if floor == 0 => super::timber::grain(r, u, v),
        Surface::Fabric | Surface::FabricAlt => textile(r, uv, [meso, micro]),
        Surface::Floor if floor == 1 => textile(r, uv, [meso, micro]),
        Surface::Leather => {
            // Irregular cells with narrow creases: leather grain is not the
            // independent pixel noise used by injection-moulded plastic.
            let cells = (r.period_m / 0.0025).round().clamp(12., 64.) as i32;
            0.84 * pebbles(uv, cells, r.seed) + 0.16 * micro
        }
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

fn textile(r: &MaterialRecipe, [u, v]: [f32; 2], [meso, micro]: [f32; 2]) -> f32 {
    let count = (r.period_m / 0.0025).round().clamp(12., 80.);
    let warp = (TAU * u * count).cos();
    let weft = (TAU * v * count).cos();
    // Alternate which rounded yarn crosses on top, with a continuous plain /
    // twill mixture and a small correlated yarn-thickness irregularity.
    let over = (TAU * (u + v) * (count * 0.5).round()).cos();
    let plain = warp.max(weft) * 0.65 + over * (warp - weft) * 0.18;
    let twill = (TAU * (u - v) * (count * 0.5).round()).cos();
    0.5 + 0.34 * (plain * (1. - r.weave_mix) + twill * r.weave_mix)
        + 0.12 * (meso - 0.5)
        + 0.08 * (micro - 0.5)
}

fn pebbles([u, v]: [f32; 2], count: i32, seed: u64) -> f32 {
    let x = u * count as f32;
    let y = v * count as f32;
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let mut nearest = [f32::INFINITY; 2];
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let (hx, hy) = (cx.rem_euclid(count) as u32, cy.rem_euclid(count) as u32);
            let px = cx as f32 + 0.25 + 0.5 * hash(hx, hy, seed);
            let py = cy as f32 + 0.25 + 0.5 * hash(hx, hy, seed.wrapping_add(31));
            let d = (px - x).powi(2) + (py - y).powi(2);
            if d < nearest[0] {
                nearest = [d, nearest[0]];
            } else if d < nearest[1] {
                nearest[1] = d;
            }
        }
    }
    let t = ((nearest[1] - nearest[0]) * 4.).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leather_grain_is_periodic_at_metric_tile_boundaries() {
        for seed in 0..32 {
            for count in [12, 19, 64] {
                for t in [0.001, 0.13, 0.41, 0.85] {
                    assert!(
                        (pebbles([0., t], count, seed) - pebbles([1., t], count, seed)).abs()
                            < 0.0001
                    );
                    assert!(
                        (pebbles([t, 0.], count, seed) - pebbles([t, 1.], count, seed)).abs()
                            < 0.0001
                    );
                    assert!((0. ..=1.).contains(&pebbles([t, t], count, seed)));
                }
            }
        }
    }
}
