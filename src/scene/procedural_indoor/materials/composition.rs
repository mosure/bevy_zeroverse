//! Bounded surface-space subdivision. Layout, content and finish use disjoint RNG streams.
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Region {
    /// Left, top, width, height in the normalized surface domain.
    pub bounds: [f32; 4],
    pub seed: u64,
}
impl Region {
    pub(super) fn pixels(&self) -> [i32; 4] {
        self.bounds.map(|v| (v * 256.).round() as i32)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Composition {
    pub gutter: f32,
    pub regions: Vec<Region>,
}
impl Composition {
    /// A count is an upper bound, not permission to create unreadably small cells.
    pub fn sample(seed: u64, count: usize, bounds: [f32; 4], vertical_bias: f32) -> Self {
        let mut rng = stream(seed, 0x50414e454c53);
        let gutter = rng.random_range(0.012..0.035);
        let mut regions = vec![Region {
            bounds,
            seed: rng.random(),
        }];
        for _ in 1..count.clamp(1, 6) {
            // Prioritize large regions, with independently varied split order.
            let candidate = regions
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let [_, _, w, h] = r.bounds;
                    (w > 0.40 || h > 0.34).then(|| (i, w * h * rng.random_range(0.65..1.35)))
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            let Some((i, _)) = candidate else { break };
            let r = regions.remove(i);
            let [x, y, w, h] = r.bounds;
            let vertical =
                w > 0.40 && (h <= 0.34 || rng.random_bool(vertical_bias.clamp(0., 1.) as f64));
            let split = rng.random_range(0.36..0.64);
            let pair = if vertical {
                let left = ((w - gutter) * split).clamp(0.18, w - gutter - 0.18);
                [
                    [x, y, left, h],
                    [x + left + gutter, y, w - left - gutter, h],
                ]
            } else {
                let top = ((h - gutter) * split).clamp(0.15, h - gutter - 0.15);
                [[x, y, w, top], [x, y + top + gutter, w, h - top - gutter]]
            };
            regions.extend(pair.map(|bounds| Region {
                bounds,
                seed: rng.random(),
            }));
        }
        Self { gutter, regions }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subdivision_is_bounded_nonoverlapping_and_replayable() {
        for seed in 0..4096 {
            let c = Composition::sample(
                seed,
                (seed % 6 + 1) as usize,
                [0.04, 0.06, 0.92, 0.88],
                0.55,
            );
            assert_eq!(
                c,
                Composition::sample(
                    seed,
                    (seed % 6 + 1) as usize,
                    [0.04, 0.06, 0.92, 0.88],
                    0.55
                )
            );
            assert!((1..=6).contains(&c.regions.len()));
            for (i, r) in c.regions.iter().enumerate() {
                let [x, y, w, h] = r.bounds;
                assert!(r.bounds.iter().all(|v| v.is_finite()));
                assert!(x >= 0.039 && y >= 0.059 && w >= 0.179 && h >= 0.149);
                assert!(x + w <= 0.961 && y + h <= 0.941);
                for s in &c.regions[..i] {
                    let [a, b, c, d] = s.bounds;
                    assert!(x + w <= a || a + c <= x || y + h <= b || b + d <= y);
                }
            }
        }
    }
}
