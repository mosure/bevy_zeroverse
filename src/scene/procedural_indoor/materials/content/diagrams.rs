//! Parameterized visual documents: distributions, trees, technical sections and
//! planning notes. Connectivity and hierarchy organize the noise samples.
use super::*;
use crate::scene::procedural_indoor::materials::screens::Palette;

pub(super) fn draw(c: &mut Canvas, seed: u64, bounds: [i32; 4], kind: u32, colors: Palette) {
    let mut rng = stream(seed, 0x4449414752414d);
    let [x, y, w, h] = bounds;
    let at = |u: f32, v: f32| [x as f32 + u * w as f32, y as f32 + v * h as f32];
    match kind {
        13 => {
            // A scatter study samples continuous covariance and clustered means.
            let slope: f32 = rng.random_range(-0.7..0.7);
            let spread = rng.random_range(0.025..0.24);
            for i in 0..rng.random_range(25..=160) {
                let u = rng.random_range(0.08..0.94);
                let v =
                    (0.5 + slope * (u - 0.5) + rng.random_range(-spread..spread)).clamp(0.05, 0.95);
                c.ellipse(
                    at(u, v),
                    [rng.random_range(0.8..2.3); 2],
                    if i % 4 == 0 {
                        colors.accent
                    } else {
                        colors.muted
                    },
                );
            }
            c.line(at(0.04, 0.95), at(0.96, 0.95), 1, colors.ink);
            c.line(at(0.04, 0.95), at(0.04, 0.04), 1, colors.ink);
        }
        14 => {
            // Related matrix values, not independent random checkerboard cells.
            let cols = rng.random_range(4..=15);
            let rows = rng.random_range(3..=12);
            let peak = [rng.random_range(0.1..0.9), rng.random_range(0.1..0.9)];
            let spread = rng.random_range(0.05..0.4);
            for row in 0..rows {
                for col in 0..cols {
                    let d = ((col as f32 / cols as f32 - peak[0]).powi(2)
                        + (row as f32 / rows as f32 - peak[1]).powi(2))
                        / spread;
                    let value = ((-d).exp() + rng.random_range(-0.12..0.12)).clamp(0., 1.);
                    let rgb = std::array::from_fn(|i| {
                        (colors.bg[i] as f32 * (1. - value) + colors.accent[i] as f32 * value) as u8
                    });
                    c.rect(
                        x + col * w / cols,
                        y + row * h / rows,
                        (w / cols - 1).max(1),
                        (h / rows - 1).max(1),
                        rgb,
                    );
                }
            }
        }
        15 => {
            // Mind-map / organization tree, children branch from parent nodes.
            let count = rng.random_range(4..=14);
            let vertical = rng.random_bool(0.5);
            let pts: Vec<_> = (0..count)
                .map(|i| {
                    let depth = ((i + 1) as f32).log2().floor();
                    let siblings = 2_f32.powf(depth);
                    let a = (i as f32 + 1. - siblings + 0.5) / siblings;
                    let b = (depth + 0.5) / ((count as f32).log2().floor() + 1.);
                    if vertical {
                        at(a, b)
                    } else {
                        at(b, a)
                    }
                })
                .collect();
            for i in 1..pts.len() {
                c.line(pts[(i - 1) / 2], pts[i], 1, colors.muted);
            }
            for (i, pt) in pts.iter().enumerate() {
                let rw = (w as f32 * rng.random_range(0.04..0.10)).max(2.);
                let rh = (h as f32 * rng.random_range(0.025..0.06)).max(2.);
                c.rect(
                    (pt[0] - rw) as i32,
                    (pt[1] - rh) as i32,
                    (2. * rw) as i32,
                    (2. * rh) as i32,
                    if i % 3 == 0 {
                        colors.accent
                    } else {
                        colors.muted
                    },
                );
            }
        }
        16 => {
            // Engineering section: an actual stepped profile and dimension lines.
            let steps = rng.random_range(3..=8);
            let mut edge = vec![at(0.08, 0.8)];
            let mut level = rng.random_range(0.25..0.65);
            for i in 0..=steps {
                let u = 0.08 + 0.84 * i as f32 / steps as f32;
                edge.push(at(u, level));
                level = rng.random_range(0.18..0.7);
                edge.push(at(u, level));
            }
            edge.push(at(0.92, 0.8));
            c.polygon(&edge, colors.muted);
            for (a, b) in edge.iter().zip(edge.iter().skip(1)) {
                c.line(*a, *b, 1, colors.accent);
            }
            c.line(at(0.08, 0.9), at(0.92, 0.9), 1, colors.ink);
            for u in [0.08, 0.92] {
                c.line(at(u, 0.78), at(u, 0.96), 1, colors.ink);
            }
            c.digit(
                rng.random_range(1..10),
                x + w / 2,
                y + (h as f32 * 0.84) as i32,
                1,
                colors.ink,
            );
        }
        17 => {
            // Task board with irregular note sizes, whitespace and sparse columns.
            let composition =
                crate::scene::procedural_indoor::materials::composition::Composition::clusters(
                    seed,
                    rng.random_range(3..=10),
                    [0.02, 0.02, 0.96, 0.96],
                );
            for r in composition.regions {
                let [a, b, d, e] = r.bounds;
                let bb = [
                    x + (a * w as f32) as i32,
                    y + (b * h as f32) as i32,
                    (d * w as f32) as i32,
                    (e * h as f32) as i32,
                ];
                c.rect(bb[0], bb[1], bb[2], bb[3], colors.accent);
                c.clipped(bb, |c| {
                    paragraph(
                        c,
                        r.seed,
                        [bb[0] + 2, bb[1] + 3, (bb[2] - 4).max(1), (bb[3] - 5).max(1)],
                        colors.ink,
                        0.7,
                    )
                });
            }
        }
        _ => {
            // Large editorial type with ragged captions; title scale follows
            // available area instead of a constant bitmap header on every pane.
            let size = rng
                .random_range(2..=5)
                .min((w / 12).max(1))
                .min((h / 10).max(1));
            let title = phrase(&mut rng, (w / (4 * size)).max(1) as usize);
            c.text(&title, x, y, size, colors.ink);
            let dy = size * 5 + rng.random_range(5..16);
            paragraph(
                c,
                seed,
                [x, y + dy, w, (h - dy).max(0)],
                colors.muted,
                rng.random_range(0.25..0.9),
            );
        }
    }
}
