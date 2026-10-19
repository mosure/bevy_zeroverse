//! Composable document primitives. Each panel has an independent data/content seed.
use super::{
    raster::Canvas,
    screens::{chart, map, palette, parameters, participants},
};
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;

pub(super) const WORDS: [&str; 48] = [
    "PROJECT", "REVIEW", "DESIGN", "SYSTEM", "NOTES", "MODEL", "STUDY", "DATA", "PLAN", "TEAM",
    "FIELD", "TIME", "TASK", "IDEA", "SAMPLE", "GROUP", "RESULT", "METHOD", "VALUE", "PHASE",
    "EVENT", "INPUT", "OUTPUT", "TEST", "PROCESS", "TARGET", "DETAIL", "CHANGE", "REPORT",
    "BUDGET", "STATUS", "MAP", "ROOM", "LEVEL", "AREA", "ROUTE", "TREND", "RATE", "TOTAL", "INDEX",
    "ENERGY", "COST", "WEEK", "MONTH", "YEAR", "SERIES", "OPTION", "STAGE",
];

pub(super) fn phrase(rng: &mut impl Rng, letters: usize) -> String {
    let mut text = String::new();
    while text.len() + 5 < letters {
        let word = WORDS[rng.random_range(0..WORDS.len())];
        if text.len() + word.len() + 1 > letters {
            break;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(word);
    }
    text
}

pub(super) fn paragraph(c: &mut Canvas, seed: u64, bounds: [i32; 4], color: [u8; 3], density: f32) {
    let mut rng = stream(seed, 9021);
    let [x, y, w, h] = bounds;
    let spacing = rng.random_range(8..12);
    for row in 0..h / spacing {
        if rng.random_bool((0.45 + density * 0.5) as f64) {
            let length = (w as f32 * rng.random_range(0.4..1.)) as usize / 4;
            let text = phrase(&mut rng, length);
            c.text(&text, x, y + row * spacing, 1, color);
        }
    }
}

/// All drawing stays within the supplied clip. Bounded counts and no external fonts/media.
pub(super) fn panel(c: &mut Canvas, seed: u64, bounds: [i32; 4], kind: u32, dark: bool) {
    let mut p = parameters(seed);
    p.dark = dark;
    let colors = palette(&p);
    let mut rng = stream(seed, 9022);
    let [x, y, w, h] = bounds;
    c.clipped(bounds, |c| {
        c.rect(x, y, w, h, colors.panel);
        let margin = ((w.min(h) as f32 * rng.random_range(0.025..0.09)) as i32).clamp(2, 10);
        let header = if rng.random_bool(0.65) {
            rng.random_range(12..23)
        } else {
            0
        };
        if header > 0 {
            let title = phrase(&mut rng, ((w - 2 * margin) / 4).max(5) as usize);
            c.text(&title, x + margin, y + margin, 1, colors.ink);
        }
        let b = [
            x + margin,
            y + margin + header,
            (w - 2 * margin).max(6),
            (h - 2 * margin - header).max(6),
        ];
        let [x, y, w, h] = b;
        c.clipped(b, |c| match kind % 13 {
            0 => paragraph(c, seed, b, colors.ink, p.density),
            1 => chart(c, seed, &p, b, colors, rng.random_range(0..6)),
            2 => {
                let cols = rng.random_range(2..=5).min((w / 20).max(2));
                let rows = (h / rng.random_range(10..18)).max(1);
                for row in 0..rows {
                    if row % 2 == 0 {
                        c.rect(x, y + row * h / rows, w, h / rows, colors.bg);
                    }
                    for col in 0..cols {
                        let xx = x + col * w / cols + 2;
                        if col == 0 {
                            let word = WORDS[rng.random_range(0..WORDS.len())];
                            c.clipped(
                                [xx, y + row * h / rows + 2, w / cols - 3, h / rows - 3],
                                |c| c.text(word, xx, y + row * h / rows + 2, 1, colors.ink),
                            );
                        } else {
                            for j in 0..rng.random_range(1..=3).min((w / cols - 4) / 7) {
                                c.digit(
                                    rng.random_range(0..10),
                                    xx + j * 7,
                                    y + row * h / rows + 1,
                                    1,
                                    colors.muted,
                                );
                            }
                        }
                    }
                }
            }
            3 => {
                let n = (h / 14).clamp(1, 8);
                for i in 0..n {
                    let start = rng.random_range(0.0..0.62);
                    let duration = rng.random_range(0.12..(1. - start));
                    let xx = x + (start * w as f32) as i32;
                    c.rect(x, y + i * h / n, w, 1, colors.bg);
                    c.rect(
                        xx,
                        y + i * h / n + 3,
                        (duration * w as f32) as i32,
                        7,
                        if i % 3 == 0 {
                            colors.accent
                        } else {
                            colors.muted
                        },
                    );
                }
            }
            4 => {
                let nodes = rng.random_range(3..=8);
                let points: Vec<_> = (0..nodes)
                    .map(|_| {
                        [
                            x as f32 + rng.random_range(0.12..0.88) * w as f32,
                            y as f32 + rng.random_range(0.12..0.88) * h as f32,
                        ]
                    })
                    .collect();
                for i in 1..points.len() {
                    c.line(points[i], points[rng.random_range(0..i)], 1, colors.muted);
                }
                for point in points {
                    c.ellipse(point, [4., 3.], colors.accent);
                }
            }
            5 => {
                // Code: independent indentation, token lengths and syntax colors.
                for row in 0..h / 8 {
                    let xx = x + rng.random_range(0..4) * 5;
                    let mut cursor = xx;
                    for token in 0..rng.random_range(2..=6) {
                        let word = WORDS[rng.random_range(0..WORDS.len())];
                        let ink = if token % 3 == 0 {
                            colors.accent
                        } else {
                            colors.ink
                        };
                        c.text(word, cursor, y + row * 8, 1, ink);
                        cursor += word.len() as i32 * 4 + 4;
                        if cursor >= x + w {
                            break;
                        }
                    }
                }
            }
            6 => participants(c, seed, &p, b, colors),
            7 => map(c, seed, &p, b, colors),
            8 => {
                // Continuous procedural media: layered terrain and sky, cropped to panel.
                let horizon = y as f32 + h as f32 * rng.random_range(0.3..0.7);
                c.rect(
                    x,
                    y,
                    w,
                    h,
                    if dark { [27, 46, 62] } else { [167, 194, 207] },
                );
                c.ellipse(
                    [
                        x as f32 + w as f32 * rng.random_range(0.2..0.8),
                        y as f32 + h as f32 * 0.2,
                    ],
                    [h as f32 * 0.08; 2],
                    [235, 222, 180],
                );
                for layer in 0..3 {
                    let mut points = vec![[x as f32, (y + h) as f32]];
                    for i in 0..=8 {
                        points.push([
                            x as f32 + w as f32 * i as f32 / 8.,
                            horizon
                                + layer as f32 * h as f32 * 0.14
                                + rng.random_range(-0.15..0.12) * h as f32,
                        ]);
                    }
                    points.push([(x + w) as f32, (y + h) as f32]);
                    c.polygon(&points, [43 + layer * 13, 72 + layer * 11, 75 - layer * 8]);
                }
            }
            9 => {
                // Slide/kanban cards: sparse independent placements in ordered columns.
                let cols = if w > 100 { rng.random_range(2..=3) } else { 1 };
                for col in 0..cols {
                    let cw = w / cols;
                    for row in 0..(h / 30).max(1) {
                        if rng.random_bool(0.2) {
                            continue;
                        }
                        let b = [x + col * cw, y + row * 30, cw - 3, 26];
                        c.rect(b[0], b[1], b[2], b[3], colors.bg);
                        c.rect(b[0], b[1], 3, 26, colors.accent);
                        paragraph(
                            c,
                            seed.wrapping_add((row * 7 + col) as u64),
                            [b[0] + 6, b[1] + 5, cw - 13, 18],
                            colors.ink,
                            p.density,
                        );
                    }
                }
            }
            10 => {
                // Plan/editor viewport: unequal connected rooms, wall thickness,
                // doors and furniture. Positions are not screen-grid landmarks.
                let plan = super::composition::Composition::sample(
                    seed,
                    rng.random_range(2..=6),
                    [0.03, 0.05, 0.94, 0.90],
                    rng.random_range(0.25..0.75),
                );
                c.rect(x, y, w, h, colors.bg);
                for r in plan.regions {
                    let [rx, ry, rw, rh] = r.bounds;
                    let a = x + (rx * w as f32) as i32;
                    let b = y + (ry * h as f32) as i32;
                    let ww = (rw * w as f32) as i32;
                    let hh = (rh * h as f32) as i32;
                    c.rect(a, b, ww, hh, colors.muted);
                    c.rect(a + 2, b + 2, (ww - 4).max(1), (hh - 4).max(1), colors.panel);
                    c.rect(a + ww / 3, b, (ww / 5).max(2), 2, colors.panel);
                    c.rect(
                        a + ww / 4,
                        b + hh / 3,
                        (ww / 3).max(2),
                        (hh / 4).max(2),
                        colors.accent,
                    );
                }
            }
            11 => {
                // Calendar: responsive week/month spans and sparse appointments.
                let cols = rng.random_range(3..=7);
                let rows = rng.random_range(2..=5);
                for i in 0..cols * rows {
                    let xx = x + i % cols * w / cols;
                    let yy = y + i / cols * h / rows;
                    c.rect(xx, yy, 1, h / rows, colors.muted);
                    c.rect(xx, yy, w / cols, 1, colors.muted);
                    if rng.random_bool(p.density as f64) {
                        c.rect(
                            xx + 2,
                            yy + 4,
                            (w / cols - 4).max(1),
                            (h / rows / 3).max(1),
                            colors.accent,
                        );
                    }
                }
            }
            _ => {
                // Full-bleed image/photo artwork, with no fixed terrain palette.
                let image = super::art::illustration(seed);
                for yy in 0..h.max(1) {
                    for xx in 0..w.max(1) {
                        let src =
                            (((yy * 255 / h.max(1)) * 256 + xx * 255 / w.max(1)) * 4) as usize;
                        c.rect(
                            x + xx,
                            y + yy,
                            1,
                            1,
                            [image[src], image[src + 1], image[src + 2]],
                        );
                    }
                }
            }
        });
    });
}
