//! Metric dial artwork. Drawing in physical proportions keeps circular marks
//! circular and type upright on narrow, wide and square housings.
use super::*;
use crate::scene::procedural_indoor::materials::raster::{Canvas, N};
use std::f32::consts::TAU;

struct Chart(Vec2);
impl Chart {
    fn new(o: &IndoorObject, p: &ClockProgram) -> Self {
        Self(Vec2::splat(N as f32) / (o.size.xy() * (1. - p.bezel_fraction)))
    }
    fn point(&self, pos: Vec2) -> [f32; 2] {
        (Vec2::splat(N as f32 * 0.5) + pos * self.0).to_array()
    }
    fn dot(&self, c: &mut Canvas, pos: Vec2, radius: f32, color: [u8; 3]) {
        c.ellipse(self.point(pos), (radius * self.0).to_array(), color);
    }
    fn stroke(&self, c: &mut Canvas, a: Vec2, b: Vec2, width: f32, color: [u8; 3]) {
        let side = (b - a).normalize().perp() * (width * 0.5);
        c.polygon(
            &[a - side, b - side, b + side, a + side].map(|v| self.point(v)),
            color,
        );
    }
}

// Single-stroke Arabic dial numerals on a 4 by 6 design grid. Glyphs stay upright.
const NUMERALS: [&[(u8, u8)]; 10] = [
    &[
        (1, 0),
        (3, 0),
        (4, 1),
        (4, 5),
        (3, 6),
        (1, 6),
        (0, 5),
        (0, 1),
        (1, 0),
    ],
    &[(0, 2), (2, 0), (2, 6)],
    &[(0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (0, 6), (4, 6)],
    &[
        (0, 0),
        (4, 0),
        (2, 3),
        (3, 3),
        (4, 4),
        (4, 5),
        (3, 6),
        (0, 6),
    ],
    &[(3, 6), (3, 0), (0, 4), (4, 4)],
    &[
        (4, 0),
        (0, 0),
        (0, 3),
        (3, 3),
        (4, 4),
        (4, 5),
        (3, 6),
        (0, 6),
    ],
    &[
        (4, 0),
        (2, 0),
        (0, 2),
        (0, 5),
        (1, 6),
        (3, 6),
        (4, 5),
        (4, 4),
        (3, 3),
        (0, 3),
    ],
    &[(0, 0), (4, 0), (1, 6)],
    &[
        (1, 3),
        (0, 2),
        (0, 1),
        (1, 0),
        (3, 0),
        (4, 1),
        (4, 2),
        (3, 3),
        (1, 3),
        (0, 4),
        (0, 5),
        (1, 6),
        (3, 6),
        (4, 5),
        (4, 4),
        (3, 3),
    ],
    &[
        (4, 3),
        (1, 3),
        (0, 2),
        (0, 1),
        (1, 0),
        (3, 0),
        (4, 1),
        (4, 4),
        (2, 6),
        (0, 6),
    ],
];
fn numeral(c: &mut Canvas, chart: &Chart, center: Vec2, n: usize, height: f32, ink: [u8; 3]) {
    let scale = Vec2::new(height * 0.134, height / 6.);
    let pitch = 5.9 * scale.x;
    let mut digit = |n: usize, offset: f32| {
        let coord = |(x, y): (u8, u8)| {
            center + Vec2::X * offset + (Vec2::new(x as f32, y as f32) - Vec2::new(2., 3.)) * scale
        };
        let width = height * 0.114;
        let mut points = NUMERALS[n].iter().copied().map(coord);
        let mut prev = points.next().unwrap();
        for point in points {
            chart.stroke(c, prev, point, width, ink);
            chart.dot(c, prev, width * 0.5, ink);
            prev = point;
        }
        chart.dot(c, prev, width * 0.5, ink);
    };
    if n >= 10 {
        digit(1, -pitch * 0.5);
        digit(n % 10, pitch * 0.5);
    } else {
        digit(n, 0.);
    }
}

pub(super) fn analog(
    c: &mut Canvas,
    o: &IndoorObject,
    p: &ClockProgram,
    dial: PrintedDial,
    fg: [u8; 3],
) {
    let chart = Chart::new(o, p);
    let radius = o.size.x.min(o.size.y) * 0.5;
    let r = radius * p.tick_radius;
    if dial.minute_rail {
        c.ring(
            chart.point(Vec2::ZERO),
            ((r + 0.009 * radius) * chart.0).to_array(),
            fg,
        );
    }
    for i in 0..60 {
        let major = i % 5 == 0;
        if !major && !p.minor_ticks {
            continue;
        }
        let a = TAU * i as f32 / 60.;
        let d = Vec2::new(a.sin(), -a.cos());
        if major && dial.style == DialStyle::Dots {
            chart.dot(
                c,
                d * (r - radius * 0.034),
                radius * if i % 15 == 0 { 0.048 } else { 0.032 },
                fg,
            );
        } else {
            let size = if major {
                Vec2::new(0.023, 0.105)
            } else {
                Vec2::new(0.010, 0.037)
            };
            chart.stroke(c, d * (r - radius * size.y), d * r, radius * size.x, fg);
        }
        if major
            && (dial.style == DialStyle::ArabicNumerals
                || (dial.style == DialStyle::QuarterNumerals && i % 15 == 0))
        {
            let h = radius * dial.numeral_height;
            let n = if i == 0 { 12 } else { i / 5 };
            numeral(c, &chart, d * (r - 0.14 * radius - h * 0.66), n, h, fg);
        }
    }
}

pub(super) fn digital(
    c: &mut Canvas,
    mut emission: Option<&mut Canvas>,
    o: &IndoorObject,
    p: &ClockProgram,
    dial: PrintedDial,
    bg: [u8; 3],
    fg: [u8; 3],
) {
    let chart = Chart::new(o, p);
    let half = o.size.xy() * (0.5 * (1. - p.bezel_fraction));
    let width = o.size.x * p.digit_width_fraction;
    let height = (o.size.y * p.digit_height_fraction).min(1.9 * width);
    let fit = ((2.50 * width / half.x).powf(p.roundness)
        + (0.55 * height / half.y).powf(p.roundness))
    .powf(1. / p.roundness)
    .max(1.);
    let (w, h) = (width / fit, height / fit);
    let thick = (w * p.digit_stroke_fraction * 1.30).min(0.14 * h);
    let off = std::array::from_fn(|i| (0.925 * bg[i] as f32 + 0.075 * fg[i] as f32).round() as u8);
    let segments = [
        ((-0.43, -0.5), (0.43, -0.5)),
        ((0.5, -0.44), (0.5, -0.06)),
        ((0.5, 0.06), (0.5, 0.44)),
        ((-0.43, 0.5), (0.43, 0.5)),
        ((-0.5, 0.06), (-0.5, 0.44)),
        ((-0.5, -0.44), (-0.5, -0.06)),
        ((-0.43, 0.), (0.43, 0.)),
    ];
    for (i, num) in dial.digits(p.seconds).into_iter().enumerate() {
        let mask = num.map_or(0, |n| screens::SEGMENTS[n as usize]);
        let x = ((i as f32 - 1.5) * 1.20 + if i < 2 { -0.08 } else { 0.08 }) * w;
        let point = |(u, v)| Vec2::new(x + u * w, v * h);
        for (bit, &(a, b)) in segments.iter().enumerate() {
            let (a, b) = (point(a), point(b));
            let d = (b - a).normalize() * thick * 0.46;
            let e = d.perp();
            let poly = [a, a + d - e, b - d - e, b, b - d + e, a + d + e].map(|v| chart.point(v));
            let active = mask & (1 << bit) != 0;
            c.polygon(&poly, if active { fg } else { off });
            if active {
                if let Some(glow) = emission.as_deref_mut() {
                    glow.polygon(&poly, fg);
                }
            }
        }
    }
    for y in [-0.185 * h, 0.185 * h] {
        let pos = Vec2::Y * y;
        chart.dot(c, pos, thick * 0.59, fg);
        if let Some(glow) = emission.as_deref_mut() {
            chart.dot(glow, pos, thick * 0.59, fg);
        }
    }
}
