//! Single-line drafting glyphs with bounded pen slant, spacing and baseline drift.
use super::super::raster::Canvas;
use rand::Rng;

// Point coordinates are digits on a 5x7 pen grid; spaces lift the pen.
const GLYPHS: [&str; 26] = [
    "062046 1434",
    "0006404626342000 0424",
    "402000062646",
    "00062646204000",
    "40000646 0424",
    "400006 0434",
    "4020000626464434",
    "0006 4046 0444",
    "0040 2026 0646",
    "00404046260604",
    "0006 400346",
    "000646",
    "0600204046",
    "06004640",
    "2000062646402000",
    "06002040222404",
    "2000062646402000 2446",
    "06002040222404 2446",
    "402000022444462606",
    "0040 2026",
    "0006264640",
    "002640",
    "0016243640",
    "0046 4006",
    "002240 2226",
    "00400646",
];
pub(super) fn text(
    c: &mut Canvas,
    rng: &mut impl Rng,
    text: &str,
    at: [f32; 2],
    height: f32,
    color: [u8; 3],
) {
    let scale = height / 6.;
    let slant = rng.random_range(-0.12..0.24);
    let mut x = at[0];
    let mut drift = 0.0_f32;
    for letter in text.bytes() {
        if !letter.is_ascii_alphabetic() {
            x += scale * 3.;
            continue;
        }
        drift = (drift + rng.random_range(-0.16..0.16)).clamp(-0.6, 0.6);
        let stretch = rng.random_range(0.85..1.12);
        let opacity = rng.random_range(0.76..1.);
        for path in GLYPHS[(letter.to_ascii_uppercase() - b'A') as usize].split(' ') {
            let points: Vec<_> = path
                .as_bytes()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| {
                    let px = (p[0] - b'0') as f32;
                    let py = (p[1] - b'0') as f32;
                    [
                        x + (px * stretch + (6. - py) * slant) * scale,
                        at[1] + py * scale + drift,
                    ]
                })
                .collect();
            for segment in points.windows(2) {
                let [a, b] = [segment[0], segment[1]];
                let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()) * 2.)
                    .ceil()
                    .max(1.) as usize;
                for i in 0..=n {
                    let t = i as f32 / n as f32;
                    let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                    // Subpixel coverage, not hard square steps at diagonal strokes.
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let alpha = (if dx == 0 {
                                1. - p[0].fract()
                            } else {
                                p[0].fract()
                            }) * (if dy == 0 {
                                1. - p[1].fract()
                            } else {
                                p[1].fract()
                            }) * opacity;
                            c.blend(
                                p[0].floor() as i32 + dx,
                                p[1].floor() as i32 + dy,
                                color,
                                alpha,
                            );
                        }
                    }
                }
            }
        }
        x += scale * (4. * stretch + rng.random_range(1.2..2.2));
    }
}
