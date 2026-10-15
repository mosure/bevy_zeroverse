//! Independent handwritten regions over a locally erased enamel surface.
use super::*;
pub(super) fn pixels(seed: u64, p: &BoardProgram) -> Vec<u8> {
    let mut rng = stream(seed, 0x4d41524b455253);
    let base = [246., 246. - p.warmth * 130., 244. - p.warmth * 230.].map(|v| v as u8);
    let mut c = Canvas::new(base);
    // Old ink is a separate coherent pass, locally wiped rather than global gray noise.
    let mut erased = Canvas::new(base);
    for r in &p.regions.regions {
        writing(
            &mut erased,
            &mut rng,
            r.pixels()[0],
            r.pixels()[1] + 14,
            r.pixels()[0] + r.pixels()[2],
            INKS[0],
            1,
        );
        erased.ring(
            [rng.random_range(30.0..225.), rng.random_range(30.0..225.)],
            [rng.random_range(8.0..35.), rng.random_range(6.0..25.)],
            INKS[1],
        );
    }
    let wipe = [
        rng.random_range(0.1..0.9),
        rng.random_range(0.1..0.9),
        rng.random_range(0.15..0.65),
    ];
    for (i, px) in erased.0.as_chunks::<4>().0.iter().enumerate() {
        let u = (i % 256) as f32 / 256.;
        let v = (i / 256) as f32 / 256.;
        let d = ((u - wipe[0]).powi(2) + (v - wipe[1]).powi(2)) / wipe[2];
        let amount = p.erasure * (0.2 + 0.8 * d.clamp(0., 1.));
        c.blend(
            (i % 256) as i32,
            (i / 256) as i32,
            [px[0], px[1], px[2]],
            amount,
        );
    }
    if p.blank {
        return c.0;
    }
    for (i, r) in p.regions.regions.iter().enumerate() {
        let mut rng = stream(r.seed, 0x4e4f544553);
        let [x, y, w, h] = r.pixels();
        let ink = INKS[(p.ink_index as usize + i % p.color_count as usize) % 4];
        let kind = rng.random_range(0..7);
        c.clipped([x, y, w, h], |c| {
            let heading = content::phrase(&mut rng, (w as f32 / 8.) as usize);
            if rng.random_bool(0.72) {
                handwriting::text(
                    c,
                    &mut rng,
                    &heading,
                    [x as f32 + 3., y as f32 + 3.],
                    8.,
                    ink,
                );
                if rng.random_bool(0.45) {
                    c.line(
                        [x as f32, y as f32 + 14.],
                        [(x + w) as f32, y as f32 + 13.],
                        1,
                        ink,
                    );
                }
            }
            let y = y + 20;
            let h = (h - 23).max(6);
            match kind {
                0 | 1 => {
                    let rows = ((h as f32 / 13.) * p.occupancy).ceil() as i32;
                    for row in 0..rows {
                        let end = x + (w as f32 * rng.random_range(0.45..1.)) as i32;
                        writing(c, &mut rng, x + 3, y + row * 13, end, ink, 2);
                        if kind == 1 && row % 3 == 1 {
                            c.line(
                                [x as f32, y as f32 + row as f32 * 13. + 8.],
                                [(end - 4) as f32, y as f32 + row as f32 * 13. + 8.],
                                1,
                                INKS[(i + 1) % 4],
                            );
                        }
                    }
                }
                2 => {
                    let n = rng.random_range(3..=6);
                    let pts: Vec<_> = (0..n)
                        .map(|_| {
                            [
                                x as f32 + w as f32 * rng.random_range(0.12..0.88),
                                y as f32 + h as f32 * rng.random_range(0.12..0.88),
                            ]
                        })
                        .collect();
                    for i in 1..pts.len() {
                        let a = pts[rng.random_range(0..i)];
                        let b = pts[i];
                        c.line(a, b, 1, ink);
                        let d = Vec2::from_array(b) - Vec2::from_array(a);
                        let t = d.normalize_or_zero();
                        let tip = Vec2::from_array(b) - t * 5.;
                        c.line(b, (tip + Vec2::new(-t.y, t.x) * 2.).to_array(), 1, ink);
                    }
                    for pt in pts {
                        c.ring(pt, [w as f32 * 0.08 + 3., h as f32 * 0.07 + 3.], ink);
                    }
                }
                3 => {
                    c.line(
                        [x as f32 + 4., y as f32],
                        [x as f32 + 4., (y + h - 3) as f32],
                        1,
                        ink,
                    );
                    c.line(
                        [x as f32 + 4., (y + h - 3) as f32],
                        [(x + w) as f32, (y + h - 3) as f32],
                        1,
                        ink,
                    );
                    let n = rng.random_range(3..=8);
                    let mut last = [x as f32 + 5., (y + h / 2) as f32];
                    for j in 1..=n {
                        let next = [
                            x as f32 + 5. + (w - 7) as f32 * j as f32 / n as f32,
                            y as f32 + h as f32 * rng.random_range(0.08..0.92),
                        ];
                        c.line(last, next, 1, INKS[(i + 1) % 4]);
                        last = next;
                    }
                }
                4 => {
                    let cols = rng.random_range(2..=4);
                    let rows = (h / 20).max(1);
                    for col in 0..=cols {
                        let xx = x + col * w / cols;
                        c.line([xx as f32, y as f32], [xx as f32, (y + h) as f32], 1, ink);
                    }
                    for row in 0..=rows {
                        let yy = y + row * h / rows;
                        c.line([x as f32, yy as f32], [(x + w) as f32, yy as f32], 1, ink);
                    }
                    for row in 0..rows {
                        for col in 0..cols {
                            if rng.random_bool(p.occupancy as f64) {
                                writing(
                                    c,
                                    &mut rng,
                                    x + col * w / cols + 3,
                                    y + row * h / rows + 4,
                                    x + (col + 1) * w / cols - 3,
                                    ink,
                                    1,
                                );
                            }
                        }
                    }
                }
                5 => {
                    let rows = (h / 17).max(1);
                    for row in 0..rows {
                        let yy = y + row * 17;
                        // Equation structure and coefficients vary independently.
                        let text = content::WORDS[rng.random_range(0..content::WORDS.len())];
                        handwriting::text(
                            c,
                            &mut rng,
                            &text[..text.len().min(3)],
                            [x as f32 + 2., yy as f32],
                            7.,
                            ink,
                        );
                        let xx = x + (w / 3).max(24);
                        c.line(
                            [xx as f32, yy as f32 + 3.],
                            [(xx + 6) as f32, yy as f32 + 3.],
                            1,
                            ink,
                        );
                        c.line(
                            [xx as f32, yy as f32 + 6.],
                            [(xx + 6) as f32, yy as f32 + 6.],
                            1,
                            ink,
                        );
                        c.digit(rng.random_range(1..10), xx + 10, yy, 1, ink);
                        c.line(
                            [(xx + 21) as f32, yy as f32 + 4.],
                            [(xx + 27) as f32, yy as f32 + 4.],
                            1,
                            ink,
                        );
                        c.line(
                            [(xx + 24) as f32, yy as f32 + 1.],
                            [(xx + 24) as f32, yy as f32 + 7.],
                            1,
                            ink,
                        );
                        c.digit(rng.random_range(0..10), xx + 32, yy, 1, ink);
                    }
                }
                _ => {
                    let n = rng.random_range(2..=5);
                    for _ in 0..n {
                        let xx = x as f32 + w as f32 * rng.random_range(0.02..0.65);
                        let yy = y as f32 + h as f32 * rng.random_range(0.02..0.65);
                        box_outline(
                            c,
                            xx,
                            yy,
                            w as f32 * rng.random_range(0.15..0.35),
                            h as f32 * rng.random_range(0.18..0.35),
                            1,
                            ink,
                        );
                    }
                }
            }
        });
    }
    c.0
}
