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
            p.inks[0],
            1,
        );
        erased.ring(
            [rng.random_range(30.0..225.), rng.random_range(30.0..225.)],
            [rng.random_range(8.0..35.), rng.random_range(6.0..25.)],
            p.inks[1],
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
        let ink = p.inks[(p.ink_index as usize + i % p.color_count as usize) % 4];
        let kind = rng.random_range(0..11);
        c.clipped([x, y, w, h], |c| {
            let padding = (rng.random_range(0.01..0.13) * w.min(h) as f32).max(1.) as i32;
            let x = x + padding;
            let w = (w - 2 * padding).max(1);
            let text_height = p.letter_height * rng.random_range(0.85..1.15);
            let mut top = padding;
            if h > 28 && rng.random_bool(p.heading_probability as f64) {
                let height = (text_height * p.heading_scale).min(h as f32 * 0.25);
                let heading = content::phrase(&mut rng, (w as f32 / (height * 0.92)) as usize);
                handwriting::text(
                    c,
                    &mut rng,
                    &heading,
                    [x as f32, (y + top) as f32],
                    height,
                    ink,
                );
                top += (height * rng.random_range(1.2..1.9)) as i32;
                if rng.random_bool(0.4) {
                    c.line(
                        [x as f32, (y + top) as f32],
                        [
                            (x + w) as f32,
                            (y + top) as f32 + rng.random_range(-2.0..2.),
                        ],
                        1,
                        ink,
                    );
                    top += 3;
                }
            }
            let y = y + top;
            let h = (h - top - padding).max(1);
            let spacing = (text_height * p.line_spacing).ceil() as i32;
            match kind {
                7..=10 => sketches::paint(c, r.seed, [x, y, w, h], kind, p),
                0 | 1 => {
                    let rows = ((h as f32 / spacing as f32) * p.occupancy).ceil() as i32;
                    for row in 0..rows {
                        let end = x + (w as f32 * rng.random_range(0.45..1.)) as i32;
                        let text = content::phrase(
                            &mut rng,
                            ((end - x) as f32 / (text_height * 0.92)) as usize,
                        );
                        handwriting::text(
                            c,
                            &mut rng,
                            &text,
                            [x as f32, (y + row * spacing) as f32],
                            text_height,
                            ink,
                        );
                        if kind == 1 && row % 3 == 1 {
                            c.line(
                                [
                                    x as f32,
                                    y as f32 + row as f32 * spacing as f32 + text_height + 2.,
                                ],
                                [
                                    (end - 4) as f32,
                                    y as f32 + row as f32 * spacing as f32 + text_height + 2.,
                                ],
                                1,
                                p.inks[(i + 1) % 4],
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
                        c.line(last, next, 1, p.inks[(i + 1) % 4]);
                        last = next;
                    }
                }
                4 => {
                    let cols = rng.random_range(2..=4);
                    let rows = (h / (spacing + 4)).max(1);
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
                    let equation_spacing = spacing.max(12);
                    let rows = (h / equation_spacing).max(1);
                    for row in 0..rows {
                        let terms = rng
                            .random_range(1..=4)
                            .min((w / (text_height * 3.) as i32).max(1));
                        let var = ["X", "Y", "Z", "T"][rng.random_range(0..4)];
                        let lhs = ["F", "G", "A", "B", "Y"][rng.random_range(0..5)];
                        let mut equation = format!("{lhs} = ");
                        for term in 0..terms {
                            if term > 0 {
                                equation.push_str(if rng.random_bool(0.5) { " + " } else { " - " });
                            }
                            equation.push_str(&rng.random_range(1..=9).to_string());
                            if term == 0 || rng.random_bool(0.65) {
                                equation.push_str(var);
                            }
                            if rng.random_bool(0.25) {
                                equation.push_str(&format!("^{}", rng.random_range(2..=4)));
                            }
                        }
                        let hh = text_height.min(w as f32 / (equation.len() as f32 * 0.94));
                        handwriting::text(
                            c,
                            &mut rng,
                            &equation,
                            [x as f32, (y + row * equation_spacing) as f32],
                            hh,
                            ink,
                        );
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
