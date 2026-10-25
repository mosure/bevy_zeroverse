//! Marker sketches with geometric relationships and independently sampled
//! annotations. Scales are relative to their region, not fixed board landmarks.
use super::*;

pub(super) fn paint(c: &mut Canvas, seed: u64, bounds: [i32; 4], kind: u32, p: &BoardProgram) {
    let mut rng = stream(seed, 0x534b45544348);
    let [x, y, w, h] = bounds;
    let at = |u: f32, v: f32| [x as f32 + u * w as f32, y as f32 + v * h as f32];
    let ink = p.inks[p.ink_index as usize];
    let accent = p.inks[(p.ink_index as usize + 1) % 4];
    match kind {
        7 => {
            // A plan with shared partitions, circulation gaps and annotations.
            let plan = super::super::composition::Composition::editorial(
                seed,
                rng.random_range(2..=8),
                [0.03, 0.04, 0.94, 0.91],
                rng.random_range(0.2..0.8),
            );
            for r in plan.regions {
                let [a, b, ww, hh] = r.bounds;
                let pos = at(a, b);
                box_outline(c, pos[0], pos[1], ww * w as f32, hh * h as f32, 1, ink);
                if ww * w as f32 > 24. && hh * h as f32 > 15. {
                    let text =
                        content::phrase(&mut rng, (ww * w as f32 / p.letter_height) as usize);
                    handwriting::text(
                        c,
                        &mut rng,
                        &text,
                        [pos[0] + 3., pos[1] + 4.],
                        p.letter_height * 0.8,
                        accent,
                    );
                }
            }
        }
        8 => {
            // Perspective product sketch with common vanishing direction.
            let a = at(rng.random_range(0.06..0.22), rng.random_range(0.35..0.53));
            let b = at(rng.random_range(0.60..0.85), rng.random_range(0.35..0.53));
            let hh = h as f32 * rng.random_range(0.25..0.46);
            let offset = [
                w as f32 * rng.random_range(0.06..0.18),
                -h as f32 * rng.random_range(0.12..0.25),
            ];
            let corners = [a, b, [b[0], b[1] + hh], [a[0], a[1] + hh]];
            for i in 0..4 {
                c.line(corners[i], corners[(i + 1) % 4], 1, ink);
            }
            for i in [0, 1, 2] {
                c.line(
                    corners[i],
                    [corners[i][0] + offset[0], corners[i][1] + offset[1]],
                    1,
                    ink,
                );
            }
            c.line(
                [a[0] + offset[0], a[1] + offset[1]],
                [b[0] + offset[0], b[1] + offset[1]],
                1,
                ink,
            );
            c.line(
                [b[0] + offset[0], b[1] + offset[1]],
                [b[0] + offset[0], b[1] + hh + offset[1]],
                1,
                ink,
            );
            let label = content::phrase(&mut rng, (w as f32 / p.letter_height) as usize);
            handwriting::text(c, &mut rng, &label, at(0.06, 0.04), p.letter_height, accent);
        }
        9 => {
            // Curves from a shared polynomial model and perturbed measurements.
            let slope: f32 = rng.random_range(-0.65..0.65);
            let curvature = rng.random_range(-1.4..1.4);
            c.line(at(0.07, 0.9), at(0.98, 0.9), 1, ink);
            c.line(at(0.07, 0.9), at(0.07, 0.04), 1, ink);
            let f = |u: f32| {
                (0.5 + slope * (u - 0.5) + curvature * (u - 0.5).powi(2)).clamp(0.05, 0.85)
            };
            for i in 0..32 {
                let u = i as f32 / 32.;
                let v = (i + 1) as f32 / 32.;
                c.line(at(u, f(u)), at(v, f(v)), 1, accent);
            }
            for _ in 0..rng.random_range(4..=20) {
                let u = rng.random_range(0.12..0.93);
                let v = f(u) + rng.random_range(-0.06..0.06);
                c.ring(at(u, v), [1.8; 2], ink);
            }
        }
        _ => {
            // Sticky planning notes: sampled ragged clusters and unequal sizes.
            let notes = super::super::composition::Composition::clusters(
                seed,
                rng.random_range(2..=9),
                [0.02, 0.02, 0.96, 0.96],
            );
            for r in notes.regions {
                let [u, v, ww, hh] = r.bounds;
                let [a, b] = at(u, v);
                let paper = [
                    rng.random_range(220..250),
                    rng.random_range(210..245),
                    rng.random_range(140..210),
                ];
                let bb = [
                    a as i32,
                    b as i32,
                    (ww * w as f32) as i32,
                    (hh * h as f32) as i32,
                ];
                c.rect(bb[0], bb[1], bb[2], bb[3], paper);
                c.clipped(bb, |c| {
                    let size = (p.letter_height * 0.7).min(hh * h as f32 * 0.28).max(3.);
                    let text = content::phrase(&mut rng, ((ww * w as f32 - 4.) / size) as usize);
                    handwriting::text(c, &mut rng, &text, [a + 2., b + 3.], size, ink);
                });
            }
        }
    }
}
