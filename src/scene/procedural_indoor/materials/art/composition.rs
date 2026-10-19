//! Compositional artwork: connected subjects and layered marks in continuous regions.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ArtComposition {
    pub regions: super::super::composition::Composition,
    pub palette: [[u8; 3]; 5],
    pub focal: [f32; 2],
    pub rotation: f32,
    pub shape_count: u32,
    pub negative_space: f32,
    pub curvature: f32,
    pub asymmetry: f32,
}
impl ArtComposition {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x415254434f4d50);
        let hue = rng.random_range(0.0..360.);
        let saturation = rng.random_range(0.03..0.75);
        let dark = rng.random_bool(0.28);
        let spread = rng.random_range(12.0..155.);
        let palette = std::array::from_fn(|i| {
            let h = hue
                + rng.random_range(-spread..spread)
                + if i == 3 {
                    rng.random_range(90.0..210.)
                } else {
                    0.
                };
            let c = Color::hsl(
                h,
                saturation * if i == 0 { 0.55 } else { 1. },
                rng.random_range(if i == 0 {
                    if dark {
                        0.06..0.23
                    } else {
                        0.70..0.97
                    }
                } else if i == 1 {
                    if dark {
                        0.72..0.95
                    } else {
                        0.05..0.30
                    }
                } else {
                    0.28..0.70
                }),
            )
            .to_srgba();
            [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8)
        });
        let count = if rng.random_bool(0.78) {
            1
        } else {
            rng.random_range(2..=4)
        };
        let margin = rng.random_range(0.0..0.11);
        Self {
            regions: if count > 1 && rng.random_bool(0.5) {
                super::super::composition::Composition::scattered(
                    seed,
                    count,
                    [margin, margin, 1. - 2. * margin, 1. - 2. * margin],
                )
            } else {
                super::super::composition::Composition::sample(
                    seed,
                    count,
                    [margin, margin, 1. - 2. * margin, 1. - 2. * margin],
                    0.55,
                )
            },
            palette,
            focal: [rng.random_range(0.12..0.88), rng.random_range(0.12..0.88)],
            rotation: rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
            shape_count: rng.random_range(3..=24),
            negative_space: rng.random_range(0.04..0.77),
            curvature: rng.random_range(0.02..0.95),
            asymmetry: rng.random_range(-0.35..0.35),
        }
    }
}

fn bezier(c: &mut Canvas, points: [[f32; 2]; 3], width: i32, color: [u8; 3]) {
    let at = |t: f32| {
        [0, 1].map(|i| {
            (1. - t).powi(2) * points[0][i]
                + 2. * t * (1. - t) * points[1][i]
                + t * t * points[2][i]
        })
    };
    for i in 0..24 {
        c.line(at(i as f32 / 24.), at((i + 1) as f32 / 24.), width, color);
    }
}

pub(super) fn paint(p: &ArtProgram) -> Canvas {
    let art = &p.composition;
    let mut canvas = Canvas::new(art.palette[0]);
    for region in &art.regions.regions {
        let mut rng = stream(region.seed, 0x4152545355424a);
        let [x, y, w, h] = region.pixels();
        let [fx, fy] = art.focal;
        let point = |u: f32, v: f32| [x as f32 + u * w as f32, y as f32 + v * h as f32];
        let palette = art.palette;
        let subject = (p.subject + rng.random_range(0..3)) % 10;
        canvas.clipped([x, y, w, h], |c| match subject {
            0 => {
                // Cropped terrain/sea silhouettes with independently sampled control points.
                c.rect(x, y, w, h, palette[4]);
                let horizon = p.horizon;
                for layer in 0..rng.random_range(3..=7) {
                    let mut points = vec![point(-0.05, 1.1)];
                    let mut level = horizon + layer as f32 * 0.09;
                    for i in 0..=14 {
                        level = (level + rng.random_range(-p.relief..p.relief)).clamp(0.12, 0.97);
                        points.push(point(i as f32 / 14., level));
                    }
                    points.push(point(1.05, 1.1));
                    c.polygon(&points, palette[1 + layer % 4]);
                }
                if rng.random_bool(0.65) {
                    let radius = w.min(h) as f32 * rng.random_range(0.025..0.12);
                    c.ellipse(point(p.sun[0], p.sun[1]), [radius; 2], palette[0]);
                }
            }
            1 => {
                // Still life: a shared supporting plane, vessels and overlapping fruit.
                let level = rng.random_range(0.64..0.85);
                c.rect(x, (y as f32 + level * h as f32) as i32, w, h, palette[2]);
                let n = (art.shape_count / 3).clamp(2, 6);
                for i in 0..n {
                    let u = (i as f32 + 0.6) / n as f32 + rng.random_range(-0.06..0.06);
                    let v = level + rng.random_range(-0.035..0.04);
                    let radius = rng.random_range(0.035..0.085);
                    if rng.random_bool(0.55) {
                        let height = rng.random_range(0.15..0.48);
                        let neck = radius * rng.random_range(0.25..0.65);
                        let shape = [
                            point(u - radius, v),
                            point(u - radius * 1.1, v - height * 0.55),
                            point(u - neck, v - height * 0.85),
                            point(u - neck, v - height),
                            point(u + neck, v - height),
                            point(u + neck, v - height * 0.85),
                            point(u + radius * 1.1, v - height * 0.55),
                            point(u + radius, v),
                        ];
                        c.polygon(&shape, palette[1 + i as usize % 4]);
                        c.ellipse(point(u, v - height), [neck * w as f32, 2.], palette[1]);
                    } else {
                        c.ellipse(
                            point(u, v - radius * 0.5),
                            [radius * w as f32, radius * h as f32 * 1.1],
                            palette[1 + i as usize % 4],
                        );
                        c.line(
                            point(u, v - radius * 1.3),
                            point(u + 0.012, v - radius * 1.7),
                            1,
                            palette[1],
                        );
                    }
                }
            }
            2 => {
                // Botanical branching with curved stems and attached leaf silhouettes.
                let count = (art.shape_count / 3).clamp(1, 6);
                for i in 0..count {
                    let root = point(fx + art.asymmetry * (i as f32 / count as f32 - 0.5), 0.95);
                    let tip = point(rng.random_range(0.1..0.9), rng.random_range(0.03..0.3));
                    let control = point(rng.random_range(0.12..0.88), 0.5);
                    bezier(c, [root, control, tip], 1, palette[1]);
                    for j in 1..rng.random_range(5..=9) {
                        let t = j as f32 / 10.;
                        let at = [0, 1].map(|k| {
                            (1. - t).powi(2) * root[k]
                                + 2. * t * (1. - t) * control[k]
                                + t * t * tip[k]
                        });
                        let side = if j % 2 == 0 { 1. } else { -1. };
                        let dx = side * w as f32 * rng.random_range(0.07..0.18);
                        let dy = -h as f32 * rng.random_range(0.035..0.14);
                        c.polygon(
                            &[
                                at,
                                [at[0] + dx * 0.5, at[1] + dy - h as f32 * 0.025],
                                [at[0] + dx, at[1] + dy],
                                [at[0] + dx * 0.6, at[1] + h as f32 * 0.02],
                            ],
                            palette[1 + j as usize % 4],
                        );
                        c.line(at, [at[0] + dx, at[1] + dy], 1, palette[0]);
                    }
                    if rng.random_bool(0.55) {
                        let r = w.min(h) as f32 * rng.random_range(0.025..0.055);
                        let petals = rng.random_range(4..=8);
                        for petal in 0..petals {
                            let a = petal as f32 * std::f32::consts::TAU / petals as f32;
                            c.ellipse(
                                [tip[0] + a.cos() * r, tip[1] + a.sin() * r],
                                [r * 0.7; 2],
                                palette[3],
                            );
                        }
                        c.ellipse(tip, [r * 0.45; 2], palette[1]);
                    }
                }
            }
            3 => {
                // Perspective architecture with a common vanishing point and varied bays.
                let vanishing = point(fx, fy * 0.65);
                for i in 0..art.shape_count.min(12) {
                    let left = rng.random_range(-0.1..0.95);
                    let width = rng.random_range(0.07..0.24);
                    let top = rng.random_range(0.1..0.7);
                    let a = point(left, 0.95);
                    let b = point(left + width, 0.95);
                    let d = point(left, top);
                    let e = point(left + width, top);
                    c.polygon(&[a, b, e, d], palette[1 + i as usize % 4]);
                    let rear = [
                        e[0] + (vanishing[0] - e[0]) * 0.18,
                        e[1] + (vanishing[1] - e[1]) * 0.18,
                    ];
                    c.polygon(&[d, e, rear, [d[0] + rear[0] - e[0], rear[1]]], palette[0]);
                    for row in 1..rng.random_range(3..=7) {
                        let v = top + (0.95 - top) * row as f32 / 7.;
                        c.line(
                            point(left + 0.015, v),
                            point(left + width - 0.015, v),
                            1,
                            palette[0],
                        );
                    }
                }
            }
            7 => {
                // Contour/ink study: related curves with independently drifting
                // control points, not repeated equally spaced sinusoidal stripes.
                let mut elevation = rng.random_range(0.02..0.18);
                for line in 0..art.shape_count {
                    let left = point(-0.05, elevation);
                    let right = point(1.05, elevation + art.asymmetry);
                    let control = point(fx, elevation + rng.random_range(-0.20..0.20));
                    bezier(
                        c,
                        [left, control, right],
                        rng.random_range(1..=2),
                        palette[1 + line as usize % 4],
                    );
                    elevation += rng.random_range(0.025..0.09);
                }
            }
            8 => {
                // Exhibition/event poster: image field, type hierarchy and
                // negative space are sampled together rather than a fixed title strip.
                let margin = rng.random_range(0.04..0.15);
                let split = rng.random_range(0.28..0.70);
                let image_top = rng.random_bool(0.5);
                let (iy, ih, ty) = if image_top {
                    (margin, split - margin, split + margin)
                } else {
                    (split, 1. - split - margin, margin)
                };
                c.polygon(
                    &[
                        point(margin, iy),
                        point(1. - margin, iy),
                        point(1. - margin, iy + ih),
                        point(margin, iy + ih),
                    ],
                    palette[2],
                );
                c.ellipse(
                    point(fx.clamp(margin + 0.1, 0.9 - margin), iy + ih * 0.5),
                    [w as f32 * 0.11, h as f32 * ih * 0.35],
                    palette[3],
                );
                let title =
                    content::phrase(&mut rng, ((w as f32 * (1. - 2. * margin)) / 8.) as usize);
                let pos = point(margin, ty);
                c.text(&title, pos[0] as i32, pos[1] as i32, 2, palette[1]);
                let subtitle = content::phrase(&mut rng, (w / 5) as usize);
                c.text(&subtitle, pos[0] as i32, pos[1] as i32 + 18, 1, palette[1]);
            }
            9 => {
                // Overlapping paper-cut silhouettes with varied contour vertices.
                for i in 0..art.shape_count {
                    let cx = rng.random_range(0.05..0.95);
                    let cy = rng.random_range(0.05..0.95);
                    let radius = rng.random_range(0.05..0.24);
                    let n = rng.random_range(5..=9);
                    let points: Vec<_> = (0..n)
                        .map(|j| {
                            let theta = art.rotation + j as f32 / n as f32 * std::f32::consts::TAU;
                            let r = radius * rng.random_range(0.65..1.);
                            point(cx + theta.cos() * r, cy + theta.sin() * r)
                        })
                        .collect();
                    c.polygon(&points, palette[1 + i as usize % 4]);
                }
            }
            _ => {
                // Abstract line/shape compositions share a focal region, palette and mark grammar.
                let (sin, cos) = art.rotation.sin_cos();
                let transform =
                    |a: f32, b: f32| point(fx + a * cos - b * sin, fy + a * sin + b * cos);
                for i in 0..art.shape_count {
                    let extent = (1. - art.negative_space) * 0.65;
                    let a = rng.random_range(-extent..extent);
                    let b = rng.random_range(-extent..extent);
                    let width = rng.random_range(0.04..0.32);
                    let height = rng.random_range(0.04..0.37);
                    let ink = palette[1 + i as usize % 4];
                    if subject == 4 {
                        c.polygon(
                            &[
                                transform(a, b),
                                transform(a + width, b + art.asymmetry * height),
                                transform(a + width * (1. - art.curvature * 0.5), b + height),
                                transform(a - width * 0.15, b + height * 0.8),
                            ],
                            ink,
                        );
                    } else if subject == 5 {
                        c.ellipse(
                            transform(a, b),
                            [width * w as f32 * 0.5, height * h as f32 * 0.5],
                            ink,
                        );
                    } else {
                        bezier(
                            c,
                            [
                                transform(a, b),
                                transform(a + width * art.curvature, b - height),
                                transform(a + width, b + height),
                            ],
                            rng.random_range(1..=3),
                            ink,
                        );
                    }
                }
            }
        });
    }
    canvas
}
