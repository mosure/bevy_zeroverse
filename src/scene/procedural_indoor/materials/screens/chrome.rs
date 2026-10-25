//! Device-aware interface proportions, sampled separately from document content.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct ChromeProgram {
    pub margins: [f32; 4],
    pub toolbar: f32,
    pub sidebar: f32,
    pub row_spacing: f32,
    pub nav_right: bool,
    pub text_scale: i32,
    pub navigation_width: f32,
}
impl ChromeProgram {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x4348524f4d455859);
        Self {
            margins: std::array::from_fn(|_| rng.random_range(0.012..0.075)),
            toolbar: rng.random_range(0.055..0.16),
            sidebar: rng.random_range(0.12..0.31),
            row_spacing: rng.random_range(0.045..0.095),
            nav_right: rng.random_bool(0.24),
            text_scale: if rng.random_bool(0.18) { 2 } else { 1 },
            navigation_width: rng.random_range(0.14..0.46),
        }
    }
}

pub(super) fn arrange(seed: u64, kind: DisplayKind, p: &mut ScreenProgram) {
    let mut rng = stream(seed, 0x444f43554d454e54);
    let g = &p.chrome_geometry;
    let [mut l, mut t, mut r, mut b] = g.margins;
    let count;
    let bias;
    match kind {
        DisplayKind::Desktop => {
            count = rng.random_range(1..=10);
            bias = rng.random_range(0.18..0.82);
            if p.chrome != 0 {
                t += g.toolbar;
            }
            if p.chrome == 2 {
                if g.nav_right {
                    r += g.sidebar;
                } else {
                    l += g.sidebar;
                }
            }
            if p.chrome == 3 {
                b += g.toolbar * 0.6;
            }
        }
        DisplayKind::Phone => {
            p.chrome = u32::from(rng.random_bool(0.90));
            count = rng.random_range(1..=7);
            bias = rng.random_range(0.0..0.12);
            if p.chrome != 0 {
                t += g.toolbar * 0.7;
                b += g.toolbar * 0.35;
            }
        }
        DisplayKind::Television => {
            p.chrome = u32::from(rng.random_bool(0.35));
            count = if rng.random_bool(0.4) {
                1
            } else {
                rng.random_range(2..=6)
            };
            bias = rng.random_range(0.3..0.9);
            if p.chrome != 0 {
                t += g.toolbar;
            }
        }
    }
    p.composition =
        composition::Composition::editorial(seed, count, [l, t, 1. - l - r, 1. - t - b], bias);
}

pub(super) fn draw(c: &mut Canvas, seed: u64, kind: DisplayKind, p: &ScreenProgram) {
    if p.chrome == 0 {
        return;
    }
    let mut rng = stream(seed, 0x4348524f4d45);
    let colors = palette(p);
    let g = &p.chrome_geometry;
    let [l, t, r, b] = g.margins.map(|v| (v * 256.) as i32);
    let top = p
        .composition
        .regions
        .iter()
        .map(|r| (r.bounds[1] * 256.) as i32)
        .min()
        .unwrap_or(t);
    let header = (top - t).max(9);
    c.rect(0, 0, N, header, colors.panel);
    let scale = g.text_scale.min((header / 8).max(1));
    let yy = ((header - 5 * scale) / 2).max(1);
    if matches!(kind, DisplayKind::Phone) {
        c.time(p.minutes, l, yy, 1, colors.ink);
        let bx = 256 - r - 18;
        c.rect(bx, yy, 17, 6, colors.muted);
        c.rect(bx + 1, yy + 1, rng.random_range(2..=15), 4, colors.ink);
        c.rect(bx + 17, yy + 2, 2, 2, colors.muted);
        let width = (g.navigation_width * 256.) as i32;
        c.rect(
            (256 - width) / 2,
            256 - (b / 2).max(3),
            width,
            2,
            colors.muted,
        );
    } else {
        let width = if matches!(kind, DisplayKind::Desktop) {
            180 - l
        } else {
            256 - l - r
        };
        let title = content::phrase(&mut rng, (width / (4 * scale)).max(1) as usize);
        c.clipped([l, 0, width, header], |c| {
            c.text(&title, l, yy, scale, colors.ink)
        });
        if matches!(kind, DisplayKind::Desktop) {
            c.time(p.minutes, 256 - r - 31, yy, 1, colors.muted);
        }
    }
    if p.chrome == 2 && matches!(kind, DisplayKind::Desktop) {
        let width = (g.sidebar * 256.) as i32;
        let x = if g.nav_right { 256 - width } else { 0 };
        c.rect(x, header, width, 256 - header, colors.panel);
        let step = (g.row_spacing * 256.) as i32;
        let active = rng.random_range(0..8);
        for (i, y) in ((top + step / 2)..(256 - b - step))
            .step_by(step as usize)
            .enumerate()
        {
            let title = content::phrase(&mut rng, ((width - 8) / 4).max(1) as usize);
            c.clipped([x + 4, y, width - 8, step - 2], |c| {
                c.text(
                    &title,
                    x + 4,
                    y,
                    1,
                    if i == active {
                        colors.accent
                    } else {
                        colors.muted
                    },
                )
            });
        }
    } else if p.chrome == 3 && matches!(kind, DisplayKind::Desktop) {
        let dock = (g.toolbar * 0.6 * 256.) as i32;
        c.rect(0, 256 - dock, 256, dock, colors.panel);
        let spacing = rng.random_range(10..24);
        let n = rng.random_range(3..=9);
        for i in 0..n {
            c.rect(
                (256 - n * spacing) / 2 + i * spacing,
                256 - dock + 3,
                spacing - 5,
                (dock - 6).max(2),
                if i == 0 { colors.accent } else { colors.muted },
            );
        }
    }
}
