//! Seeded compositional software and media displays; generated once per palette.
mod chrome;
mod tv;
mod workspace;
use super::{
    raster::{Canvas, N},
    *,
};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
pub use chrome::ChromeProgram;
use rand::Rng;

pub const LAYOUTS: u32 = 12;
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub enum DisplayKind {
    Desktop,
    Phone,
    Television,
}
fn device_seed(seed: u64, kind: DisplayKind) -> u64 {
    match kind {
        DisplayKind::Desktop => seed,
        DisplayKind::Phone => seed ^ 0x50484f4e45,
        DisplayKind::Television => seed ^ 0x54454c4556495349,
    }
}
pub fn parameters_for(seed: u64, kind: DisplayKind) -> ScreenProgram {
    device_parameters(device_seed(seed, kind), kind)
}
fn device_parameters(seed: u64, kind: DisplayKind) -> ScreenProgram {
    let mut p = base_parameters(seed);
    let mut rng = stream(seed, 0x4445564943455549);
    let mut output = stream(seed, 0x444953504f555450);
    let (lo, hi): (f32, f32) = if output.random_bool(0.80) {
        (80., 350.)
    } else {
        match kind {
            DisplayKind::Desktop => (8., 550.),
            DisplayKind::Phone => (3., 1100.),
            DisplayKind::Television => (12., 800.),
        }
    };
    p.luminance = output.random_range(lo.ln()..hi.ln()).exp();
    if !matches!(kind, DisplayKind::Desktop) {
        p.full_bleed = rng.random_bool(if matches!(kind, DisplayKind::Phone) {
            0.24
        } else {
            0.48
        });
    }
    chrome::arrange(seed, kind, &mut p);
    p
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ScreenProgram {
    pub composition: composition::Composition,
    /// Single familiar application or an independently composed workspace.
    pub composed: bool,
    pub full_bleed: bool,
    pub chrome: u32,
    pub chrome_geometry: ChromeProgram,
    pub finish: display_finish::DisplayFinish,
    pub layout: u32,
    pub minutes: u32,
    pub dark: bool,
    /// cd/m² of a white pixel. Content texture modulates this emission.
    pub luminance: f32,
    pub white_kelvin: f32,
    pub white_tint: f32,
    pub dormant: bool,
    pub sidebar: f32,
    pub content_split: f32,
    pub density: f32,
    pub panels: u32,
    pub accent_hue: f32,
    pub accent_saturation: f32,
    pub background_tint: f32,
    pub background_lightness: f32,
    pub panel_contrast: f32,
    pub chart_slope: f32,
    pub chart_curvature: f32,
    pub framing: f32,
    pub text_scale: i32,
}
pub fn parameters(seed: u64) -> ScreenProgram {
    device_parameters(seed, DisplayKind::Desktop)
}
pub(super) fn base_parameters(seed: u64) -> ScreenProgram {
    let mut rng = stream(seed, 861);
    let mut layout = stream(seed, 0x444953504c4159);
    let chrome = layout.random_range(0..4);
    let mut palette = stream(seed, 0x50414c45545445);
    let mut output = stream(seed, 0x53435245454e4c54);
    let white_kelvin = if output.random_bool(0.25) {
        output.random_range(2700.0..5800.0)
    } else {
        output.random_range(5600.0..9000.0)
    };
    ScreenProgram {
        composition: composition::Composition {
            gutter: 0.,
            regions: Vec::new(),
        },
        composed: layout.random_bool(0.96),
        full_bleed: stream(seed, 0x46554c4c424c4545).random_bool(0.12),
        chrome,
        chrome_geometry: ChromeProgram::sample(seed),
        finish: display_finish::DisplayFinish::sample(seed, false),
        layout: rng.random_range(0..LAYOUTS),
        minutes: rng.random_range(0..1440),
        dark: rng.random_bool(0.55),
        // Preserve the content RNG stream when changing only output photometry.
        // The device output distribution resolves this field below. Keep this
        // f32 draw here so appearance changes do not reshuffle content/dormancy.
        luminance: rng.random_range(65.0..280.0),
        white_kelvin,
        white_tint: output.random_range(-0.025..0.025),
        dormant: rng.random_bool(0.12),
        sidebar: rng.random_range(0.065..0.25),
        content_split: rng.random_range(0.30..0.74),
        density: rng.random_range(0.16..0.97),
        panels: rng.random_range(1..7),
        accent_hue: rng.random_range(0.0..360.0),
        accent_saturation: palette.random_range(0.25..0.65),
        background_tint: palette.random_range(0.015..0.15),
        background_lightness: palette.random_range(0.04..0.16),
        panel_contrast: palette.random_range(0.035..0.11),
        chart_slope: rng.random_range(-0.45..0.60),
        chart_curvature: rng.random_range(-0.7..0.7),
        framing: rng.random_range(0.58..1.0),
        text_scale: if rng.random_bool(0.25) { 2 } else { 1 },
    }
}
pub(super) fn apply(
    seed: u64,
    mobile: bool,
    mat: &mut StandardMaterial,
    images: &mut impl AssetStore<Image>,
) {
    prepare(
        seed,
        if mobile {
            DisplayKind::Phone
        } else {
            DisplayKind::Desktop
        },
    )
    .apply(mat, images);
}
pub(super) fn apply_tv(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    prepare(seed, DisplayKind::Television).apply(mat, images);
}
pub fn television_pixels(seed: u64) -> Vec<u8> {
    device_pixels(
        device_seed(seed, DisplayKind::Television),
        DisplayKind::Television,
    )
}
pub fn pixels(seed: u64) -> Vec<u8> {
    device_pixels(seed, DisplayKind::Desktop)
}
pub fn mobile_pixels(seed: u64) -> Vec<u8> {
    device_pixels(device_seed(seed, DisplayKind::Phone), DisplayKind::Phone)
}
fn device_pixels(seed: u64, kind: DisplayKind) -> Vec<u8> {
    let p = device_parameters(seed, kind);
    if p.dormant {
        return dormant_canvas(seed).0;
    }
    if p.full_bleed {
        return art::illustration(seed ^ 0x4d45444941);
    }
    if p.composed {
        return workspace::pixels(seed, kind, &p);
    }
    match kind {
        DisplayKind::Desktop => single_pixels(seed),
        DisplayKind::Phone => single_mobile_pixels(seed),
        DisplayKind::Television => tv::pixels(seed),
    }
}
pub(super) struct PreparedScreen {
    finish: display_finish::DisplayFinish,
    roughness: Option<Image>,
    color: Image,
    dormant: bool,
    emission: LinearRgba,
}

pub(super) fn prepare(seed: u64, kind: DisplayKind) -> PreparedScreen {
    let seed = device_seed(seed, kind);
    let pixels = device_pixels(seed, kind);
    let p = device_parameters(seed, kind);
    let roughness = p.finish.prepare(seed, false, &pixels);
    let mut screen = mip_image(pixels, 256, MapType::Color);
    screen.sampler = ImageSampler::linear();
    PreparedScreen {
        finish: p.finish,
        roughness,
        color: screen,
        dormant: p.dormant,
        emission: super::super::photometry::display_emission(
            if p.dormant { 0. } else { p.luminance },
            p.white_kelvin,
            p.white_tint,
        ),
    }
}

impl PreparedScreen {
    pub(super) fn apply(self, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
        self.finish.apply_prepared(self.roughness, mat, images);
        let texture = images.add(self.color);
        mat.base_color = Color::WHITE;
        mat.base_color_texture = Some(texture.clone());
        mat.emissive_texture = if self.dormant { None } else { Some(texture) };
        mat.emissive = self.emission;
        mat.emissive_exposure_weight = 1.;
        mat.metallic = 0.;
        mat.uv_transform = bevy::math::Affine2::IDENTITY;
    }
}
pub const SEGMENTS: [u8; 10] = [
    0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111,
    0b1111111, 0b1101111,
];
#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub(super) bg: [u8; 3],
    pub(super) panel: [u8; 3],
    pub(super) ink: [u8; 3],
    pub(super) muted: [u8; 3],
    pub(super) accent: [u8; 3],
}
pub(super) fn palette(p: &ScreenProgram) -> Palette {
    let rgb = |h, s, l| {
        let c = Color::hsl(h, s, l).to_srgba();
        [c.red, c.green, c.blue].map(|v| (v * 255.).round() as u8)
    };
    let base = if p.dark {
        p.background_lightness
    } else {
        0.94 - p.background_lightness
    };
    Palette {
        bg: rgb(p.accent_hue, p.background_tint, base),
        panel: rgb(
            p.accent_hue,
            p.background_tint * 0.4,
            (base + p.panel_contrast).min(0.98),
        ),
        ink: rgb(
            p.accent_hue,
            p.background_tint * 0.7,
            if p.dark {
                0.81 + p.background_lightness * 0.8
            } else {
                0.08 + p.background_lightness
            },
        ),
        muted: rgb(
            p.accent_hue,
            p.background_tint,
            if p.dark {
                0.32 + p.panel_contrast
            } else {
                0.51 + p.panel_contrast
            },
        ),
        accent: rgb(
            p.accent_hue,
            p.accent_saturation,
            if p.dark { 0.65 } else { 0.41 },
        ),
    }
}
fn dormant_canvas(seed: u64) -> Canvas {
    // An unlit panel retains a dark, low-frequency coating tint. It is still
    // dormant (no emissive map); don't invent active content just to be unique.
    let mut rng = stream(seed, 0x434f4154494e47);
    let tint: [f32; 3] = std::array::from_fn(|_| rng.random_range(2.0..7.));
    let angle: f32 = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
    let center = rng.random_range(-0.35..0.35);
    let width = rng.random_range(0.25..0.85);
    let strength = rng.random_range(1.0..7.);
    let (sn, cs) = angle.sin_cos();
    let mut c = Canvas::new([4, 5, 7]);
    for y in 0..N {
        for x in 0..N {
            let d = ((x as f32 / 256. - 0.5) * cs + (y as f32 / 256. - 0.5) * sn - center) / width;
            let sheen = strength * (1. - d * d).max(0.).powi(2);
            c.rect(x, y, 1, 1, tint.map(|t| (t + sheen) as u8));
        }
    }
    c
}
fn text_rows(
    c: &mut Canvas,
    rng: &mut impl Rng,
    bounds: [i32; 4],
    color: [u8; 3],
    density: f32,
    indent: bool,
) {
    let [x, y, w, h] = bounds;
    let spacing = if density > 0.65 { 6 } else { 9 };
    for row in 0..(h / spacing).max(1) {
        if rng.random_bool((density * 0.4 + 0.59) as f64) {
            let dx = if indent {
                rng.random_range(0..4) * 6
            } else {
                0
            };
            let available = (w - dx).max(2);
            let length = rng.random_range((available / 4).max(1)..=available);
            c.rect(x + dx, y + row * spacing, length, 2, color);
        }
    }
}
pub(super) fn chart(
    c: &mut Canvas,
    seed: u64,
    p: &ScreenProgram,
    bounds: [i32; 4],
    colors: Palette,
    mode: u32,
) {
    let [x, y, w, h] = bounds;
    let mut rng = stream(seed, 864);
    for i in 1..4 {
        c.line(
            [x as f32, (y + h * i / 4) as f32],
            [(x + w) as f32, (y + h * i / 4) as f32],
            1,
            colors.muted,
        );
    }
    if mode.is_multiple_of(3) {
        let count = 3 + (p.density * 11.) as i32;
        let width = (w / count).max(3);
        for i in 0..count {
            let height = rng.random_range(0.15..0.93) * h as f32;
            c.rect(
                x + i * width + 1,
                y + h - height as i32,
                (width - 3).max(1),
                height as i32,
                colors.accent,
            );
        }
    } else {
        for series in 0..(1 + p.panels.min(3)) {
            let phase = rng.random_range(0.0..6.);
            let point = |i: i32| {
                let t = i as f32 / w.max(1) as f32;
                let value = 0.5
                    + p.chart_slope * (t - 0.5)
                    + p.chart_curvature * (t * (1. - t) - 0.125)
                    + (t * 9. + phase).sin() * 0.09
                    + series as f32 * 0.065;
                [
                    (x + i) as f32,
                    y as f32 + (1. - value.clamp(0.03, 0.97)) * h as f32,
                ]
            };
            let ink = if series == 0 {
                colors.accent
            } else {
                colors.ink
            };
            for i in 0..w {
                c.line(point(i), point(i + 1), if series == 0 { 2 } else { 1 }, ink);
            }
        }
    }
}
pub(super) fn participants(
    c: &mut Canvas,
    seed: u64,
    p: &ScreenProgram,
    bounds: [i32; 4],
    colors: Palette,
) {
    let mut rng = stream(seed, 865);
    let [x, y, w, h] = bounds;
    // Keep the grid bound explicit. The conditional form compiled without its
    // upper cap in an optimized native build after unrelated code changes.
    // More than three participants must wrap, preserving seeded screen pixels.
    let cols = p.panels.clamp(1, 3) as i32;
    let rows = p.panels.div_ceil(cols as u32) as i32;
    for i in 0..p.panels as i32 {
        let sx = x + i % cols * w / cols;
        let sy = y + i / cols * h / rows;
        let tw = w / cols - 3;
        let th = h / rows - 3;
        c.rect(
            sx,
            sy,
            tw,
            th,
            [
                rng.random_range(52..108),
                rng.random_range(59..111),
                rng.random_range(65..116),
            ],
        );
        let cx = sx as f32 + tw as f32 * rng.random_range(0.35..0.65);
        let skin = [
            rng.random_range(107..221),
            rng.random_range(78..174),
            rng.random_range(58..152),
        ];
        let radius = tw.min(th) as f32 * rng.random_range(0.14..0.20);
        c.ellipse_clipped(
            [cx, sy as f32 + th as f32 * 0.38],
            [radius, radius * 1.25],
            skin,
            [sx, sy, tw, th],
        );
        c.ellipse_clipped(
            [cx, sy as f32 + th as f32 * 0.91],
            [radius * 2.1, radius * 2.3],
            colors.accent,
            [sx, sy, tw, th],
        );
        c.rect(sx + 3, sy + th - 6, tw * 2 / 3, 2, colors.ink);
    }
}
pub(super) fn map(c: &mut Canvas, seed: u64, p: &ScreenProgram, bounds: [i32; 4], colors: Palette) {
    let mut rng = stream(seed, 866);
    let [x, y, w, h] = bounds;
    if w <= 0 || h <= 0 {
        return;
    }
    c.rect(
        x,
        y,
        w,
        h,
        if p.dark {
            [42, 54, 48]
        } else {
            [212, 222, 209]
        },
    );
    let step = rng.random_range(18..37);
    for i in -4..16 {
        let fx = x + i * step;
        c.line(
            [fx as f32, y as f32],
            [(fx + h / 3) as f32, (y + h) as f32],
            3,
            colors.panel,
        );
    }
    for i in 0..(h / step) {
        c.line(
            [x as f32, (y + i * step) as f32],
            [(x + w) as f32, (y + i * step + w / 9) as f32],
            3,
            colors.panel,
        );
    }
    for _ in 0..(p.density * 9.).ceil() as usize {
        // Composed map panels can be smaller than the usual 12-pixel padding.
        // Keep a nonempty interval while preserving larger panels byte-for-byte.
        let inset_x = 6.min((w - 1) / 2);
        let inset_y = 6.min((h - 1) / 2);
        let point = [
            rng.random_range((x + inset_x) as f32..(x + w - inset_x) as f32),
            rng.random_range((y + inset_y) as f32..(y + h - inset_y) as f32),
        ];
        c.ellipse(point, [3., 4.], colors.accent);
    }
}
fn single_pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    if p.dormant {
        return dormant_canvas(seed).0;
    }
    let colors = palette(&p);
    let mut rng = stream(seed, 862);
    let mut c = Canvas::new(colors.bg);
    let sidebar = (p.sidebar * 256.) as i32;
    let toolbar = 13 + p.text_scale * 5;
    c.rect(0, 0, N, toolbar, colors.panel);
    c.text(
        [
            "ANALYTICS",
            "DOCUMENT",
            "WORKSPACE",
            "CALENDAR",
            "MEETING",
            "GALLERY",
            "WELCOME",
            "BOARD",
            "MODEL",
            "MAP",
            "SCHEDULE",
            "MEDIA",
        ][p.layout as usize],
        9,
        7,
        1,
        colors.ink,
    );
    c.rect(0, toolbar + 2, sidebar, 227 - toolbar, colors.panel);
    for row in 0..(3. + p.density * 9.) as i32 {
        c.rect(
            7,
            toolbar + 12 + row * 15,
            (sidebar - 14).max(3),
            3,
            if row == p.layout as i32 % 4 {
                colors.accent
            } else {
                colors.muted
            },
        );
    }
    let x = sidebar + 9;
    let y = toolbar + 10;
    let w = 247 - x;
    let h = 226 - y;
    c.rect(x, y, w, h, colors.panel);
    match p.layout {
        0 => {
            let split = (h as f32 * p.content_split) as i32;
            c.text("REVENUE", x + 7, y + 8, 1, colors.ink);
            chart(
                &mut c,
                seed,
                &p,
                [x + 8, y + 24, w - 16, split - 24],
                colors,
                p.panels,
            );
            for col in 0..3 {
                let cx = x + 6 + col * (w - 12) / 3;
                c.rect(cx, y + split + 10, (w - 20) / 3, 17, colors.bg);
                c.digit(
                    rng.random_range(0..10),
                    cx + 4,
                    y + split + 13,
                    1,
                    colors.accent,
                );
            }
            text_rows(
                &mut c,
                &mut rng,
                [x + 8, y + split + 38, w - 16, h - split - 45],
                colors.muted,
                p.density,
                false,
            );
        }
        1 => {
            c.text("PROJECT NOTES", x + 9, y + 11, p.text_scale, colors.ink);
            let col = if p.content_split > 0.58 { 2 } else { 1 };
            for i in 0..col {
                text_rows(
                    &mut c,
                    &mut rng,
                    [x + 9 + i * w / col, y + 33, w / col - 20, h - 45],
                    colors.ink,
                    p.density,
                    false,
                );
            }
        }
        2 => {
            c.rect(x, y, 15, h, colors.bg);
            for row in 0..(h / 8) {
                c.digit((row % 10) as u32, x + 3, y + row * 8, 1, colors.muted);
            }
            text_rows(
                &mut c,
                &mut rng,
                [x + 23, y + 8, w - 31, h - 18],
                colors.ink,
                p.density,
                true,
            );
            for row in (0..h / 8).step_by(4) {
                c.rect(
                    x + 31,
                    y + row * 8 + 8,
                    rng.random_range(15..(w - 40).max(16)),
                    2,
                    colors.accent,
                );
            }
        }
        3 | 10 => {
            let columns = if p.layout == 3 {
                7
            } else {
                4 + p.panels as i32 / 2
            };
            let rows = if p.layout == 3 {
                5
            } else {
                5 + (p.density * 10.) as i32
            };
            for row in 0..rows {
                for col in 0..columns {
                    let cx = x + col * w / columns;
                    let cy = y + row * h / rows;
                    c.rect(cx, cy, w / columns - 1, h / rows - 1, colors.bg);
                    if rng.random_bool(p.density as f64) {
                        c.rect(
                            cx + 3,
                            cy + h / rows / 2,
                            (w / columns - 6).max(2),
                            4,
                            colors.accent,
                        );
                    }
                    if p.layout == 3 {
                        c.digit(
                            ((row * columns + col + 1) % 10) as u32,
                            cx + 2,
                            cy + 2,
                            1,
                            colors.muted,
                        );
                    }
                }
            }
        }
        4 => participants(&mut c, seed, &p, [x + 5, y + 5, w - 10, h - 21], colors),
        5 | 6 => {
            let art = art::parameters(seed ^ 0x4c414e4453434150);
            c = art::landscape(seed, &art, p.content_split > 0.54);
            if p.layout == 6 {
                c.time(p.minutes, 29, 37, 5, [241, 241, 234]);
                c.text("WELCOME", 79, 98, 2, [241, 241, 234]);
            } else {
                c.rect(0, 0, N, 19, colors.panel);
                c.text("IMAGE LIBRARY", 8, 7, 1, colors.ink);
                c.rect(0, 215, N, 29, colors.panel);
                for i in 0..p.panels {
                    c.rect(10 + i as i32 * 37, 219, 29, 20, colors.accent);
                }
            }
        }
        7 => {
            let columns = 2 + p.panels.min(3) as i32;
            for col in 0..columns {
                let cx = x + 4 + col * w / columns;
                c.text(
                    ["IDEAS", "PLAN", "ACTIVE", "REVIEW", "DONE"][col as usize],
                    cx + 2,
                    y + 9,
                    1,
                    colors.ink,
                );
                for row in 0..(p.density * 6.).ceil() as i32 {
                    if rng.random_bool(0.8) {
                        c.rect(cx, y + 23 + row * 25, w / columns - 8, 20, colors.bg);
                        c.rect(
                            cx + 3,
                            y + 28 + row * 25,
                            (w / columns - 17).max(2),
                            2,
                            colors.muted,
                        );
                        c.rect(cx + 3, y + 39 + row * 25, 6, 2, colors.accent);
                    }
                }
            }
        }
        8 => {
            // CAD viewport with orthographic geometry and construction guides.
            let bx = x as f32 + w as f32 * p.content_split;
            let by = y as f32 + h as f32 * 0.6;
            let ww = w as f32 * 0.49;
            let hh = h as f32 * 0.26;
            let offset = [ww * p.framing * 0.45, -hh * 0.63];
            let front = [
                [bx - ww / 2., by - hh / 2.],
                [bx + ww / 2., by - hh / 2.],
                [bx + ww / 2., by + hh / 2.],
                [bx - ww / 2., by + hh / 2.],
            ];
            for i in 0..4 {
                let next = (i + 1) % 4;
                c.line(front[i], front[next], 2, colors.accent);
                let rear = [front[i][0] + offset[0], front[i][1] + offset[1]];
                let rear_next = [front[next][0] + offset[0], front[next][1] + offset[1]];
                c.line(rear, rear_next, 1, colors.ink);
                c.line(front[i], rear, 1, colors.ink);
            }
            c.text("PERSPECTIVE", x + 6, y + 7, 1, colors.muted);
            c.line(
                [x as f32 + 15., by + hh],
                [x as f32 + 48., by + hh],
                2,
                [174, 72, 66],
            );
        }
        9 => map(&mut c, seed, &p, [x + 5, y + 5, w - 10, h - 10], colors),
        _ => {
            let media_h = (h as f32 * p.framing) as i32;
            chart(
                &mut c,
                seed,
                &p,
                [x + 8, y + 15, w - 16, media_h - 20],
                colors,
                p.panels + 1,
            );
            c.text("TIMELINE", x + 7, y + media_h + 4, 1, colors.ink);
            for i in 0..p.panels {
                c.rect(
                    x + 7,
                    y + media_h + 17 + i as i32 * 8,
                    (w - 13) * rng.random_range(35..100) / 100,
                    4,
                    if i.is_multiple_of(2) {
                        colors.accent
                    } else {
                        colors.muted
                    },
                );
            }
        }
    }
    c.rect(0, 244, N, 12, colors.panel);
    c.time(p.minutes, 219, 246, 1, colors.ink);
    for icon in 0..(3 + p.panels) {
        c.rect(16 + icon as i32 * 18, 248, 9, 5, colors.accent);
    }
    c.0
}
fn single_mobile_pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    if p.dormant {
        return dormant_canvas(seed).0;
    }
    let colors = palette(&p);
    let mut rng = stream(seed, 863);
    let mut c = Canvas::new(colors.bg);
    c.time(p.minutes, 12, 7, 1, colors.ink);
    c.rect(220, 8, 21, 5, colors.ink);
    match p.layout % 6 {
        0 | 1 => {
            c.text(
                if p.layout.is_multiple_of(2) {
                    "MESSAGES"
                } else {
                    "NOTES"
                },
                17,
                28,
                2,
                colors.ink,
            );
            for row in 0..(2. + p.density * 5.) as i32 {
                let x = if row % 2 == 0 { 17 } else { 46 };
                let y = 53 + row * 27;
                c.rect(
                    x,
                    y,
                    178,
                    22,
                    if row % 2 == 0 {
                        colors.panel
                    } else {
                        colors.accent
                    },
                );
                text_rows(
                    &mut c,
                    &mut rng,
                    [x + 6, y + 5, 160, 12],
                    colors.ink,
                    p.density,
                    false,
                );
            }
        }
        2 => {
            c.text("APPLICATIONS", 18, 28, 2, colors.ink);
            for row in 0..4 {
                for col in 0..4 {
                    if rng.random_bool((0.52 + p.density * 0.45) as f64) {
                        let x = 21 + col * 57;
                        let y = 54 + row * 43;
                        c.ellipse(
                            [(x + 15) as f32, (y + 12) as f32],
                            [15., 12.],
                            if (col + row) % 2 == 0 {
                                colors.accent
                            } else {
                                colors.muted
                            },
                        );
                        c.rect(x + 5, y + 8, 20, 6, colors.panel);
                        c.rect(x + 3, y + 28, 24, 2, colors.ink);
                    }
                }
            }
        }
        3 => map(&mut c, seed, &p, [10, 30, 236, 207], colors),
        4 => {
            c.text("ACTIVITY", 17, 31, 2, colors.ink);
            chart(&mut c, seed, &p, [25, 70, 204, 121], colors, p.panels);
            c.rect(20, 208, 216, 23, colors.panel);
            c.text("WEEKLY SUMMARY", 28, 216, 1, colors.ink);
        }
        _ => {
            let art = art::parameters(seed ^ 3110);
            c = art::landscape(seed, &art, p.content_split > 0.5);
            c.time(p.minutes, 38, 44, 6, [244; 3]);
            c.rect(18, 145, 220, 38, colors.panel);
            c.text("REMINDER", 29, 155, 1, colors.ink);
            c.rect(29, 170, (p.density * 170.) as i32 + 12, 2, colors.muted);
        }
    }
    c.rect(92, 247, 72, 3, colors.ink);
    c.0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_map_panels_have_nonempty_sampling_ranges_and_repeatable_pixels() {
        let seed = 355;
        let p = parameters(seed);
        for w in -1..=16 {
            for h in -1..=16 {
                let draw = || {
                    let mut c = Canvas::new([1, 2, 3]);
                    map(&mut c, seed, &p, [17, 21, w, h], palette(&p));
                    c.0
                };
                let pixels = draw();
                assert_eq!(pixels, draw(), "map bounds {w} x {h}");
                if w <= 0 || h <= 0 {
                    assert_eq!(pixels, Canvas::new([1, 2, 3]).0);
                }
            }
        }
    }
    #[test]
    fn composed_participant_grid_preserves_three_column_pixel_replay() {
        // Exercise the grid itself so unrelated palette/header/media expansion
        // cannot silently turn this compiler regression into a different test.
        let seed = 4_561_871_313_230_591_933;
        let mut p = parameters(seed);
        p.panels = 4;
        let draw = || {
            let mut c = Canvas::new([1, 2, 3]);
            participants(&mut c, seed, &p, [12, 14, 222, 202], palette(&p));
            c.0
        };
        let pixels = draw();
        assert_eq!(pixels, draw());
        for (x, y, occupied) in [
            (13, 15, true),
            (87, 15, true),
            (161, 15, true),
            (13, 116, true),
            (87, 116, false),
            (84, 15, false),
            (0, 0, false),
        ] {
            let offset = (y * 256 + x) * 4;
            assert_eq!(pixels[offset..offset + 3] != [1, 2, 3], occupied);
        }
    }

    #[test]
    fn screen_emission_agrees_with_dormant_state_and_content_maps() {
        let mut images = Assets::<Image>::default();
        let mut states = std::collections::BTreeSet::new();
        for seed in 0..32 {
            let p = parameters(seed);
            states.insert(p.dormant);
            let mut material = StandardMaterial::default();
            apply(seed, false, &mut material, &mut images);
            assert_eq!(
                material.emissive,
                super::super::super::photometry::display_emission(
                    if p.dormant { 0. } else { p.luminance },
                    p.white_kelvin,
                    p.white_tint
                )
            );
            assert_eq!(material.emissive_texture.is_none(), p.dormant);
            assert_eq!(material.metallic, 0.);
            assert!((0.06..0.4).contains(&material.perceptual_roughness));
            let image = images
                .get(material.base_color_texture.as_ref().unwrap())
                .unwrap();
            assert_eq!(image.texture_descriptor.mip_level_count, 9);
        }
        assert_eq!(states.len(), 2);
    }
    #[test]
    fn family_compositions_and_device_content_are_independent() {
        let mut by_family =
            std::collections::BTreeMap::<u32, std::collections::BTreeSet<Vec<u8>>>::new();
        for seed in 0..128 {
            let p = parameters(seed);
            if p.dormant {
                continue;
            }
            let image = pixels(seed);
            assert_eq!(image.len(), 256 * 256 * 4);
            assert_ne!(image, mobile_pixels(seed));
            assert_ne!(image, television_pixels(seed));
            by_family.entry(p.layout).or_default().insert(image);
        }
        assert_eq!(by_family.len(), LAYOUTS as usize);
        assert!(by_family.values().all(|images| images.len() >= 3));
        assert_eq!(pixels(355), pixels(355));
        assert_eq!(mobile_pixels(355), mobile_pixels(355));
    }
    #[test]
    fn every_device_uses_the_recorded_content_seed_and_finish() {
        for seed in (0..12).chain([u64::MAX - 7]) {
            for kind in [
                DisplayKind::Desktop,
                DisplayKind::Phone,
                DisplayKind::Television,
            ] {
                let p = parameters_for(seed, kind);
                let mut mat = StandardMaterial::default();
                let mut images = Assets::<Image>::default();
                let expected = match kind {
                    DisplayKind::Desktop => {
                        apply(seed, false, &mut mat, &mut images);
                        pixels(seed)
                    }
                    DisplayKind::Phone => {
                        apply(seed, true, &mut mat, &mut images);
                        mobile_pixels(seed)
                    }
                    DisplayKind::Television => {
                        apply_tv(seed, &mut mat, &mut images);
                        television_pixels(seed)
                    }
                };
                let image = images
                    .get(mat.base_color_texture.as_ref().unwrap())
                    .unwrap();
                assert_eq!(&image.data.as_ref().unwrap()[..256 * 256 * 4], &expected);
                assert_eq!(
                    mat.emissive,
                    super::super::super::photometry::display_emission(
                        if p.dormant { 0. } else { p.luminance },
                        p.white_kelvin,
                        p.white_tint
                    )
                );
                assert_eq!(mat.clearcoat, p.finish.clearcoat);
                assert_eq!(mat.emissive_texture.is_none(), p.dormant);
            }
        }
    }
}
