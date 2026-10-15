//! Asset-free marker compositions with independent occupied/erased regions.
mod composition;
mod handwriting;
use super::{raster::Canvas, *};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BoardProgram {
    pub regions: super::composition::Composition,
    pub blank: bool,
    pub composed: bool,
    pub finish: display_finish::DisplayFinish,
    pub layout: u32,
    pub occupancy: f32,
    pub columns: u32,
    pub rows: u32,
    pub color_count: u32,
    pub ink_index: u32,
    pub erasure: f32,
    pub stroke: i32,
    pub margin: f32,
    pub slope: f32,
    pub curvature: f32,
    pub roughness: f32,
    pub warmth: f32,
}
pub fn parameters(seed: u64) -> BoardProgram {
    let mut rng = stream(seed, 3080);
    let mut layout = stream(seed, 0x424f415244);
    let margin = layout.random_range(0.025..0.10);
    let count = if layout.random_bool(0.25) {
        1
    } else {
        layout.random_range(2..=6)
    };
    let finish = display_finish::DisplayFinish::sample(seed, true);
    BoardProgram {
        regions: super::composition::Composition::sample(
            seed,
            count,
            [margin, margin, 1. - 2. * margin, 1. - 2. * margin],
            layout.random_range(0.3..0.7),
        ),
        blank: layout.random_bool(0.045),
        composed: layout.random_bool(0.85),
        roughness: finish.roughness,
        finish,
        layout: rng.random_range(0..10),
        occupancy: rng.random_range(0.14..0.97),
        columns: rng.random_range(2..6),
        rows: rng.random_range(3..9),
        color_count: rng.random_range(1..5),
        ink_index: rng.random_range(0..4),
        erasure: rng.random_range(0.0..0.16),
        stroke: if rng.random_bool(0.28) { 1 } else { 2 },
        margin: rng.random_range(0.04..0.11),
        slope: rng.random_range(-0.5..0.55),
        curvature: rng.random_range(-0.6..0.6),
        warmth: rng.random_range(-0.012..0.028),
    }
}
const INKS: [[u8; 3]; 4] = [[28, 47, 76], [38, 89, 161], [159, 52, 49], [39, 116, 86]];
fn writing(
    c: &mut Canvas,
    rng: &mut impl Rng,
    x: i32,
    y: i32,
    end: i32,
    ink: [u8; 3],
    stroke: i32,
) {
    let height = if stroke > 1 { 7. } else { 5. };
    let text = content::phrase(rng, ((end - x).max(0) as f32 / (height * 0.92)) as usize);
    c.clipped([x - 1, y - 2, (end - x + 2).max(0), 12], |c| {
        handwriting::text(c, rng, &text, [x as f32, y as f32], height, ink)
    });
}
fn box_outline(c: &mut Canvas, x: f32, y: f32, w: f32, h: f32, stroke: i32, ink: [u8; 3]) {
    for (a, b) in [
        ([x, y], [x + w, y]),
        ([x + w, y], [x + w, y + h]),
        ([x + w, y + h], [x, y + h]),
        ([x, y + h], [x, y]),
    ] {
        c.line(a, b, stroke, ink);
    }
}
pub fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    if p.composed || p.blank {
        composition::pixels(seed, &p)
    } else {
        single_pixels(seed)
    }
}
fn single_pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut rng = stream(seed, 3082);
    let base = [246., 246. - p.warmth * 130., 244. - p.warmth * 230.].map(|v| v as u8);
    let mut c = Canvas::new(base);
    let ink = INKS[p.ink_index as usize];
    let mut erased = Canvas::new(base);
    for _ in 0..4 {
        let x = rng.random_range(9.0..240.0);
        let y = rng.random_range(12.0..240.0);
        let radius = [rng.random_range(12.0..44.0), rng.random_range(7.0..30.0)];
        erased.ring([x, y], radius, INKS[rng.random_range(0..4)]);
        for row in 0..3 {
            writing(
                &mut erased,
                &mut rng,
                x as i32 - 20,
                y as i32 + row * 8,
                x as i32 + 20,
                ink,
                1,
            );
        }
    }
    for (i, pixel) in erased.0.as_chunks::<4>().0.iter().enumerate() {
        c.blend(
            i as i32 % 256,
            i as i32 / 256,
            [pixel[0], pixel[1], pixel[2]],
            p.erasure,
        );
    }
    let margin = (p.margin * 256.) as i32;
    let end = 256 - margin;
    let width = end - margin;
    let color = |i: usize| INKS[(p.ink_index as usize + i % p.color_count as usize) % 4];
    c.text(
        [
            "PROCESS",
            "TRENDS",
            "RESULTS",
            "AGENDA",
            "PLAN",
            "IDEAS",
            "CALCULATIONS",
            "NOTES",
            "ROOM PLAN",
            "DESIGN",
        ][p.layout as usize],
        margin + 3,
        margin + 2,
        2,
        ink,
    );
    let top = margin + 25;
    let height = end - top;
    match p.layout {
        0 => {
            let columns = p.columns.min(3) as i32;
            let rows = p.rows.min(5) as i32;
            for row in 0..rows {
                for col in 0..columns {
                    let x = margin + col * width / columns;
                    let y = top + row * height / rows;
                    let w = width / columns - 16;
                    let h = height / rows - 15;
                    box_outline(
                        &mut c,
                        x as f32,
                        y as f32,
                        w as f32,
                        h as f32,
                        p.stroke,
                        color((col + row) as usize),
                    );
                    writing(&mut c, &mut rng, x + 6, y + h / 2, x + w - 4, ink, p.stroke);
                    if col + 1 < columns {
                        let ay = (y + h / 2) as f32;
                        let ax = (x + w + 2) as f32;
                        c.line([ax, ay], [ax + 12., ay], p.stroke, ink);
                        c.line([ax + 8., ay - 4.], [ax + 12., ay], p.stroke, ink);
                    }
                }
            }
        }
        1 | 2 => {
            c.line(
                [margin as f32 + 9., end as f32 - 12.],
                [end as f32, end as f32 - 12.],
                p.stroke,
                ink,
            );
            c.line(
                [margin as f32 + 9., end as f32 - 12.],
                [margin as f32 + 9., top as f32],
                p.stroke,
                ink,
            );
            let w = width - 22;
            let h = height - 25;
            if p.layout == 1 {
                for series in 0..p.color_count {
                    let phase = rng.random_range(0.0..6.);
                    let point = |i: i32| {
                        let t = i as f32 / w as f32;
                        let v = 0.5
                            + p.slope * (t - 0.5)
                            + p.curvature * t * (1. - t)
                            + 0.07 * (t * 9. + phase).sin();
                        [
                            (margin + 15 + i) as f32,
                            (end - 13) as f32 - v.clamp(0.02, 0.95) * h as f32,
                        ]
                    };
                    for i in 0..w {
                        c.line(point(i), point(i + 1), p.stroke, color(series as usize));
                    }
                }
            } else {
                let count = p.columns as i32 + 2;
                for i in 0..count {
                    let bw = (w / count - 6).max(3);
                    let bh = (rng.random_range(0.13..0.97) * h as f32) as i32;
                    // Marker chart bars are outlined rather than solid paint.
                    box_outline(
                        &mut c,
                        (margin + 17 + i * w / count) as f32,
                        (end - 13 - bh) as f32,
                        bw as f32,
                        bh as f32,
                        p.stroke,
                        color(i as usize),
                    );
                }
            }
        }
        3 | 7 => {
            let rows = (p.rows as f32 * p.occupancy).ceil() as i32;
            for row in 0..rows {
                let y = top + 4 + row * height / (p.rows as i32 + 1);
                if p.layout == 3 {
                    box_outline(
                        &mut c,
                        margin as f32 + 3.,
                        y as f32 - 3.,
                        8.,
                        8.,
                        p.stroke,
                        ink,
                    );
                }
                let x = margin + if p.layout == 3 { 20 } else { 3 };
                let length = rng.random_range(width / 2..width - 6);
                writing(
                    &mut c,
                    &mut rng,
                    x,
                    y,
                    (x + length).min(end),
                    color(row as usize),
                    p.stroke,
                );
            }
        }
        4 => {
            let cols = p.columns as i32;
            let rows = p.rows as i32;
            for col in 0..=cols {
                c.line(
                    [(margin + col * width / cols) as f32, top as f32],
                    [(margin + col * width / cols) as f32, end as f32],
                    1,
                    ink,
                );
            }
            for row in 0..=rows {
                c.line(
                    [margin as f32, (top + row * height / rows) as f32],
                    [end as f32, (top + row * height / rows) as f32],
                    1,
                    ink,
                );
            }
            for row in 0..rows {
                for col in 0..cols {
                    if rng.random_bool(p.occupancy as f64) {
                        let x = margin + col * width / cols + 3;
                        let y = top + row * height / rows + 3;
                        let w = width / cols - 6;
                        let h = height / rows - 6;
                        c.rect(x, y, w, h, [227, 216, 149]);
                        writing(&mut c, &mut rng, x + 3, y + h / 2, x + w - 3, ink, 1);
                    }
                }
            }
        }
        5 => {
            let center = [128., (top + height / 2) as f32];
            c.ring(center, [26., 17.], ink);
            let branches = p.columns + 2;
            let phase = rng.random_range(-0.3..0.3);
            for i in 0..branches {
                let t = i as f32 * std::f32::consts::TAU / branches as f32 + phase;
                let point = [
                    128. + t.cos() * width as f32 * 0.38,
                    center[1] + t.sin() * height as f32 * 0.34,
                ];
                c.line(
                    [128. + t.cos() * 26., center[1] + t.sin() * 17.],
                    point,
                    p.stroke,
                    color(i as usize),
                );
                c.ring(point, [22., 12.], color(i as usize));
                writing(
                    &mut c,
                    &mut rng,
                    point[0] as i32 - 14,
                    point[1] as i32,
                    point[0] as i32 + 14,
                    ink,
                    1,
                );
            }
        }
        6 => {
            for row in 0..p.rows.min(6) as i32 {
                for col in 0..p.columns as i32 {
                    let x = margin + col * width / p.columns as i32;
                    let y = top + row * 22;
                    c.digit(rng.random_range(0..10), x, y, 1, ink);
                    c.rect(x + 10, y + 4, 6, 1, ink);
                    c.digit(rng.random_range(0..10), x + 21, y, 1, color(row as usize));
                }
            }
            c.ring([192., 206.], [28., 24.], color(1));
            c.line([164., 215.], [214., 187.], p.stroke, color(2));
        }
        8 => {
            let split_x = margin + (width as f32 * rng.random_range(0.3..0.7)) as i32;
            let split_y = top + (height as f32 * rng.random_range(0.3..0.7)) as i32;
            box_outline(
                &mut c,
                margin as f32 + 3.,
                top as f32,
                (width - 6) as f32,
                height as f32,
                p.stroke,
                ink,
            );
            // Hand-drawn plan with explicit doorway gaps in internal walls.
            c.line(
                [split_x as f32, top as f32],
                [split_x as f32, split_y as f32 - 14.],
                p.stroke,
                ink,
            );
            c.line(
                [split_x as f32, split_y as f32 + 14.],
                [split_x as f32, end as f32],
                p.stroke,
                ink,
            );
            c.line(
                [margin as f32 + 3., split_y as f32],
                [split_x as f32 - 12., split_y as f32],
                p.stroke,
                ink,
            );
            c.line(
                [split_x as f32 + 12., split_y as f32],
                [end as f32 - 3., split_y as f32],
                p.stroke,
                ink,
            );
            for _ in 0..(p.occupancy * 8.).ceil() as usize {
                let x = rng.random_range(margin + 11..end - 21);
                let y = rng.random_range(top + 10..end - 17);
                box_outline(&mut c, x as f32, y as f32, 14., 8., 1, color(1));
            }
        }
        _ => {
            let cx = (margin + width / 2) as f32;
            let cy = (top + height / 2) as f32;
            let w = width as f32 * 0.53;
            let h = height as f32 * 0.36;
            let points = [
                [cx - w / 2., cy - h / 2.],
                [cx + w / 2., cy - h / 2.],
                [cx + w / 2., cy + h / 2.],
                [cx - w / 2., cy + h / 2.],
            ];
            let offset = [w * (0.15 + p.occupancy * 0.3), -h * 0.5];
            for i in 0..4 {
                let j = (i + 1) % 4;
                let a = points[i];
                let b = points[j];
                c.line(a, b, p.stroke, ink);
                c.line(
                    [a[0] + offset[0], a[1] + offset[1]],
                    [b[0] + offset[0], b[1] + offset[1]],
                    p.stroke,
                    color(1),
                );
                c.line(a, [a[0] + offset[0], a[1] + offset[1]], p.stroke, ink);
            }
            writing(
                &mut c,
                &mut rng,
                margin + 9,
                end - 15,
                end - 30,
                color(2),
                p.stroke,
            );
        }
    }
    c.0
}
pub(super) fn apply(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    let p = parameters(seed);
    let content = pixels(seed);
    p.finish.apply(seed, true, &content, mat, images);
    let mut texture = mip_image(content, 256, MapType::Color);
    texture.sampler = ImageSampler::linear();
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(images.add(texture));
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
    mat.perceptual_roughness = p.roughness;
    mat.emissive = LinearRgba::BLACK;
    mat.emissive_texture = None;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whiteboard_compositions_have_independent_layout_occupancy_and_replay() {
        let mut layouts = std::collections::BTreeSet::new();
        let mut images = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let p = parameters(seed);
            layouts.insert(p.layout);
            let pixels = pixels(seed);
            assert_eq!(pixels.len(), 256 * 256 * 4);
            assert!(pixels.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
            images.insert(pixels);
        }
        assert_eq!(layouts.len(), 10);
        assert_eq!(images.len(), 128);
        assert_eq!(pixels(219), pixels(219));
    }
    #[test]
    fn marker_boards_remain_dielectric_nonemissive_surfaces() {
        let mut images = Assets::<Image>::default();
        for seed in 0..8 {
            let mut material = StandardMaterial::default();
            apply(seed, &mut material, &mut images);
            assert_eq!(material.perceptual_roughness, parameters(seed).roughness);
            assert_eq!(material.metallic, 0.);
            assert_eq!(material.emissive, LinearRgba::BLACK);
            assert!(material.emissive_texture.is_none());
            assert_eq!(material.uv_transform, bevy::math::Affine2::IDENTITY);
        }
    }
}
