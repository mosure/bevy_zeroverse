//! Seeded marker programs: diagrams, graphs, agendas, grids and erased remnants.
use super::{raster::Canvas, *};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

pub fn pixels(seed: u64) -> Vec<u8> {
    let mut rng = stream(seed, 3080);
    let mut c = Canvas(vec![255; 256 * 256 * 4]);
    c.rect(0, 0, 256, 256, [246, 246, 241]);
    let inks = [[28, 47, 76], [38, 89, 161], [159, 52, 49], [39, 116, 86]];
    let ink = inks[rng.random_range(0..4)];
    let mut writing = |c: &mut Canvas, x: i32, y: i32, width: i32| {
        let mut x = x;
        while x < width {
            let n = rng.random_range(6..19).min(width - x);
            for dx in 0..n {
                let dy = rng.random_range(-1..2);
                c.rect(x + dx, y + dy, 1, 2, ink);
            }
            x += n + rng.random_range(4..9);
        }
    };
    // A faint previous sketch remains independent of the new foreground ink.
    c.ring([180., 160.], [35., 28.], [233, 235, 231]);
    let mode = seed % 8;
    match mode {
        0 => {
            // Directional process graph.
            for row in 0..3 {
                for col in 0..2 {
                    let x = 24 + col * 122;
                    let y = 30 + row * 75;
                    for (a, b) in [
                        ([x, y], [x + 80, y]),
                        ([x + 80, y], [x + 80, y + 40]),
                        ([x + 80, y + 40], [x, y + 40]),
                        ([x, y + 40], [x, y]),
                    ] {
                        c.line(a.map(|v| v as f32), b.map(|v| v as f32), 2, ink);
                    }
                    writing(&mut c, x + 10, y + 17, x + 69);
                    if col == 0 {
                        c.line([105., (y + 20) as f32], [143., (y + 20) as f32], 2, ink);
                        c.line([134., (y + 14) as f32], [143., (y + 20) as f32], 2, ink);
                    }
                }
            }
        }
        1 | 2 => {
            // Axes with an independently curved trajectory or bars.
            c.line([27., 211.], [236., 211.], 2, ink);
            c.line([27., 211.], [27., 27.], 2, ink);
            if mode == 1 {
                let phase = (seed % 29) as f32 * 0.07;
                for x in 30..235 {
                    let y =
                        |x: i32| 151. - 57. * (x as f32 * 0.023 + phase).sin() - x as f32 * 0.18;
                    c.line([x as f32, y(x)], [(x + 1) as f32, y(x + 1)], 2, inks[1]);
                }
            } else {
                for i in 0..7 {
                    let h = (seed.rotate_right(i * 3) % 128 + 24) as i32;
                    c.rect(39 + i as i32 * 27, 211 - h, 15, h, inks[i as usize % 4]);
                }
            }
            for row in 0..3 {
                writing(&mut c, 130, 22 + row * 9, 222);
            }
        }
        3 => {
            // Agenda with checkbox rows.
            for row in 0..8 {
                let y = 32 + row * 25;
                c.rect(21, y - 3, 8, 8, ink);
                writing(&mut c, 39, y, 226 - (row % 3) * 17);
            }
        }
        4 => {
            // Planning grid with colored work items.
            for x in [22, 75, 128, 181, 234] {
                c.line([x as f32, 26.], [x as f32, 232.], 1, ink);
            }
            for y in [26, 61, 96, 131, 166, 201, 232] {
                c.line([22., y as f32], [234., y as f32], 1, ink);
            }
            for i in 0..11 {
                let x = 27 + (i * 73 % 4) * 53;
                let y = 67 + (i * 31 % 5) * 35;
                c.rect(x, y, 39, 19, [228, 215, 150]);
                writing(&mut c, x + 3, y + 9, x + 32);
            }
        }
        5 => {
            // Mind map with branches and variable labels.
            c.ring([128., 127.], [34., 20.], ink);
            for i in 0..6 {
                let t = i as f32 * std::f32::consts::TAU / 6.;
                let p = [128. + t.cos() * 89., 127. + t.sin() * 85.];
                c.line(
                    [128. + t.cos() * 35., 127. + t.sin() * 21.],
                    p,
                    2,
                    inks[i % 4],
                );
                c.ring(p, [23., 14.], ink);
                writing(&mut c, p[0] as i32 - 14, p[1] as i32, p[0] as i32 + 15);
            }
        }
        6 => {
            // Numerical sketches and geometry.
            for row in 0..5 {
                for col in 0..5 {
                    c.digit(
                        ((seed >> (col + row)) % 10) as u32,
                        25 + col * 25,
                        27 + row * 24,
                        1,
                        ink,
                    );
                    c.rect(37 + col * 25, 30 + row * 24, 5, 1, ink);
                }
            }
            c.ring([185., 178.], [39., 36.], inks[1]);
            c.line([150., 193.], [208., 155.], 2, inks[2]);
        }
        _ => {
            // Sparse recently erased board.
            writing(&mut c, 24, 37, 169);
            writing(&mut c, 26, 54, 139);
        }
    }
    c.0
}
pub(super) fn apply(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    let mut texture = mip_image(pixels(seed), 256, MapType::Color);
    texture.sampler = ImageSampler::linear();
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(images.add(texture));
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
    mat.perceptual_roughness = 0.23;
    mat.clearcoat = 0.28;
    mat.clearcoat_perceptual_roughness = 0.18;
}
