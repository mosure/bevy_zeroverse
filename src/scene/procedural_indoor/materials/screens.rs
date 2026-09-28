//! Seeded software layouts and time displays, rasterized without font/image assets.
mod tv;
use super::{
    raster::{Canvas, N},
    *,
};
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;

#[derive(Debug, serde::Serialize)]
pub struct ScreenProgram {
    pub layout: u32,
    pub minutes: u32,
    pub dark: bool,
    pub luminance: f32,
}
pub fn parameters(seed: u64) -> ScreenProgram {
    let mut rng = stream(seed, 861);
    ScreenProgram {
        layout: rng.random_range(0..8),
        minutes: rng.random_range(0..1440),
        dark: rng.random_bool(0.55),
        luminance: rng.random_range(65.0..280.0),
    }
}
pub(super) fn apply(
    seed: u64,
    mobile: bool,
    mat: &mut StandardMaterial,
    images: &mut impl AssetStore<Image>,
) {
    apply_pixels(
        seed,
        if mobile {
            mobile_pixels(seed)
        } else {
            pixels(seed)
        },
        mat,
        images,
    );
}
pub(super) fn apply_tv(seed: u64, mat: &mut StandardMaterial, images: &mut impl AssetStore<Image>) {
    apply_pixels(seed, tv::pixels(seed), mat, images);
}
fn apply_pixels(
    seed: u64,
    pixels: Vec<u8>,
    mat: &mut StandardMaterial,
    images: &mut impl AssetStore<Image>,
) {
    let p = parameters(seed);
    let mut screen = mip_image(pixels, 256, MapType::Color);
    screen.sampler = ImageSampler::linear();
    let texture = images.add(screen);
    mat.base_color = Color::WHITE;
    mat.base_color_texture = Some(texture.clone());
    mat.emissive_texture = Some(texture);
    let brightness = if p.layout == 7 { 0. } else { p.luminance };
    mat.emissive = LinearRgba::new(brightness, brightness, brightness, 1.);
    mat.emissive_exposure_weight = 1.;
    mat.uv_transform = bevy::math::Affine2::IDENTITY;
}
pub const SEGMENTS: [u8; 10] = [
    0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111,
    0b1111111, 0b1101111,
];
pub fn mobile_pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut rng = stream(seed, 863);
    let mut c = Canvas(vec![255; (N * N * 4) as usize]);
    let bg = if p.dark {
        [12, 22, 38]
    } else {
        [232, 237, 242]
    };
    let ink = if p.dark {
        [226, 236, 246]
    } else {
        [34, 47, 61]
    };
    let accent = [
        rng.random_range(48..195),
        rng.random_range(60..190),
        rng.random_range(90..220),
    ];
    c.rect(0, 0, N, N, if p.layout == 7 { [3, 4, 5] } else { bg });
    if p.layout == 7 {
        return c.0;
    }
    c.time(p.minutes, 13, 7, 1, ink);
    c.rect(218, 8, 22, 5, ink);
    if p.layout >= 5 {
        c.time(p.minutes, 40, 38, 6, ink);
        c.rect(20, 134, 216, 28, accent);
        for y in [140, 149] {
            c.rect(30, y, rng.random_range(60..180), 2, ink);
        }
    } else if p.layout <= 2 {
        // Chat, mail or notes: alternating message cards, no desktop sidebar.
        for row in 0..rng.random_range(4..7) {
            let x = if row % 2 == 0 { 17 } else { 50 };
            let y = 32 + row * 29;
            c.rect(x, y, 184, 22, accent);
            for dy in [5, 12] {
                c.rect(x + 7, y + dy, rng.random_range(55..164), 2, ink);
            }
        }
    } else {
        // Home screen/app grid; independent icon and folder occupancy.
        for row in 0..4 {
            for col in 0..4 {
                let x = 20 + col * 57;
                let y = 39 + row * 45;
                let color = [
                    rng.random_range(50..220),
                    rng.random_range(65..210),
                    rng.random_range(65..230),
                ];
                c.rect(x, y, 28, 21, color);
                c.rect(x + 7, y + 7, 14, 7, ink);
            }
        }
    }
    c.rect(95, 247, 66, 3, ink);
    c.0
}
pub fn pixels(seed: u64) -> Vec<u8> {
    let p = parameters(seed);
    let mut rng = stream(seed, 862);
    let mut c = Canvas(vec![255; (N * N * 4) as usize]);
    let bg = if p.dark {
        [17, 23, 31]
    } else {
        [226, 230, 234]
    };
    let panel = if p.dark {
        [30, 40, 51]
    } else {
        [248, 247, 244]
    };
    let ink = if p.dark {
        [170, 188, 202]
    } else {
        [45, 59, 71]
    };
    let accent = [
        rng.random_range(45..210),
        rng.random_range(65..205),
        rng.random_range(80..225),
    ];
    c.rect(0, 0, N, N, if p.layout == 7 { [3, 4, 5] } else { bg });
    if p.layout == 7 {
        return c.0;
    }
    if p.layout == 5 || p.layout == 6 {
        // Procedural landscape or lock screen; broad continuous color fields.
        for y in 0..256 {
            let t = y as f32 / 256.;
            let color = std::array::from_fn(|i| (accent[i] as f32 * (0.35 + t * 0.65)) as u8);
            c.rect(0, y, 256, 1, color);
            for x in 0..256 {
                let ridge = 140. + 22. * (x as f32 * 0.028 + seed as f32 % 7.).sin();
                if y as f32 > ridge {
                    c.rect(x, y, 1, 1, [24, 49, 52]);
                }
            }
        }
        c.time(p.minutes, 39, 52, 6, [238, 241, 244]);
    } else {
        c.rect(0, 0, 256, 20, panel);
        c.rect(0, 22, 37, 222, panel);
        for row in 0..8 {
            c.rect(7, 32 + row * 22, rng.random_range(12..25), 3, ink);
        }
        c.rect(48, 30, 195, 204, panel);
        c.rect(55, 38, rng.random_range(64..160), 5, accent);
        match p.layout {
            0 => {
                // Dashboard, unequal charts and cards.
                for bar in 0..8 {
                    let h = rng.random_range(14..86);
                    c.rect(59 + bar * 22, 149 - h, 13, h, accent);
                }
                for row in 0..5 {
                    c.rect(58, 165 + row * 12, rng.random_range(45..170), 3, ink);
                }
            }
            1 | 2 => {
                // Document or source editor; variable paragraph indentation.
                for row in 0..23 {
                    let indent = if p.layout == 2 {
                        rng.random_range(0..4) * 10
                    } else {
                        0
                    };
                    let y = 56 + row * 7;
                    let length = rng.random_range(40..165 - indent);
                    c.rect(
                        59 + indent,
                        y,
                        length,
                        2,
                        if p.layout == 2 && row % 4 == 0 {
                            accent
                        } else {
                            ink
                        },
                    );
                }
            }
            3 => {
                // Calendar/spreadsheet with independent occupied cells.
                for row in 0..7 {
                    for col in 0..6 {
                        let x = 56 + col * 29;
                        let y = 59 + row * 23;
                        c.rect(x, y, 26, 20, bg);
                        if rng.random_bool(0.44) {
                            c.rect(x + 2, y + 10, rng.random_range(7..22), 5, accent);
                        }
                        c.digit(rng.random_range(0..10), x + 2, y + 1, 1, ink);
                    }
                }
            }
            _ => {
                // Remote meeting/media mosaic with small participant controls.
                for row in 0..2 {
                    for col in 0..3 {
                        let x = 55 + col * 61;
                        let y = 58 + row * 74;
                        c.rect(x, y, 56, 65, bg);
                        let skin = [
                            rng.random_range(90..215),
                            rng.random_range(65..170),
                            rng.random_range(45..145),
                        ];
                        c.rect(x + 19, y + 12, 18, 20, skin);
                        c.rect(x + 10, y + 34, 35, 24, accent);
                    }
                }
            }
        }
    }
    c.rect(0, 244, 256, 12, panel);
    c.time(p.minutes, 219, 246, 1, ink);
    for icon in 0..rng.random_range(4..10) {
        c.rect(18 + icon * 16, 248, 8, 5, accent);
    }
    c.0
}
