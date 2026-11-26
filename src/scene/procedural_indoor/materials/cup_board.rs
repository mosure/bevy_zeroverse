//! Pressed and printed cup jackets. The forming tool is independent of the ink lot.
//! Each occupied lot shares its sheet and each press shares one tangent-space map.
use super::*;
use crate::scene::procedural_indoor::{
    clutter::beverages,
    layout::{stream, IndoorObject, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

pub(crate) const INTRODUCED: u32 = 266;
const FIRST: usize = 840;
const LOTS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub(crate) enum Press {
    Smooth,
    Microflute,
    Embossed,
}
const PRESSES: [Press; 3] = [Press::Smooth, Press::Microflute, Press::Embossed];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct Stock {
    pub lot: usize,
    pub press: Press,
    pub printed: bool,
}
impl Stock {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        let mut rng = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"CupStock"),
        );
        let press = match rng.random_range(0..10) {
            0..=4 => Press::Smooth,
            5..=7 => Press::Microflute,
            _ => Press::Embossed,
        };
        Self {
            press,
            lot: rng.random_range(0..LOTS),
            printed: rng.random_bool(0.56),
        }
    }
    fn slot(self) -> usize {
        FIRST + (self.press as usize * 2 + usize::from(self.printed)) * LOTS + self.lot
    }
    fn from_slot(slot: usize) -> Self {
        let i = slot - FIRST;
        Self {
            press: PRESSES[i / (2 * LOTS)],
            printed: (i / LOTS) % 2 == 1,
            lot: i % LOTS,
        }
    }
    pub(crate) fn label(self) -> String {
        format!("drink_container#finish{}#uv_local", self.slot())
    }
}
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Paper && (FIRST..FIRST + 2 * PRESSES.len() * LOTS).contains(&slot)
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < INTRODUCED {
        return;
    }
    let slots: BTreeSet<_> = if let Some(s) = selection {
        s.finishes
            .iter()
            .filter(|&&(surface, i)| supports(surface, i))
            .map(|&(_, i)| i)
            .collect()
    } else {
        scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::CoffeeCup && beverages::parameters(o).style % 3 != 2)
            .map(|o| Stock::for_object(o).slot())
            .collect()
    };
    let mut sheets = BTreeMap::new();
    let mut presses = BTreeMap::new();
    for slot in slots {
        let stock = Stock::from_slot(slot);
        let sheet = sheets.entry((stock.lot, stock.printed)).or_insert_with(|| {
            let seed = scene.material_seed() ^ u64::from_le_bytes(*b"CupSheet") ^ stock.lot as u64;
            images.add(wrap_image(mip_image(
                sheet_pixels(seed, stock.printed),
                256,
                MapType::Color,
            )))
        });
        let relief = presses.entry(stock.press).or_insert_with(|| {
            static NORMALS: [OnceLock<Image>; 3] = [const { OnceLock::new() }; 3];
            images.add(
                NORMALS[stock.press as usize]
                    .get_or_init(|| press_map(stock.press))
                    .clone(),
            )
        });
        let mut r = stream(
            scene.material_seed(),
            u64::from_le_bytes(*b"CupSize ") ^ stock.lot as u64,
        );
        let roughness = r.random_range(0.78..0.94) - if stock.printed { 0.06 } else { 0. };
        let mat = StandardMaterial {
            base_color_texture: Some(sheet.clone()),
            normal_map_texture: Some(relief.clone()),
            perceptual_roughness: roughness,
            reflectance: 0.4,
            ..default()
        };
        variants.insert((Surface::Paper, slot), materials.add(mat));
    }
}

fn wrap_image(mut image: Image) -> Image {
    if let ImageSampler::Descriptor(s) = &mut image.sampler {
        s.address_mode_v = ImageAddressMode::ClampToEdge;
    }
    image
}

fn sheet_pixels(seed: u64, printed: bool) -> Vec<u8> {
    let mut r = stream(seed, 0);
    let stock = [
        [187, 149, 102],
        [208, 180, 139],
        [157, 120, 82],
        [228, 218, 194],
        [208, 202, 184],
    ][r.random_range(0..5)];
    let shade: i16 = r.random_range(-9..10);
    let base = stock.map(|x: i16| (x + shade) as u8);
    let mut c = raster::Canvas::new(base);
    if printed {
        let ink = [
            [48, 62, 57],
            [58, 40, 34],
            [36, 55, 72],
            [101, 54, 47],
            [46, 47, 41],
        ][r.random_range(0..5)];
        let name = format!(
            "{}{}",
            ["AL", "ROW", "EL", "BRI", "TOR", "KES", "MON"][r.random_range(0..7)],
            ["DER", "NA", "VEN", "LY", "RO", "TON", "VA"][r.random_range(0..7)]
        );
        let word_y = r.random_range(94..142);
        let first = r.random_range(39..85);
        let second = first + r.random_range(110..140);
        let layout = r.random_range(0..4);
        for x in [first, second] {
            let title_width = 4 * name.len() as i32;
            c.lettering(&name, x - title_width / 2, word_y, [1, 6], ink);
            match layout {
                0 => {
                    c.rect(x - title_width / 2, word_y + 38, title_width, 3, ink);
                    c.lettering("ROAST", x - 10, word_y + 49, [1, 3], ink);
                }
                1 => {
                    c.ellipse([x as f32, (word_y - 26) as f32], [4.4, 16.], ink);
                    c.line(
                        [(x - 2) as f32, (word_y - 35) as f32],
                        [(x + 2) as f32, (word_y - 17) as f32],
                        1,
                        base,
                    );
                }
                2 => {
                    c.polygon(
                        &[
                            [(x - 8) as f32, (word_y - 12) as f32],
                            [x as f32, (word_y - 44) as f32],
                            [(x + 8) as f32, (word_y - 12) as f32],
                        ],
                        ink,
                    );
                }
                _ => {
                    c.lettering("CAFE", x - 8, word_y + 45, [1, 3], ink);
                }
            }
        }
        if r.random_bool(0.45) {
            let y = r.random_range(177..221);
            c.rect(0, y, 256, r.random_range(2..6), ink);
        }
    }
    // Fine short fibres and pulp flecks stay fixed if a sheet receives ink.
    for (i, pixel) in c.0.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = ((i % 256) as u32, (i / 256) as u32);
        let fibre = hash(x / 3, y, seed);
        let tint = ((fibre - 0.5) * 8. + (hash(x, y, seed ^ 0x71) - 0.5) * 5.
            - if fibre > 0.987 { 12. } else { 0. }) as i16;
        for c in &mut pixel[..3] {
            *c = (i16::from(*c) + tint).clamp(0, 255) as u8;
        }
    }
    c.0
}

fn press_map(press: Press) -> Image {
    use std::f32::consts::TAU;
    const N: usize = 128;
    let mut field = Vec::with_capacity(N * N);
    for y in 0..N {
        for x in 0..N {
            let u = (x as f32 + 0.5) / N as f32;
            let v = (y as f32 + 0.5) / N as f32;
            let (mut dx, mut dy) = match press {
                Press::Smooth => (0., 0.),
                Press::Microflute => (0.22 * (TAU * 30. * u).sin(), 0.),
                Press::Embossed => {
                    let (a, b) = (
                        (TAU * (18. * u + 3. * v)).sin(),
                        (TAU * (18. * u - 3. * v)).sin(),
                    );
                    (0.060 * (a + b), 0.055 * (a - b))
                }
            };
            dx += 0.08 * (hash(x as u32, y as u32, 73) - 0.5);
            dy += 0.08 * (hash(x as u32, y as u32, 277) - 0.5);
            field.push(Vec3::new(dx, dy, 1.).normalize());
        }
    }
    // Retain the unquantized area sum between levels. Rounding the encoded
    // directions repeatedly gives an otherwise flat distant sheet a false tilt.
    let mut bytes = Vec::with_capacity((4 * N * N - 1) / 3 * 4);
    let mut n = N;
    loop {
        for v in &field[..n * n] {
            bytes.extend(
                v.normalize()
                    .to_array()
                    .map(|c| (127.5 * (c + 1.)).round() as u8),
            );
            bytes.push(255);
        }
        if n == 1 {
            break;
        }
        let half = n / 2;
        for y in 0..half {
            for x in 0..half {
                let i = 2 * (y * n + x);
                field[y * half + x] = field[i] + field[i + 1] + field[i + n] + field[i + n + 1];
            }
        }
        n = half;
    }
    wrap_image(mip_chain_image(bytes, N as u32, MapType::Normal))
}

#[cfg(test)]
mod tests;
