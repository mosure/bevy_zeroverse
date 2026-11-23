//! Printed opaque keycaps. All legends share a single mipmapped atlas per
//! stock, across every keyboard in a room. UVs on the crown select the key.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::{collections::BTreeMap, sync::OnceLock};

const DARK: usize = 124;
const PALE: usize = 125;

pub(crate) mod legend {
    pub const ESC: u8 = 128;
    pub const TAB: u8 = 129;
    pub const CAPS: u8 = 130;
    pub const SHIFT: u8 = 131;
    pub const ENTER: u8 = 132;
    pub const BACK: u8 = 133;
    pub const CTRL: u8 = 134;
    pub const ALT: u8 = 135;
    pub const FN: u8 = 136;
    pub const SUPER: u8 = 137;
    pub const LEFT: u8 = 138;
    pub const RIGHT: u8 = 139;
    pub const UP: u8 = 140;
    pub const DOWN: u8 = 141;
    pub const HOME: u8 = 142;
    pub const END: u8 = 143;
    pub const INS: u8 = 144;
    pub const DEL: u8 = 145;
    pub const PGUP: u8 = 146;
    pub const PGDN: u8 = 147;
    pub const NUM: u8 = 148;
    pub const F1: u8 = 149;
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Plastic && matches!(slot, DARK | PALE)
}
fn pale(o: &IndoorObject) -> bool {
    stream(o.finish_seed.unwrap_or(o.seed), 0x4b4559504c415354).random_bool(
        if o.kind == ObjectKind::Laptop {
            0.20
        } else {
            0.33
        },
    )
}
pub(crate) fn label(o: &IndoorObject) -> &'static str {
    match (o.kind, pale(o)) {
        (ObjectKind::Laptop, true) => "laptop#finish125#uv_local",
        (ObjectKind::Laptop, false) => "laptop#finish124#uv_local",
        (_, true) => "keyboard#finish125#uv_local",
        (_, false) => "keyboard#finish124#uv_local",
    }
}

const SIDE: usize = 256;
const CELL: usize = 16;
pub(crate) fn uv(code: u8, local: Vec2) -> Vec2 {
    // Keep the crown inside its swatch, even under bilinear sampling.
    let cell = Vec2::new((code % 16) as f32, (code / 16) as f32);
    (cell * CELL as f32 + Vec2::splat(0.5) + local * (CELL - 1) as f32) / SIDE as f32
}

fn glyph(ch: u8) -> [u8; 5] {
    if let Some(rows) = raster::letter_rows(ch) {
        return rows;
    }
    match ch {
        b'0' => [7, 5, 5, 5, 7],
        b'1' => [2, 6, 2, 2, 7],
        b'2' => [7, 1, 7, 4, 7],
        b'3' => [7, 1, 7, 1, 7],
        b'4' => [5, 5, 7, 1, 1],
        b'5' => [7, 4, 7, 1, 7],
        b'6' => [7, 4, 7, 5, 7],
        b'7' => [7, 1, 2, 2, 2],
        b'8' => [7, 5, 7, 5, 7],
        b'9' => [7, 5, 7, 1, 7],
        b'+' => [0, 2, 7, 2, 0],
        b'-' => [0, 0, 7, 0, 0],
        b'=' => [0, 7, 0, 7, 0],
        b'/' => [1, 1, 2, 4, 4],
        b'\\' => [4, 4, 2, 1, 1],
        b'[' => [6, 4, 4, 4, 6],
        b']' => [3, 1, 1, 1, 3],
        b';' => [0, 2, 0, 2, 4],
        b'\'' => [2, 2, 0, 0, 0],
        b'`' => [4, 2, 0, 0, 0],
        b',' => [0, 0, 0, 2, 4],
        b'.' => [0, 0, 0, 0, 2],
        b'*' => [0, 5, 2, 5, 0],
        _ => [0; 5],
    }
}

fn imprint(mut dot: impl FnMut(usize, usize), code: u8) {
    use legend::*;
    let text: &[u8] = match code {
        ESC => b"ESC",
        TAB => b"TAB",
        CAPS => b"CAP",
        CTRL => b"CTL",
        ALT => b"ALT",
        FN => b"FN",
        HOME => b"HM",
        END => b"END",
        INS => b"INS",
        DEL => b"DEL",
        PGUP => b"PU",
        PGDN => b"PD",
        NUM => b"NUM",
        F1..=160 => [
            b"F1".as_slice(),
            b"F2",
            b"F3",
            b"F4",
            b"F5",
            b"F6",
            b"F7",
            b"F8",
            b"F9",
            b"F10",
            b"F11",
            b"F12",
        ][(code - F1) as usize],
        _ => std::slice::from_ref(&code),
    };
    let left = 2;
    for (i, &ch) in text.iter().enumerate() {
        let rows = match ch {
            SHIFT => [2, 7, 5, 2, 2],
            BACK | LEFT => [1, 2, 7, 2, 1],
            RIGHT => [4, 2, 7, 2, 4],
            UP => [2, 7, 2, 2, 0],
            DOWN => [0, 2, 2, 7, 2],
            ENTER => [1, 1, 5, 7, 4],
            SUPER => [7, 5, 5, 5, 7],
            _ => glyph(ch),
        };
        for (row, bits) in rows.into_iter().enumerate() {
            for col in 0..3 {
                if bits & (4 >> col) != 0 {
                    dot(left + i * 4 + col, 3 + row);
                }
            }
        }
    }
}

fn atlas(pale: bool) -> &'static Image {
    static ATLASES: [OnceLock<Image>; 2] = [OnceLock::new(), OnceLock::new()];
    ATLASES[pale as usize].get_or_init(|| {
        let (body, ink) = if pale {
            ([212, 211, 205, 255], [48, 51, 54])
        } else {
            ([48, 51, 55, 255], [220, 223, 222])
        };
        let mut bytes = body.repeat(SIDE * SIDE);
        for code in 0..=u8::MAX {
            let x = usize::from(code % 16) * CELL;
            let y = usize::from(code / 16) * CELL;
            imprint(
                |dx, dy| {
                    let idx = ((y + dy) * SIDE + x + dx) * 4;
                    bytes[idx..idx + 3].copy_from_slice(&ink);
                },
                code,
            );
        }
        let mut image = mip_image(bytes, SIDE as u32, MapType::Color);
        if let ImageSampler::Descriptor(s) = &mut image.sampler {
            s.address_mode_u = ImageAddressMode::ClampToEdge;
            s.address_mode_v = ImageAddressMode::ClampToEdge;
        }
        image
    })
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for (light, slot) in [(false, DARK), (true, PALE)] {
        let key = (Surface::Plastic, slot);
        let used = selection.map_or_else(
            || {
                scene.objects.iter().any(|o| {
                    matches!(o.kind, ObjectKind::Keyboard | ObjectKind::Laptop) && pale(o) == light
                })
            },
            |s| s.finishes.contains(&key),
        );
        if used {
            let mut r = stream(scene.material_seed(), 0x4b45595042520000 | slot as u64);
            let v = r.random_range(0.87..1.);
            let mat = StandardMaterial {
                base_color: Color::srgb(
                    v,
                    v * r.random_range(0.95..1.0),
                    v * r.random_range(0.93..1.0),
                ),
                base_color_texture: Some(images.add(atlas(light).clone())),
                perceptual_roughness: r.random_range(0.42..0.67),
                ..default()
            };
            variants.insert(key, materials.add(mat));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legends_and_blank_stock_survive_the_complete_linear_light_mip_chain() {
        for light in [false, true] {
            let a = atlas(light);
            assert!(std::ptr::eq(a, atlas(light)));
            assert_eq!(a.texture_descriptor.mip_level_count, 9);
            let b = a.data.as_ref().unwrap();
            let sample = |code, x: usize, y: usize| {
                let px = usize::from(code % 16) * CELL + x;
                let py = usize::from(code / 16) * CELL + y;
                b[(py * SIDE + px) * 4]
            };
            let blank = sample(b' ', 8, 8);
            for code in (b'A'..=b'Z')
                .chain(b'0'..=b'9')
                .chain(legend::ESC..=legend::F1 + 11)
            {
                let mut strokes = 0;
                for y in 0..CELL {
                    for x in 0..CELL {
                        let pixel = sample(code, x, y);
                        if pixel != blank {
                            assert!((1..CELL - 1).contains(&x) && (1..CELL - 1).contains(&y));
                            assert_eq!(pixel < blank, light);
                            strokes += 1;
                        }
                        assert_eq!(sample(b' ', x, y), blank);
                    }
                }
                assert!(strokes > 4, "empty legend {code}");
            }
            assert!(b.as_chunks::<4>().0.iter().all(|c| c[3] == 255));
            assert!(b[b.len() - 4].abs_diff(blank) < 12);
        }
    }
}
