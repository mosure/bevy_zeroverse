//! Small, filtered glyph sheets shared by appliance fascias and elevator stations. Duration, digit order and
//! key layout are mesh UVs; a new timer value never needs another texture.
use super::*;
use crate::scene::procedural_indoor::{
    objects::kitchen::appliances::controls, preparation::AssetStore,
};
use std::collections::{BTreeMap, BTreeSet};

const FIRST: usize = 805;
const SIDE: usize = 128;
const CELL: usize = SIDE / 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Glyph {
    Number(u8),
    Colon,
    Start,
    Stop,
    Plus,
    Scale,
    Blank,
}
impl Glyph {
    const fn index(self) -> usize {
        match self {
            Self::Number(n) => n as usize,
            Self::Colon => 10,
            Self::Start => 11,
            Self::Stop => 12,
            Self::Plus => 13,
            Self::Scale => 14,
            Self::Blank => 15,
        }
    }
    pub(crate) fn uvs(self) -> [[f32; 2]; 4] {
        let i = self.index();
        let x = (i % 4) as f32 * 0.25;
        let y = (i / 4) as f32 * 0.25;
        let inset = 0.5 / SIDE as f32;
        [
            [x + inset, y + 0.25 - inset],
            [x + 0.25 - inset, y + 0.25 - inset],
            [x + 0.25 - inset, y + inset],
            [x + inset, y + inset],
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Film {
    DarkKey,
    LightKey,
    ReflectiveLcd,
    Led,
    BlankLcd,
    DarkWindow,
}

impl Film {
    const ALL: [Self; 6] = [
        Self::DarkKey,
        Self::LightKey,
        Self::ReflectiveLcd,
        Self::Led,
        Self::BlankLcd,
        Self::DarkWindow,
    ];
    fn slot(self) -> usize {
        FIRST + self as usize
    }
    pub(crate) fn label(self, class: &str) -> String {
        format!("{class}#finish{}#uv_local", self.slot())
    }
}
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Ink && (FIRST..FIRST + Film::ALL.len()).contains(&slot)
}

// Leave a gutter around each glyph for bilinear/trilinear sampling. The keypad
// uses printed numerals, while an LCD or LED window uses seven electrodes.
fn coverage(tile: usize, x: usize, y: usize, digital: bool) -> bool {
    match tile {
        0..=9 if digital => {
            if !(6..26).contains(&x) || !(3..30).contains(&y) {
                return false;
            }
            let (col, row) = ((x - 6) / 4, (y - 3) / 3);
            let segment = match (col, row) {
                (1..=3, 0) => 0,
                (4, 1..=3) => 1,
                (4, 5..=7) => 2,
                (1..=3, 8) => 3,
                (0, 5..=7) => 4,
                (0, 1..=3) => 5,
                (1..=3, 4) => 6,
                _ => return false,
            };
            screens::SEGMENTS[tile] & (1 << segment) != 0
        }
        0..=9 => {
            (10..22).contains(&x)
                && (6..26).contains(&y)
                && raster::number_rows(tile)[(y - 6) / 4] & (4 >> ((x - 10) / 4)) != 0
        }
        10 => (12..20).contains(&x) && ((9..13).contains(&y) || (20..24).contains(&y)),
        _ if digital => false,
        11 => (10..25).contains(&x) && (y as i32 - 16).abs() < (25 - x) as i32 * 2 / 3,
        12 => (10..22).contains(&x) && (10..22).contains(&y),
        13 => {
            ((9..23).contains(&x) && (14..18).contains(&y))
                || ((14..18).contains(&x) && (9..23).contains(&y))
        }
        14 => {
            let p = Vec2::new(x as f32 - 15.5, 15.5 - y as f32);
            let angle = p.x.atan2(p.y);
            (12.4..15.2).contains(&p.length())
                && angle.abs() < 2.56
                && (angle / 0.5 - (angle / 0.5).round()).abs() < 0.085
        }
        _ => false,
    }
}

fn colors(film: Film) -> ([u8; 3], [u8; 3]) {
    match film {
        Film::DarkKey => ([42, 47, 49], [224, 228, 224]),
        Film::LightKey => ([192, 187, 174], [49, 53, 51]),
        Film::ReflectiveLcd | Film::BlankLcd => ([163, 177, 152], [42, 50, 45]),
        Film::Led | Film::DarkWindow => ([24, 29, 29], [91, 99, 95]),
    }
}

fn atlas(film: Film, emission: bool) -> &'static Image {
    use std::sync::OnceLock;
    // Four color sheets and one LED mask for the lifetime of the process.
    static ATLASES: [OnceLock<Image>; 5] = [const { OnceLock::new() }; 5];
    let index = if emission { 4 } else { film as usize };
    ATLASES[index].get_or_init(|| {
        let (base, ink) = if emission {
            ([0; 3], [255; 3])
        } else {
            colors(film)
        };
        let mut bytes = Vec::with_capacity(SIDE * SIDE * 4);
        for y in 0..SIDE {
            for x in 0..SIDE {
                let mark = coverage(
                    y / CELL * 4 + x / CELL,
                    x % CELL,
                    y % CELL,
                    matches!(film, Film::Led | Film::ReflectiveLcd),
                );
                let [r, g, b] = if mark { ink } else { base };
                bytes.extend_from_slice(&[r, g, b, 255]);
            }
        }
        let mut image = mip_image(bytes, SIDE as u32, MapType::Color);
        if let ImageSampler::Descriptor(s) = &mut image.sampler {
            s.address_mode_u = ImageAddressMode::ClampToEdge;
            s.address_mode_v = ImageAddressMode::ClampToEdge;
        }
        image
    })
}

fn make_material(film: Film, seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    use rand::Rng;
    let mut r =
        super::super::layout::stream(seed, u64::from_le_bytes(*b"FasciaUV") ^ film.slot() as u64);
    let roughness = match film {
        Film::DarkKey | Film::LightKey => r.random_range(0.46..0.67),
        Film::ReflectiveLcd | Film::BlankLcd => r.random_range(0.29..0.43),
        Film::Led | Film::DarkWindow => r.random_range(0.22..0.34),
    };
    let mut mat = StandardMaterial {
        perceptual_roughness: roughness,
        ..default()
    };
    if matches!(film, Film::BlankLcd | Film::DarkWindow) {
        let [r, g, b] = colors(film).0;
        mat.base_color = Color::srgb_u8(r, g, b);
        return mat;
    }
    let value = r.random_range(0.88..1.0);
    mat.base_color = Color::srgb(
        value,
        value * r.random_range(0.96..1.),
        value * r.random_range(0.94..1.),
    );
    mat.base_color_texture = Some(images.add(atlas(film, false).clone()));
    if film == Film::Led {
        mat.emissive_texture = Some(images.add(atlas(film, true).clone()));
        let tint = if r.random_bool(0.53) {
            Vec3::new(1., r.random_range(0.32..0.53), 0.055)
        } else {
            Vec3::new(0.20, 1., r.random_range(0.40..0.70))
        };
        let e = tint * r.random_range(0.9..1.9);
        mat.emissive = LinearRgba::rgb(e.x, e.y, e.z);
    }
    mat
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < controls::INTRODUCED {
        return;
    }
    let used: BTreeSet<_> = if let Some(s) = selection {
        Film::ALL
            .into_iter()
            .filter(|f| s.finishes.contains(&(Surface::Ink, f.slot())))
            .collect()
    } else {
        use super::super::{layout::ObjectKind, objects::public::elevator};
        let mut films = BTreeSet::new();
        for o in &scene.objects {
            if let Some(p) = controls::for_object(o) {
                films.insert(p.key_film);
                films.extend(p.readout.map(|d| d.film));
            } else if scene.generator_version >= elevator::LANDING_VERSION
                && o.kind == ObjectKind::Elevator
            {
                let station = elevator::Station::for_object(o);
                films.extend([station.key_film, station.display_film]);
            }
        }
        films
    };
    for f in used {
        variants.insert(
            (Surface::Ink, f.slot()),
            materials.add(make_material(f, scene.material_seed(), images)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn filtered_stocks_emit_only_on_segments_and_blank_windows_need_no_images() {
        let mut scene =
            IndoorManifest::generate_with_humans(94001, IndoorLayout::Kitchen, 0.65, 0, 0.)
                .unwrap();
        let mut materials = Assets::<StandardMaterial>::default();
        let mut images = Assets::<Image>::default();
        let mut variants = BTreeMap::new();
        let mut selection = MaterialSelection {
            surfaces: default(),
            finishes: default(),
            direct_surfaces: default(),
        };
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut variants,
        );
        assert!(images.is_empty() && variants.is_empty());
        selection.finishes = Film::ALL
            .into_iter()
            .map(|f| (Surface::Ink, f.slot()))
            .collect();
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut variants,
        );
        assert_eq!(images.len(), 5);
        for f in Film::ALL {
            let key = (Surface::Ink, f.slot());
            assert!(registered_finish(key.0, key.1));
            assert!(!supports(Surface::Screen, key.1));
            let mat = materials.get(&variants[&key]).unwrap();
            assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
            assert_eq!(mat.metallic, 0.);
            assert_eq!(mat.emissive_texture.is_some(), f == Film::Led);
            let Some(color) = mat.base_color_texture.as_ref() else {
                assert!(matches!(f, Film::BlankLcd | Film::DarkWindow));
                continue;
            };
            let image = images.get(color).unwrap();
            assert_eq!(image.texture_descriptor.mip_level_count, 8);
            assert!(image.texture_descriptor.format.is_srgb());
            if let Some(h) = &mat.emissive_texture {
                let bytes = images.get(h).unwrap().data.as_ref().unwrap();
                let base = image.data.as_ref().unwrap();
                assert_eq!(bytes.len(), base.len());
                for (i, rgba) in bytes[..SIDE * SIDE * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                {
                    let x = i % SIDE;
                    let y = i / SIDE;
                    let expected = coverage(y / CELL * 4 + x / CELL, x % CELL, y % CELL, true);
                    assert_eq!(rgba[..3] != [0; 3], expected);
                    assert_eq!(base[4 * i..4 * i + 3] != colors(f).0, expected);
                }
            }
        }
        // Full installations of archived scenes and empty current rooms do
        // not allocate any panel asset, even without precomputed selection.
        for version in [controls::INTRODUCED - 1, controls::INTRODUCED] {
            scene.generator_version = version;
            if version == controls::INTRODUCED {
                scene.objects.clear();
            }
            let mut unused = BTreeMap::new();
            install(&scene, &mut images, &mut materials, None, &mut unused);
            assert!(unused.is_empty());
            assert_eq!(images.len(), 5);
        }
    }
}
