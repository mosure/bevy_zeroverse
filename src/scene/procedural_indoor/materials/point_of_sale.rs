//! Small point-of-sale documents. A shop shares a palette and a used document
//! across terminals and camera views. Colour and emission reuse one mip chain.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject},
    objects::public::{retail::checkout, CheckoutForm},
    preparation::AssetStore,
};
use rand::{seq::SliceRandom, Rng};
use std::collections::BTreeMap;

pub(crate) mod stock;
const FIRST: usize = 982;
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum Page {
    GrocerySale,
    ApparelSale,
    Welcome,
    GroceryPayment,
    Keypad,
    ApparelPayment,
}
use Page::*;
const PAGES: [Page; 6] = [
    GrocerySale,
    ApparelSale,
    Welcome,
    GroceryPayment,
    Keypad,
    ApparelPayment,
];
pub(crate) const READER: &str = "control_panel#finish986#uv_local";

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    (s == Surface::Ink && (FIRST..FIRST + PAGES.len()).contains(&slot)) || stock::supports(s, slot)
}
fn page(o: &IndoorObject, form: CheckoutForm) -> Page {
    match form {
        CheckoutForm::ConveyorLane | CheckoutForm::BasketRegister => GrocerySale,
        CheckoutForm::ApparelWrap => ApparelSale,
        CheckoutForm::SelfCheckout => {
            if stream(o.seed, u64::from_le_bytes(*b"SalePage")).random_bool(0.48) {
                Welcome
            } else if CheckoutForm::apparel_assortment(o) {
                ApparelPayment
            } else {
                GroceryPayment
            }
        }
    }
}
pub(crate) fn label(o: &IndoorObject, form: CheckoutForm) -> String {
    format!(
        "control_panel#finish{}#uv_local",
        FIRST + page(o, form) as usize
    )
}
fn money(c: &mut raster::Canvas, cents: u32, x: i32, y: i32, scale: i32, ink: [u8; 3]) {
    let mut at = x;
    for ch in format!("{}.{:02}", cents / 100, cents % 100).bytes() {
        if ch == b'.' {
            c.rect(at, y + 4 * scale, scale, scale, ink);
            at += 2 * scale;
        } else {
            for (row, bits) in raster::number_rows((ch - b'0') as usize)
                .into_iter()
                .enumerate()
            {
                for col in 0..3 {
                    if bits & (4 >> col) != 0 {
                        c.rect(at + col * scale, y + row as i32 * scale, scale, scale, ink);
                    }
                }
            }
            at += 4 * scale;
        }
    }
}

fn image(seed: u64, doc: Page) -> Image {
    let apparel = matches!(doc, ApparelSale | ApparelPayment);
    let mut shop = stream(seed, u64::from_le_bytes(*b"POSBrand"));
    let accent = [
        [37, 108, 113],
        [103, 69, 113],
        [48, 88, 132],
        [134, 76, 49],
        [55, 105, 74],
    ][shop.random_range(0..5)];
    let dark = [31, 39, 44];
    let paper = [232, 235, 228];
    let mut r = stream(seed, u64::from_le_bytes(*b"POSText ") + doc as u64);
    let mut c = raster::Canvas::new(if doc == Keypad { dark } else { paper });
    if doc == Keypad {
        c.rect(10, 9, 236, 71, [156, 189, 168]);
        c.text("TAP CARD", 40, 34, 4, dark);
        for row in 0..4 {
            for col in 0..3 {
                let x = 12 + col * 80;
                let y = 95 + row * 38;
                c.rect(
                    x,
                    y,
                    70,
                    32,
                    if row == 3 && col == 2 {
                        [52, 121, 76]
                    } else {
                        [72, 79, 82]
                    },
                );
                if row < 3 || col == 1 {
                    let n = if row == 3 { 0 } else { 1 + 3 * row + col };
                    c.digit(n as u32, x + 24, y + 4, 2, paper);
                }
            }
        }
    } else {
        let pad = r.random_range(10..19);
        let head = r.random_range(33..50);
        c.rect(0, 0, 256, head, accent);
        c.text(
            if doc == ApparelSale {
                "CASH WRAP"
            } else if doc == Welcome {
                "WELCOME"
            } else {
                "CHECKOUT"
            },
            pad,
            (head - 15) / 2,
            3,
            [247, 249, 246],
        );
        if doc == Welcome {
            // A scan cue and a single touch target on the idle screen.
            c.rect(85, 70, 90, 66, [212, 218, 214]);
            let mut x = 97;
            while x < 161 {
                let width = r.random_range(2..6);
                c.rect(x, 81, width, 43, dark);
                x += width + r.random_range(2..5);
            }
            c.text("SCAN AN ITEM", pad, 153, 3, dark);
            c.rect(pad, 202, 256 - 2 * pad, 34, accent);
            c.text("START", pad + 10, 214, 2, paper);
        } else {
            let mut names = if apparel {
                ["SHIRT", "DENIM", "SCARF", "SOCKS", "JACKET", "KNIT"]
            } else {
                ["PEARS", "COFFEE", "BREAD", "MILK", "RICE", "LIMES"]
            };
            names.shuffle(&mut r);
            let count = r.random_range(2..=5);
            let mut total = 0;
            for (i, name) in names.iter().take(count).enumerate() {
                let y = head + 17 + i as i32 * 23;
                let price = r.random_range(if apparel { 495..8995 } else { 89..1295 });
                total += price;
                c.text(name, pad, y, 2, dark);
                money(&mut c, price, 181, y, 2, dark);
                c.rect(pad, y + 14, 256 - pad * 2, 1, [174, 183, 179]);
            }
            c.text("TOTAL", pad, 179, 3, dark);
            money(&mut c, total, 148, 179, 3, dark);
            c.rect(pad, 211, 256 - 2 * pad, 30, accent);
            c.text(
                if matches!(doc, GroceryPayment | ApparelPayment) {
                    "PAY NOW"
                } else {
                    "PAYMENT"
                },
                pad + 9,
                220,
                2,
                paper,
            );
        }
    }
    mip_image(c.0, raster::N as u32, MapType::Color)
}
pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < checkout::VERSION {
        return;
    }
    stock::install(scene, images, materials, selection, variants);
    let mut used = [false; PAGES.len()];
    if let Some(s) = selection {
        for (_, slot) in s
            .finishes
            .range((Surface::Ink, FIRST)..(Surface::Ink, FIRST + PAGES.len()))
        {
            used[*slot - FIRST] = true;
        }
    } else {
        for o in &scene.objects {
            if let Some(f) = CheckoutForm::for_object(o) {
                used[page(o, f) as usize] = true;
                used[Keypad as usize] = true;
            }
        }
    }
    for (doc, needed) in PAGES.into_iter().zip(used) {
        if !needed {
            continue;
        }
        let art = if doc == Keypad {
            static KEYPAD: std::sync::OnceLock<Image> = std::sync::OnceLock::new();
            KEYPAD.get_or_init(|| image(0, Keypad)).clone()
        } else {
            image(scene.material_seed(), doc)
        };
        let map = images.add(art);
        variants.insert(
            (Surface::Ink, FIRST + doc as usize),
            materials.add(StandardMaterial {
                base_color_texture: Some(map.clone()),
                emissive_texture: Some(map),
                emissive: LinearRgba::rgb(0.72, 0.72, 0.72),
                perceptual_roughness: 0.33,
                reflectance: 0.40,
                ..default()
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::ObjectKind,
        objects::{audit, build_object},
    };
    use std::collections::BTreeSet;
    #[test]
    fn point_of_sale_binds_only_visible_documents_once_and_keeps_complete_mips() {
        let mut room =
            IndoorManifest::generate_with_humans(700, Default::default(), 0.65, 0, 0.).unwrap();
        room.objects.clear();
        let test = |scene: &IndoorManifest| {
            let expected: BTreeSet<_> = scene
                .objects
                .iter()
                .flat_map(|o| build_object(o).parts.into_keys())
                .filter_map(|(s, l)| {
                    finish_slot(s, &l)
                        .filter(|&slot| supports(s, slot))
                        .map(|slot| (s, slot))
                })
                .collect();
            let subset = MaterialSelection {
                surfaces: default(),
                direct_surfaces: default(),
                finishes: expected.clone(),
            };
            let mut all_modes = Vec::new();
            for selection in [None, Some(&subset)] {
                let mut images = Assets::<Image>::default();
                let mut materials = Assets::<StandardMaterial>::default();
                let mut bindings = BTreeMap::new();
                install(scene, &mut images, &mut materials, selection, &mut bindings);
                assert_eq!(expected, bindings.keys().copied().collect());
                assert_eq!(
                    images.len(),
                    expected
                        .iter()
                        .filter(|&&(s, slot)| !stock::supports(s, slot) || s == Surface::Rubber)
                        .count()
                );
                let pixels: Vec<_> = bindings
                    .iter()
                    .map(|(&(surface, slot), handle)| {
                        let m = materials.get(handle).unwrap();
                        if stock::supports(surface, slot) {
                            assert_eq!(m.metallic, 0.);
                            assert_eq!(m.alpha_mode, AlphaMode::Opaque);
                            assert!(m.emissive_texture.is_none() && m.base_color_texture.is_none());
                            assert!(m.base_color.to_srgba().red < 0.19);
                            if surface == Surface::Rubber {
                                assert!((0.60..0.86).contains(&m.perceptual_roughness));
                                let g = images
                                    .get(m.metallic_roughness_texture.as_ref().unwrap())
                                    .unwrap();
                                assert_eq!(g.texture_descriptor.mip_level_count, 6);
                                assert!(g
                                    .data
                                    .as_ref()
                                    .unwrap()
                                    .as_chunks::<4>()
                                    .0
                                    .iter()
                                    .all(|v| v[2] == 0 && (209..=248).contains(&v[1])));
                                return g.data.as_ref().unwrap().clone();
                            }
                            assert!((0.09..0.20).contains(&m.perceptual_roughness));
                            return vec![];
                        }
                        assert_eq!(m.base_color_texture, m.emissive_texture);
                        let img = images.get(m.base_color_texture.as_ref().unwrap()).unwrap();
                        assert_eq!(img.texture_descriptor.mip_level_count, 9);
                        img.data.as_ref().unwrap().clone()
                    })
                    .collect();
                all_modes.push(pixels);
            }
            assert_eq!(all_modes[0], all_modes[1]);
            expected.len()
        };
        assert_eq!(test(&room), 0);
        room.objects = (0..48)
            .map(|seed| audit::fixture(ObjectKind::CheckoutCounter, seed))
            .collect();
        assert_eq!(test(&room), PAGES.len() + 2);
        assert_ne!(image(31, GrocerySale).data, image(32, GrocerySale).data);
        assert_ne!(
            image(31, GroceryPayment).data,
            image(31, ApparelPayment).data
        );
        assert_eq!(image(31, Keypad).data, image(32, Keypad).data);
    }
}
