//! Floor identification printed once per distinct landing, format and stock.
use super::*;
use crate::scene::procedural_indoor::objects::public::signage::{
    self,
    floor_guide::{Connection, Edition, Sheet, END_SLOT, FIRST_SLOT, INTRODUCED},
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::PrintedPaper && Edition::from_slot(slot).is_some()
}

fn pixels(edition: Edition, seed: u64) -> Vec<u8> {
    let mut rng = stream(seed, u64::from_le_bytes(*b"WayPrint"));
    let (paper, ink, accent) = [
        ([244, 241, 230], [35, 43, 46], [41, 100, 97]),
        ([235, 236, 227], [32, 45, 58], [50, 86, 133]),
        ([245, 233, 213], [57, 44, 42], [142, 67, 50]),
        ([239, 236, 222], [39, 45, 48], [104, 84, 39]),
    ][usize::from(edition.stock)];
    let x = rng.random_range(15..24);
    let right = 256 - x;
    let mut c = Canvas::new(paper);
    let text = |c: &mut Canvas, line: &str, x: i32, y: i32, w: i32, h: i32, stroke, col| {
        lettering::Type::on_sheet(line, w as f32, h as f32, edition.aspect(), stroke).draw(
            c,
            line,
            Vec2::new(x as f32, y as f32),
            false,
            col,
        );
    };
    let g = edition.guide;
    let heading = match g.connection {
        Connection::Lift => "LIFT LOBBY",
        Connection::Stairs => "STAIR HALL",
    };
    let numeral = g.level.to_string();
    let range = match g.connection {
        Connection::Lift => format!("LEVELS 0 TO {}", g.highest),
        Connection::Stairs => if g.level == 0 {
            "LOWER LANDING"
        } else {
            "UPPER LANDING"
        }
        .to_string(),
    };
    if rng.random_bool(0.55) {
        let head = rng.random_range(42..56);
        c.rect(0, 0, 256, head, accent);
        text(&mut c, heading, x, head / 2 - 7, right - x, 15, 1, paper);
        text(&mut c, "LEVEL", x, head + 16, right - x, 13, 1, ink);
        text(
            &mut c,
            &numeral,
            x + 2,
            head + 42,
            right - x - 8,
            95,
            3,
            ink,
        );
    } else {
        c.rect(x, 17, 6, 166, accent);
        text(&mut c, heading, x + 18, 24, right - x - 22, 17, 1, ink);
        text(&mut c, &numeral, x + 20, 64, right - x - 30, 94, 3, accent);
        text(
            &mut c,
            "THIS LEVEL",
            x + 18,
            176,
            right - x - 22,
            12,
            1,
            ink,
        );
    }
    c.rect(x, 203, right - x, 2, accent);
    text(
        &mut c,
        &range,
        x,
        217,
        right - x - 3,
        rng.random_range(14..20),
        1,
        ink,
    );
    c.0
}

pub(crate) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < INTRODUCED {
        return;
    }
    let surface = Surface::PrintedPaper;
    let slots: BTreeSet<_> = if let Some(selection) = selection {
        selection
            .finishes
            .range((surface, FIRST_SLOT)..(surface, END_SLOT))
            .map(|&(_, slot)| slot)
            .collect()
    } else {
        scene
            .objects
            .iter()
            .filter(|o| {
                signage::floor_guide::FloorGuide::for_object(o, scene.generator_version).is_some()
            })
            .filter_map(|o| Sheet::for_stand(o, scene.generator_version, &signage::parameters(o)))
            .map(|s| s.edition.slot())
            .collect()
    };
    for slot in slots {
        let Some(edition) = Edition::from_slot(slot) else {
            continue;
        };
        let seed = variants::content_seed(scene.material_seed(), surface, slot);
        let material = StandardMaterial {
            base_color_texture: Some(images.add(texture(
                pixels(edition, seed),
                256,
                MapType::Color,
            ))),
            perceptual_roughness: [0.81, 0.68, 0.84, 0.62][usize::from(edition.stock)],
            reflectance: 0.34,
            ..default()
        };
        variants.insert((surface, slot), materials.add(material));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::IndoorLayout,
        objects::{audit, public::signage::floor_guide::FloorGuide},
    };

    #[test]
    fn selected_and_whole_room_materials_use_the_same_print_and_mips() {
        let mut s =
            IndoorManifest::generate_with_humans(74002, IndoorLayout::StairHall, 0.65, 0, 0.)
                .unwrap();
        s.objects.clear();
        let empty = || MaterialSelection {
            surfaces: Default::default(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let mut selected = empty();
        let mut o = audit::fixture(
            crate::scene::procedural_indoor::layout::ObjectKind::InformationStand,
            8,
        );
        FloorGuide::lift(0, 14).identify(&mut o);
        let e = Sheet::for_stand(&o, INTRODUCED, &signage::parameters(&o))
            .unwrap()
            .edition;
        let surface = Surface::PrintedPaper;
        let key = (surface, e.slot());
        assert!(registered_finish(surface, e.slot()) && supports(surface, e.slot()));
        assert!(!supports(Surface::Art, e.slot()));
        selected.finishes.insert(key);
        let bake = |s: &IndoorManifest, selection: Option<&MaterialSelection>| {
            let mut images = Assets::default();
            let mut mats = Assets::default();
            let mut slots = BTreeMap::new();
            install(s, &mut images, &mut mats, selection, &mut slots);
            assert_eq!(images.len(), slots.len());
            assert_eq!(mats.len(), slots.len());
            slots.get(&key).map(|h| {
                let mat = mats.get(h).unwrap();
                assert!(mat.perceptual_roughness > 0.6);
                assert_eq!(mat.emissive, LinearRgba::BLACK);
                let image = images
                    .get(mat.base_color_texture.as_ref().unwrap())
                    .unwrap();
                assert!(image.texture_descriptor.format.is_srgb());
                assert_eq!(image.texture_descriptor.mip_level_count, 9);
                assert!(matches!(&image.sampler, ImageSampler::Descriptor(s)
                    if matches!(s.address_mode_u,ImageAddressMode::ClampToEdge)
                    && matches!(s.address_mode_v,ImageAddressMode::ClampToEdge)));
                image.data.clone().unwrap()
            })
        };
        assert!(bake(&s, None).is_none());
        s.objects = vec![o.clone(), o];
        assert!(bake(&s, Some(&empty())).is_none());
        assert_eq!(bake(&s, None).unwrap(), bake(&s, Some(&selected)).unwrap());
        s.generator_version = INTRODUCED - 1;
        assert!(bake(&s, None).is_none());
    }

    #[test]
    fn printed_numerals_and_served_ranges_are_distinguishable() {
        let edition = Edition {
            guide: FloorGuide::lift(0, 14),
            landscape: false,
            stock: 0,
        };
        let rows = |p: &[u8], from: usize, to: usize| p[from * 256 * 4..to * 256 * 4].to_vec();
        let base = pixels(edition, 17);
        let mut glyphs = BTreeSet::new();
        for level in 0..=14 {
            let p = pixels(
                Edition {
                    guide: FloorGuide::lift(level, 14),
                    ..edition
                },
                17,
            );
            assert_eq!(rows(&p, 210, 248), rows(&base, 210, 248));
            assert!(glyphs.insert(rows(&p, 64, 188)));
        }
        let other = pixels(
            Edition {
                guide: FloorGuide::lift(0, 24),
                ..edition
            },
            17,
        );
        assert_ne!(rows(&other, 210, 248), rows(&base, 210, 248));
        assert_eq!(rows(&other, 0, 200), rows(&base, 0, 200));
    }
}
