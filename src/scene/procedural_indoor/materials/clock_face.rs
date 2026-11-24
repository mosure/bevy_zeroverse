//! Dial printing belongs to the clock instance, including its displayed time.
//! Rasterize once per used face and share the handle across all camera views.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::clocks::{self, ClockProgram, DialStyle, PrintedDial},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

mod print;

// Instance finishes occupy a disjoint, platform-independent range. IDs are
// indices in the room's object array; all preparation paths use those same IDs.
const INSTANCE: usize = 1 << 20;
const CHARCOAL: usize = 610;
const VERMILION: usize = 612;

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    (s == Surface::Paper && (INSTANCE..(1 << 31)).contains(&slot))
        || (s == Surface::Ink && (CHARCOAL..=VERMILION).contains(&slot))
}
pub(crate) fn face_label(o: &IndoorObject) -> String {
    format!("clock#finish{}#uv_local", INSTANCE + o.id)
}
pub(crate) fn hand_label(dark: bool, second: bool) -> &'static str {
    if second {
        "clock#finish612"
    } else if dark {
        "clock#finish611"
    } else {
        "clock#finish610"
    }
}

fn material(
    o: &IndoorObject,
    p: &ClockProgram,
    dial: PrintedDial,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut ink = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"DialInks"),
    );
    let (bg, fg) = match dial.style {
        DialStyle::ReflectiveLcd => (
            [
                [174, 183, 153],
                [182, 184, 168],
                [189, 177, 145],
                [172, 186, 184],
            ][ink.random_range(0..4)],
            [40, 47, 42],
        ),
        DialStyle::Led => (
            [19, 22, 23],
            [
                [252, 91, 60],
                [255, 172, 54],
                [126, 236, 193],
                [223, 232, 244],
            ][ink.random_range(0..4)],
        ),
        _ if p.dark_dial => (
            [[40, 46, 50], [40, 50, 60], [57, 59, 55], [46, 54, 49]][ink.random_range(0..4)],
            [228, 226, 212],
        ),
        _ => (
            [
                [235, 233, 220],
                [223, 227, 226],
                [229, 219, 199],
                [234, 235, 236],
            ][ink.random_range(0..4)],
            [37, 43, 47],
        ),
    };
    let mut color = raster::Canvas::new(bg);
    let mut emission = (dial.style == DialStyle::Led).then(|| raster::Canvas::new([0; 3]));
    if p.digital {
        print::digital(&mut color, emission.as_mut(), o, p, dial, bg, fg);
    } else {
        print::analog(&mut color, o, p, dial, fg);
    }
    let mut texture = |pixels: raster::Canvas| {
        let mut image = mip_image(pixels.0, raster::N as u32, MapType::Color);
        if let ImageSampler::Descriptor(s) = &mut image.sampler {
            s.address_mode_u = ImageAddressMode::ClampToEdge;
            s.address_mode_v = ImageAddressMode::ClampToEdge;
        }
        images.add(image)
    };
    let strength = if emission.is_some() {
        ink.random_range(45.0..175.)
    } else {
        0.
    };
    StandardMaterial {
        base_color_texture: Some(texture(color)),
        perceptual_roughness: ink.random_range(if p.digital { 0.30..0.49 } else { 0.53..0.77 }),
        reflectance: if p.digital { 0.44 } else { 0.36 },
        clearcoat: if p.digital { 0.26 } else { 0.11 },
        clearcoat_perceptual_roughness: ink.random_range(0.28..0.48),
        emissive_texture: emission.map(texture),
        emissive: LinearRgba::rgb(strength, strength, strength),
        emissive_exposure_weight: 1.,
        ..default()
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    mats: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if selection.is_some_and(|s| {
        !s.finishes
            .iter()
            .any(|&(surface, slot)| supports(surface, slot))
    }) {
        return;
    }
    let mut hands = [false; 3];
    for o in scene.objects.iter().filter(|o| o.kind == ObjectKind::Clock) {
        let p = clocks::parameters(o);
        let Some(dial) = p.printed else { continue };
        let key = (Surface::Paper, INSTANCE + o.id);
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            variants.insert(key, mats.add(material(o, &p, dial, images)));
        }
        if !p.digital {
            hands[usize::from(p.dark_dial)] = true;
            hands[2] |= p.second_hand;
        }
    }
    for (i, color) in [[38, 40, 43], [230, 227, 215], [202, 59, 38]]
        .into_iter()
        .enumerate()
    {
        let key = (Surface::Ink, CHARCOAL + i);
        if selection.map_or(hands[i], |s| s.finishes.contains(&key)) {
            variants.insert(
                key,
                mats.add(StandardMaterial {
                    base_color: Color::srgb_u8(color[0], color[1], color[2]),
                    perceptual_roughness: 0.48,
                    ..default()
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};

    #[test]
    fn clock_digit_emission_matches_only_inked_segments_and_every_face_is_filtered() {
        let mut visited = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let o = audit::fixture(ObjectKind::Clock, seed);
            let p = clocks::parameters(&o);
            let d = p.printed.unwrap();
            if !visited.insert(format!("{:?}", d.style)) {
                continue;
            }
            let mut images = Assets::default();
            let m = material(&o, &p, d, &mut images);
            let color = images.get(m.base_color_texture.unwrap().id()).unwrap();
            assert_eq!(color.texture_descriptor.mip_level_count, 9);
            assert_eq!(
                color.texture_descriptor.format,
                TextureFormat::Rgba8UnormSrgb
            );
            assert_eq!(m.metallic, 0.);
            let bytes = &color.data.as_ref().unwrap()[..256 * 256 * 4];
            assert!(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .any(|k| k[..3] != bytes[..3]));
            if d.style == DialStyle::Led {
                let glow = images.get(m.emissive_texture.unwrap().id()).unwrap();
                assert_eq!(glow.texture_descriptor.mip_level_count, 9);
                let glow = &glow.data.as_ref().unwrap()[..bytes.len()];
                let mut on = 0;
                for (a, b) in bytes.as_chunks::<4>().0.iter().zip(glow.as_chunks::<4>().0) {
                    let lit = b[..3] != [0, 0, 0];
                    assert_eq!(lit, a[..3].iter().max().unwrap() > &100);
                    if lit {
                        on += 1;
                        assert_eq!(a, b);
                    }
                }
                assert!((250..14000).contains(&on));
            } else {
                assert!(m.emissive_texture.is_none() && m.emissive.red == 0.);
            }
        }
        assert_eq!(visited.len(), 6);
    }

    #[test]
    fn selected_clock_faces_follow_object_ids_and_unused_instances_cost_no_images() {
        let mut scene = IndoorManifest::generate(11, IndoorLayout::Mixed, 0.65, 0).unwrap();
        let mut a = audit::fixture(ObjectKind::Clock, 55);
        a.id = 17;
        let mut b = audit::fixture(ObjectKind::Clock, 56);
        b.id = 43;
        scene.objects = vec![b.clone(), a.clone()];
        let key = (Surface::Paper, INSTANCE + b.id);
        let selected = MaterialSelection {
            surfaces: [Surface::Paper].into(),
            finishes: [key].into(),
            direct_surfaces: default(),
        };
        assert!(registered_finish(key.0, key.1) && selected.omits_parent_maps(&scene, key.0));
        let (mut images, mut mats) = (
            Assets::<Image>::default(),
            Assets::<StandardMaterial>::default(),
        );
        let mut variants = BTreeMap::new();
        install(
            &scene,
            &mut images,
            &mut mats,
            Some(&selected),
            &mut variants,
        );
        assert_eq!(variants.keys().copied().collect::<Vec<_>>(), [key]);
        assert_eq!(
            images.len(),
            if clocks::parameters(&b).printed.unwrap().style == DialStyle::Led {
                2
            } else {
                1
            }
        );
        let count = images.len();
        let empty = MaterialSelection {
            surfaces: default(),
            finishes: default(),
            direct_surfaces: default(),
        };
        install(&scene, &mut images, &mut mats, Some(&empty), &mut variants);
        assert_eq!(images.len(), count);
    }
}
