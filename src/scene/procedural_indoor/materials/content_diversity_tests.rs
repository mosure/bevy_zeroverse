use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn content_regions_remain_inside_the_document_and_do_not_overlap() {
    for seed in 0..1024 {
        for bounds in [[0.02, 0.03, 0.95, 0.90], [0.1, 0.1, 0.06, 0.08]] {
            for c in [
                composition::Composition::editorial(seed, 12, bounds, 0.5),
                composition::Composition::clusters(seed, 12, bounds),
            ] {
                assert!(!c.regions.is_empty() && c.regions.len() <= 12);
                for (i, r) in c.regions.iter().enumerate() {
                    let [x, y, w, h] = r.bounds;
                    assert!(w > 0. && h > 0. && x >= bounds[0] && y >= bounds[1]);
                    assert!(
                        x + w <= bounds[0] + bounds[2] + 1e-6
                            && y + h <= bounds[1] + bounds[3] + 1e-6
                    );
                    for other in &c.regions[..i] {
                        let [a, b, d, e] = other.bounds;
                        assert!(x + w <= a || a + d <= x || y + h <= b || b + e <= y);
                    }
                }
            }
        }
    }
}

#[test]
fn all_document_grammars_obey_nested_clips() {
    for seed in 0..12 {
        for kind in 0..content::KINDS {
            let mut c = raster::Canvas::new([1, 2, 3]);
            c.clipped([19, 23, 91, 53], |c| {
                content::panel(c, seed, [12, 12, 137, 101], kind, seed % 2 == 0)
            });
            assert_eq!(c.0.len(), 256 * 256 * 4);
            for (i, pixel) in c.0.as_chunks::<4>().0.iter().enumerate() {
                assert_eq!(pixel[3], 255);
                let (x, y) = (i % 256, i / 256);
                if !(19..110).contains(&x) || !(23..76).contains(&y) {
                    assert_eq!(pixel, &[1, 2, 3, 255]);
                }
            }
        }
    }
}

#[test]
fn responsive_devices_and_boards_have_continuous_layout_and_type_ranges() {
    let mut counts = std::collections::BTreeSet::new();
    let mut margins = std::collections::BTreeSet::new();
    let mut headings = std::collections::BTreeSet::new();
    for seed in 0..512 {
        let b = boards::parameters(seed);
        headings.insert((b.letter_height * 4.) as i32);
        for kind in [
            screens::DisplayKind::Desktop,
            screens::DisplayKind::Phone,
            screens::DisplayKind::Television,
        ] {
            let p = screens::parameters_for(seed, kind);
            counts.insert(p.composition.regions.len());
            margins.insert((p.composition.regions[0].bounds[0] * 1000.) as u32);
            assert_eq!(
                serde_json::to_value(&p).unwrap(),
                serde_json::to_value(screens::parameters_for(seed, kind)).unwrap()
            );
            for r in &p.composition.regions {
                let [x, y, w, h] = r.bounds;
                assert!(
                    w > 0. && h > 0. && x >= 0. && y >= 0. && x + w <= 1.00001 && y + h <= 1.00001
                );
            }
        }
    }
    assert!(counts.len() >= 8 && margins.len() > 100 && headings.len() > 20);
}

#[test]
fn expanded_content_finishes_resolve_without_baking_unused_slots() {
    let scene = IndoorManifest::generate_with_humans(7, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut images = Assets::<Image>::default();
    let mut materials = Assets::<StandardMaterial>::default();
    let label = format!("picture#finish{}", variants::CONTENT_COUNT - 1);
    let set = IndoorMaterials::build_for_parts(
        &scene,
        &mut images,
        &mut materials,
        [(Surface::Art, label.as_str())],
    );
    assert_eq!(
        set.variants
            .keys()
            .filter(|(s, _)| variants::is_content(*s))
            .count(),
        1
    );
    let handle = set.for_part(Surface::Art, &label);
    assert_ne!(handle, set.get(Surface::Art));
    let mat = materials.get(&handle).unwrap();
    let image = images
        .get(mat.base_color_texture.as_ref().unwrap())
        .unwrap();
    assert_eq!(image.texture_descriptor.mip_level_count, 9);
    let expected = art::pixels(variants::screen_seed(
        scene.material_seed(),
        variants::CONTENT_COUNT - 1,
    ));
    assert_eq!(&image.data.as_ref().unwrap()[..256 * 256 * 4], expected);
    assert_eq!(mat.metallic, 0.);
    assert_eq!(mat.emissive, LinearRgba::BLACK);
}

#[test]
fn upper_seed_bits_change_content_not_only_substrate_noise() {
    for seed in 0..8 {
        assert_ne!(art::illustration(seed), art::illustration(seed + (1 << 32)));
        assert_ne!(boards::pixels(seed), boards::pixels(seed + (1 << 32)));
        assert_ne!(screens::pixels(seed), screens::pixels(seed + (1 << 32)));
        assert_ne!(
            screens::mobile_pixels(seed),
            screens::mobile_pixels(seed + (1 << 32))
        );
    }
}
