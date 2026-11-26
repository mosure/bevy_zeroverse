use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorLayout,
    objects::{audit, build_object},
};
use sha2::{Digest, Sha256};

#[test]
fn selected_stock_binds_repeatable_shared_ink_and_mip_filtered_press_relief() {
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut selection = MaterialSelection {
        surfaces: Default::default(),
        direct_surfaces: Default::default(),
        finishes: Default::default(),
    };
    let bake = |s: &IndoorManifest, selection: Option<&MaterialSelection>| {
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let mut variants = BTreeMap::new();
        install(s, &mut images, &mut materials, selection, &mut variants);
        let mut receipts = BTreeMap::new();
        let mut sheets = BTreeMap::new();
        let mut presses = BTreeMap::new();
        for (&(surface, slot), handle) in &variants {
            assert!(supports(surface, slot) && registered_finish(surface, slot));
            let stock = Stock::from_slot(slot);
            let m = materials.get(handle).unwrap();
            assert!(
                m.metallic == 0.
                    && m.clearcoat == 0.
                    && (0.70..0.95).contains(&m.perceptual_roughness)
            );
            let albedo = m.base_color_texture.as_ref().unwrap();
            let relief = m.normal_map_texture.as_ref().unwrap();
            assert_eq!(
                sheets
                    .entry((stock.lot, stock.printed))
                    .or_insert(albedo.id()),
                &albedo.id()
            );
            assert_eq!(
                presses.entry(stock.press).or_insert(relief.id()),
                &relief.id()
            );
            let maps = [
                (albedo, 9, TextureFormat::Rgba8UnormSrgb),
                (relief, 8, TextureFormat::Rgba8Unorm),
            ];
            for (id, mips, format) in maps {
                let im = images.get(id).unwrap();
                assert_eq!(im.texture_descriptor.format, format);
                assert_eq!(im.texture_descriptor.mip_level_count, mips);
                let ImageSampler::Descriptor(s) = &im.sampler else {
                    panic!("missing chart sampler")
                };
                assert_eq!(s.address_mode_u, ImageAddressMode::Repeat);
                assert_eq!(s.address_mode_v, ImageAddressMode::ClampToEdge);
                assert_eq!(s.mipmap_filter, ImageFilterMode::Linear);
            }
            let normal = images.get(relief).unwrap().data.as_ref().unwrap();
            let far = &normal[normal.len() - 4..];
            assert!(far[0].abs_diff(128) <= 1 && far[1].abs_diff(128) <= 1 && far[2] >= 254);
            let hashes: [[u8; 32]; 2] = [albedo, relief]
                .map(|id| Sha256::digest(images.get(id).unwrap().data.as_ref().unwrap()).into());
            receipts.insert(slot, hashes);
        }
        assert_eq!(sheets.len() + presses.len(), images.len());
        receipts
    };
    assert!(bake(&scene, None).is_empty());
    for seed in 0..192 {
        let mut cup = audit::fixture(ObjectKind::CoffeeCup, seed);
        cup.finish_seed = Some(seed);
        for (s, tag) in build_object(&cup).parts.into_keys() {
            if let Some(i) = finish_slot(s, &tag).filter(|&i| supports(s, i)) {
                selection.finishes.insert((s, i));
            }
        }
        scene.objects.push(cup);
    }
    let full = bake(&scene, None);
    assert_eq!(full.len(), LOTS * PRESSES.len() * 2);
    assert_eq!(full, bake(&scene, Some(&selection)));
    assert_eq!(
        full.values().map(|h| h[0]).collect::<BTreeSet<_>>().len(),
        LOTS * 2
    );
    assert_eq!(
        full.values().map(|h| h[1]).collect::<BTreeSet<_>>().len(),
        PRESSES.len()
    );
    selection.finishes.retain(|(_, slot)| *slot == FIRST + 5);
    assert_eq!(
        bake(&scene, Some(&selection)),
        BTreeMap::from([(FIRST + 5, full[&(FIRST + 5)])])
    );
    scene.generator_version = INTRODUCED - 1;
    assert!(bake(&scene, None).is_empty() && bake(&scene, Some(&selection)).is_empty());
}
