use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorLayout,
    objects::{audit, build_object},
};
use sha2::{Digest, Sha256};

#[test]
fn installed_packaging_matches_consumed_parts_and_replays_filtered_ink_per_lot() {
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut sel = MaterialSelection {
        surfaces: Default::default(),
        finishes: Default::default(),
        direct_surfaces: Default::default(),
    };
    let bake = |s: &IndoorManifest, sel: Option<&MaterialSelection>| {
        let mut images = Assets::default();
        let mut mats = Assets::default();
        let mut variants = BTreeMap::new();
        install(s, &mut images, &mut mats, sel, &mut variants);
        assert_eq!(mats.len(), variants.len());
        let mut maps = BTreeMap::new();
        for (&key, h) in &variants {
            let mat = mats.get(h).unwrap();
            assert_eq!(mat.metallic > 0.9, key == (Surface::Metal, ALLOY));
            if let Some(texture) = mat.base_color_texture.as_ref() {
                let im = images.get(texture).unwrap();
                assert_eq!(im.texture_descriptor.format, TextureFormat::Rgba8UnormSrgb);
                assert_eq!(im.texture_descriptor.mip_level_count, 9);
                let ImageSampler::Descriptor(sampler) = &im.sampler else {
                    panic!("missing repeat sampler")
                };
                assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
                assert_eq!(sampler.address_mode_v, ImageAddressMode::ClampToEdge);
                assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
                assert!(mat.clearcoat > 0. && mat.metallic == 0.);
                let bytes = im.data.as_ref().unwrap();
                assert!(bytes.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
                maps.insert(key, <[u8; 32]>::from(Sha256::digest(bytes)));
            }
        }
        assert_eq!(maps.len(), images.len());
        (variants.into_keys().collect::<BTreeSet<_>>(), maps)
    };
    assert_eq!(bake(&scene, None).0.len(), 0);
    for seed in 0..128 {
        for kind in [ObjectKind::SodaCan, ObjectKind::WaterBottle] {
            let mut o = audit::fixture(kind, seed);
            o.finish_seed = Some(seed);
            for (s, l) in build_object(&o).parts.into_keys() {
                if let Some(i) = finish_slot(s, &l).filter(|i| supports(s, *i)) {
                    assert!(registered_finish(s, i));
                    sel.finishes.insert((s, i));
                }
            }
            scene.objects.push(o);
        }
    }
    let full = bake(&scene, None);
    assert_eq!(full.0, sel.finishes);
    assert_eq!(full, bake(&scene, Some(&sel)));
    let unique: BTreeSet<_> = full.1.values().copied().collect();
    assert_eq!(unique.len(), 2 * LOTS);
    let mut retail_finishes = BTreeSet::new();
    scene.objects.clear();
    for seed in 0..20 {
        let mut fixture = audit::fixture(ObjectKind::GroceryDisplay, seed);
        fixture.size.y = 1.48;
        if seed % 2 == 1 {
            fixture.variant |=
                crate::scene::procedural_indoor::objects::public::retail::REFRIGERATED;
        }
        for (surface, label) in build_object(&fixture).parts.into_keys() {
            if let Some(i) = finish_slot(surface, &label).filter(|i| supports(surface, *i)) {
                retail_finishes.insert((surface, i));
            }
        }
        scene.objects.push(fixture);
    }
    assert!(retail_finishes.iter().any(|(s, _)| *s == Surface::Paper));
    sel.finishes = retail_finishes;
    let retail = bake(&scene, None);
    assert_eq!(retail.0, sel.finishes);
    assert_eq!(retail, bake(&scene, Some(&sel)));
    let key = *full.1.keys().next().unwrap();
    sel.finishes = BTreeSet::from([key]);
    let single = bake(&scene, Some(&sel));
    assert_eq!(single.0, sel.finishes);
    assert_eq!(single.1, BTreeMap::from([(key, full.1[&key])]));
}
