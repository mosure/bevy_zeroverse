use super::*;
use crate::scene::procedural_indoor::{
    layout::IndoorLayout,
    objects::{audit, build_object},
};
use sha2::{Digest, Sha256};

#[test]
fn visible_stock_selects_the_same_filtered_assets_in_both_material_paths() {
    let mut scene =
        IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut selection = MaterialSelection {
        surfaces: Default::default(),
        finishes: Default::default(),
        direct_surfaces: Default::default(),
    };
    let build = |s: &IndoorManifest, selection: Option<&MaterialSelection>| {
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let mut variants = BTreeMap::new();
        install(s, &mut images, &mut materials, selection, &mut variants);
        let mut maps = BTreeMap::new();
        let mut count = 0;
        for (key, handle) in variants {
            let mat = materials.get(&handle).unwrap();
            assert_eq!(mat.metallic, 0.);
            assert!(mat.perceptual_roughness >= 0.57);
            let fiber = key.1 == FIBER;
            let textures = [
                &mat.base_color_texture,
                &mat.normal_map_texture,
                &mat.metallic_roughness_texture,
            ];
            let mut hashes = Vec::new();
            for (i, tex) in textures.into_iter().enumerate() {
                let Some(handle) = tex else {
                    assert!(!fiber);
                    continue;
                };
                let im = images.get(handle).unwrap();
                assert_eq!(
                    im.texture_descriptor.mip_level_count,
                    if fiber { 6 } else { 8 }
                );
                assert_eq!(
                    im.texture_descriptor.format,
                    if i == 0 {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    }
                );
                let ImageSampler::Descriptor(sampler) = &im.sampler else {
                    panic!("expected filtered sampler")
                };
                assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
                let bytes = im.data.as_ref().unwrap();
                if i == 2 {
                    assert!(bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|p| p[1] >= 217 && p[2] == 0));
                }
                hashes.push(<[u8; 32]>::from(Sha256::digest(bytes)));
            }
            count += hashes.len();
            maps.insert(key, (mat.base_color.to_srgba().to_u8_array(), hashes));
        }
        assert_eq!(count, images.len());
        maps
    };
    assert!(build(&scene, None).is_empty());
    for seed in 0..128 {
        let mut o = audit::fixture(ObjectKind::TissueBox, seed);
        o.finish_seed = Some(seed);
        for (surface, label) in build_object(&o).parts.into_keys() {
            if let Some(i) = finish_slot(surface, &label).filter(|i| supports(surface, *i)) {
                assert!(registered_finish(surface, i));
                selection.finishes.insert((surface, i));
            }
        }
        scene.objects.push(o);
    }
    let all = build(&scene, None);
    assert_eq!(all.len(), LOTS + 2);
    assert_eq!(all, build(&scene, Some(&selection)));
    selection.finishes = BTreeSet::from([(Surface::Paper, FIBER)]);
    assert_eq!(
        build(&scene, Some(&selection))[&(Surface::Paper, FIBER)],
        all[&(Surface::Paper, FIBER)]
    );
    let families: BTreeSet<_> = (0..80)
        .map(|seed| print::pattern(seed, FIRST) as u8)
        .collect();
    assert_eq!(families.len(), 4);
    let sealed = scene
        .objects
        .iter()
        .find(|o| {
            tissue::parameters(o)
                .dispenser
                .is_some_and(|d| d.draw == Draw::Sealed)
        })
        .unwrap()
        .clone();
    scene.objects = vec![sealed];
    let only = build(&scene, None);
    assert_eq!(only.len(), 1);
    assert!(only.contains_key(&(Surface::Paper, slot(&scene.objects[0]))));
}
