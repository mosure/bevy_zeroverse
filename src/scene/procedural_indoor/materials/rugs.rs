//! Rugs own their dye/pile program instead of inheriting chair upholstery.
//! Four shared slots bound texture residency, independent of rug count.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

pub const COUNT: usize = 4;
pub fn slot(seed: u64) -> usize {
    stream(seed, 0x525547534c4f54).random_range(0..COUNT)
}
/// (palette slot, binding rather than patterned field).
pub fn key(label: &str) -> Option<(usize, bool)> {
    let (suffix, binding) = if let Some((_, s)) = label.rsplit_once("#rug_binding") {
        (s, true)
    } else {
        (label.rsplit_once("#rug")?.1, false)
    };
    let slot = suffix.parse::<usize>().ok()?;
    (slot < COUNT).then_some((slot, binding))
}
/// Internal slots follow the ordinary upholstery palette. The label remains
/// typed, while rendering and diffuse transport share one material registry.
pub(crate) fn finish_slot(slot: usize, binding: bool) -> usize {
    variants::COUNT + slot * 2 + usize::from(binding)
}
pub fn recipe(parent: &program::MaterialRecipe, slot: usize) -> program::MaterialRecipe {
    let mut r = parent.clone();
    r.seed = stream(parent.seed, 0x525547445945 + slot as u64).random();
    let mut rng = stream(r.seed, 0x50494c45);
    let tone = rng.random_range(0.16..0.68);
    let warm = rng.random_range(-0.025..0.09);
    let neutral = [tone, tone * (1. - warm), tone * (1. - warm * 1.6)];
    let chroma = if rng.random_bool(0.65) {
        0.15
    } else {
        rng.random_range(0.35..0.70)
    };
    r.color = [0, 1, 2].map(|i| neutral[i] * (1. - chroma) + r.color[i] * chroma);
    r.period_m = rng.random_range(1.6..3.2);
    // The parent already carries AppearanceSettings::material_detail. Keep
    // that control (including flat-detail ablations) on the derived rug too.
    r.relief_m = parent.relief_m * rng.random_range(1.0..3.0);
    r.roughness = rng.random_range(0.86..0.98);
    r.carpet = Some(carpet::CarpetDesign::sample(r.seed, r.color, true));
    r
}

pub(super) async fn build(
    scene: &IndoorManifest,
    parent: &Handle<StandardMaterial>,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    gpu: bool,
    requests: &mut gpu::Requests,
    cooperative: bool,
) -> BTreeMap<usize, [Handle<StandardMaterial>; 2]> {
    let mut result = BTreeMap::new();
    let Some(program) = &scene.program else {
        return result;
    };
    let slots: BTreeSet<_> = scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::Rug)
        .map(|o| slot(o.finish_seed.unwrap_or(o.seed)))
        .collect();
    for slot in slots {
        let r = recipe(&program.materials[Surface::FabricAlt as usize], slot);
        let [color, normal, data] =
            gpu::Maps::prepare(&r, scene.floor_style, gpu).insert(images, requests);
        let mut mat = StandardMaterial {
            base_color: Color::WHITE,
            base_color_texture: Some(color),
            normal_map_texture: Some(normal),
            occlusion_texture: Some(data.clone()),
            metallic_roughness_texture: Some(data),
            perceptual_roughness: 1.,
            uv_transform: bevy::math::Affine2::from_scale(r.period_uv().recip()),
            ..default()
        };
        r.apply_pbr(&mut mat);
        let mut binding = materials.get(parent).unwrap().clone();
        let color = r.carpet.as_ref().unwrap().palette[0];
        let tone = 0.65 + hash(29, 13, r.seed) * 0.30;
        binding.base_color = Color::srgb(color[0] * tone, color[1] * tone, color[2] * tone);
        result.insert(slot, [materials.add(mat), materials.add(binding)]);
        if cooperative {
            super::super::preparation::cooperate().await;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derived_pile_respects_the_recorded_material_detail_factor() {
        for seed in 0..32 {
            let mut parent = program::sample(seed)[Surface::FabricAlt as usize].clone();
            let full = recipe(&parent, slot(seed));
            parent.relief_m *= 0.5;
            assert_eq!(recipe(&parent, slot(seed)).relief_m, full.relief_m * 0.5);
            parent.relief_m = 0.;
            assert_eq!(recipe(&parent, slot(seed)).relief_m, 0.);
        }
    }
    #[test]
    fn field_and_binding_resolve_identically_for_raster_and_transport_with_bounded_residency() {
        use crate::scene::procedural_indoor::{
            layout::IndoorLayout, objects::audit, IndoorQuality,
        };
        let mut scene =
            IndoorManifest::generate_with_humans(19, IndoorLayout::Conference, 0., 0, 0.).unwrap();
        scene.objects = (0..24)
            .map(|seed| {
                let mut o = audit::fixture(ObjectKind::Rug, seed);
                o.finish_seed = Some(seed);
                o
            })
            .collect();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let selection = MaterialSelection {
            surfaces: [Surface::FabricAlt].into(),
            finishes: Default::default(),
            direct_surfaces: [Surface::FabricAlt].into(),
        };
        let set = IndoorMaterials::build_with_selection(
            &scene,
            IndoorQuality::Auto,
            &mut images,
            &mut materials,
            Some(&selection),
            true,
        );
        // 24 rugs still use only four GPU color/normal/ORM atlases, plus
        // one shared upholstery atlas for their metric woven bindings.
        assert_eq!(set.gpu_bakes.statistics().gpu_materials, COUNT + 1);
        for slot in 0..COUNT {
            let mut resolved = Vec::new();
            for binding in [false, true] {
                let label = format!(
                    "floormat#rug{}{slot}",
                    if binding { "_binding" } else { "" }
                );
                let key = super::super::finish_slot(Surface::FabricAlt, &label).unwrap();
                let handle = set.for_part(Surface::FabricAlt, &label);
                assert_eq!(handle, set.variants[&(Surface::FabricAlt, key)]);
                let mat = materials.get(&handle).unwrap();
                assert!(mat.base_color_texture.is_some() && mat.normal_map_texture.is_some());
                resolved.push(handle);
            }
            assert_ne!(resolved[0], resolved[1]);
        }
        assert!(key("floormat#rug4").is_none());
        assert!(key("floormat#rug_bindingbad").is_none());
    }
}
