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
    // Center/border coordinates must not inherit upholstery rotations/phases.
    r.phase = [0.; 2];
    r.layers = None;
    let dye = r.carpet.as_mut().unwrap();
    // Pigmented wool/dyed yarn palettes: mostly earth, ink and mineral hues,
    // with a small full-hue tail. Do not equate diversity with fluorescent dyes.
    let choice = rng.random_range(0.0..1.);
    let hue = rng.random_range(if choice < 0.5 {
        12.0..60.0
    } else if choice < 0.76 {
        190.0..245.0
    } else if choice < 0.9 {
        75.0..140.0
    } else {
        0.0..360.0
    });
    let saturation = rng.random_range(0.035..0.34);
    for (i, c) in dye.palette.iter_mut().enumerate() {
        let v = Color::hsl(
            hue + if i == 1 {
                rng.random_range(5.0..40.)
            } else {
                rng.random_range(-18.0..18.)
            },
            saturation * if i == 1 { 0.45 } else { 1. },
            rng.random_range(if i == 1 {
                0.46..0.78
            } else if i == 2 {
                0.08..0.27
            } else {
                0.18..0.56
            }),
        )
        .to_srgba();
        *c = [v.red, v.green, v.blue];
    }
    if parent.construction.is_some() {
        // Rugs keep their whole-field composition and fixed atlas count; only
        // pile/finish construction is independently sampled for the rug slot.
        let inherited_relief = r.relief_m;
        super::construction::sample(&mut r, 1);
        // Preserve direct parent relief overrides as well as material_detail.
        // The slot's construction varies pile height without reviving flat rugs.
        let p = r.construction.as_ref().unwrap();
        r.relief_m = inherited_relief
            * (0.65 + p.weights[1] * 0.55 + p.weights[2] * 0.25 + p.weights[3] * 0.85);
    }
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
            // Whole-rug field UVs are produced by floor::rug; bindings retain metric UVs.
            uv_transform: bevy::math::Affine2::IDENTITY,
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
    fn whole_rug_borders_clamp_in_cpu_and_gpu_backends_while_floor_carpet_repeats() {
        let recipes = program::sample_with_floor(7, 1).unwrap();
        let r = recipe(&recipes[Surface::FabricAlt as usize], 0);
        for gpu in [false, true] {
            for (recipe, expected) in [
                (&r, ImageAddressMode::ClampToEdge),
                (&recipes[Surface::Floor as usize], ImageAddressMode::Repeat),
            ] {
                let mut images = Assets::<Image>::default();
                let handles = gpu::Maps::prepare(recipe, 1, gpu)
                    .insert(&mut images, &mut gpu::Requests::default());
                for handle in handles {
                    let ImageSampler::Descriptor(s) = &images.get(&handle).unwrap().sampler else {
                        panic!("missing material sampler")
                    };
                    assert_eq!(s.address_mode_u, expected);
                    assert_eq!(s.address_mode_v, expected);
                }
            }
        }
    }
    #[test]
    fn whole_rug_programs_replay_cover_components_and_preserve_old_carpet_recipes() {
        let mut dominants = [0; 4];
        let (mut bordered, mut centered, mut quiet) = (0, 0, 0);
        for seed in 0..512 {
            let parent = program::sample(seed)[Surface::FabricAlt as usize].clone();
            let r = recipe(&parent, slot(seed));
            assert_eq!(r, recipe(&parent, slot(seed)));
            r.validate().unwrap();
            assert_eq!(r.phase, [0.; 2]);
            assert!(r.layers.is_none());
            let c = r.carpet.as_ref().unwrap();
            let d = c.rug.as_ref().unwrap();
            let index = d
                .weights
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            dominants[index] += 1;
            bordered += usize::from(d.border_width > 0.);
            centered += usize::from(d.center > 0.);
            quiet += usize::from(d.quiet > 0.7);
            for y in 0..8 {
                for x in 0..8 {
                    let t = r.texel((x as f32 + 0.5) / 8., (y as f32 + 0.5) / 8., 0);
                    assert!(t
                        .color
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.).contains(v)));
                    assert!(t.height.is_finite() && t.height.abs() <= r.relief_m);
                    assert!((0.8..=1.).contains(&t.roughness));
                }
            }
            if seed == 0 {
                let mut old = serde_json::to_value(c).unwrap();
                old.as_object_mut().unwrap().remove("rug");
                let old: carpet::CarpetDesign = serde_json::from_value(old).unwrap();
                assert!(old.rug.is_none());
                old.validate().unwrap();
            }
        }
        assert!(dominants.into_iter().all(|n| n > 65), "{dominants:?}");
        assert!(
            bordered > 300
                && bordered < 480
                && centered > 170
                && centered < 340
                && quiet > 45
                && quiet < 150
        );
    }
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
