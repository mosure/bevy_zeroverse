//! One exterior skin for the dry resin and the water-backed part of a carboy.
//! A separate water/air interface closes the fluid at its sampled level.
use super::GlassRecipe;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorManifest, ObjectKind},
    materials::{MaterialSelection, Surface},
    objects::public::cooler::{Cooler, Loading},
    preparation::AssetStore,
    IndoorQuality,
};
use bevy::prelude::*;
use rand::Rng;
use std::collections::BTreeMap;

const DRY_SLOT: usize = 1061;
const WET_SLOT: usize = 1062;
const CAP_SLOT: usize = 1063;
pub(crate) const CAP: &str = "drink_container#finish1063";
pub(crate) const DRY: &str = "drink_container#finish1061";
pub(crate) const WET: &str = "drink_container#finish1062";

pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    (surface == Surface::ContainerGlass && matches!(slot, DRY_SLOT | WET_SLOT))
        || (surface == Surface::Plastic && slot == CAP_SLOT)
}
fn material(seed: u64, water: bool, quality: IndoorQuality) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"BottlePC"));
    let mut m = StandardMaterial::default();
    let blue: f32 = r.random_range(0.14..0.45);
    let recipe = GlassRecipe {
        roughness: r.random_range(0.04..0.115),
        ior: 1.575,
        thickness_m: if water { 0.115 } else { 0.0015 },
        attenuation: [1. - blue, 1. - blue * 0.37, 1. - blue * 0.065],
        attenuation_distance_m: if water { 0.13 } else { 0.0033 },
    };
    recipe.apply(&mut m, quality);
    if !quality.specular_transmission() {
        m.base_color = m.base_color.with_alpha(if water { 0.269 } else { 0.142 });
    }
    m
}
pub(crate) fn install(
    scene: &IndoorManifest,
    quality: IndoorQuality,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < crate::scene::procedural_indoor::objects::public::cooler::VERSION {
        return;
    }
    let mut unfiltered = [false; 2];
    if selection.is_none() {
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::WaterCooler)
        {
            let p = Cooler::for_object(o);
            if p.loading == Loading::TopBottle {
                unfiltered[0] = true;
                unfiltered[1] |= p.bottle.is_some_and(|b| b.fill_fraction > 0.);
            }
        }
    }
    let cap_key = (Surface::Plastic, CAP_SLOT);
    if selection.map_or(unfiltered[0], |s| s.finishes.contains(&cap_key)) {
        let mut rng = stream(scene.material_seed(), u64::from_le_bytes(*b"SealCap3"));
        let tint = if rng.random_bool(0.22) {
            Color::srgb(0.83, 0.86, 0.85)
        } else {
            Color::srgb(
                0.075,
                rng.random_range(0.22..0.41),
                rng.random_range(0.60..0.76),
            )
        };
        variants.insert(
            cap_key,
            materials.add(StandardMaterial {
                base_color: tint,
                perceptual_roughness: rng.random_range(0.27..0.45),
                ..default()
            }),
        );
    }
    for (i, slot) in [DRY_SLOT, WET_SLOT].into_iter().enumerate() {
        let key = (Surface::ContainerGlass, slot);
        if selection.map_or(unfiltered[i], |s| s.finishes.contains(&key)) {
            variants.insert(
                key,
                materials.add(material(scene.material_seed(), i == 1, quality)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::glass::AnnotationGlass;
    #[test]
    fn cooler_resin_is_transmissive_in_both_renderers_and_obeys_annotation_policy() {
        for (label, slot, wet) in [(DRY, DRY_SLOT, false), (WET, WET_SLOT, true)] {
            use crate::scene::procedural_indoor::materials::{finish_slot, registered_finish};
            assert_eq!(finish_slot(Surface::ContainerGlass, label), Some(slot));
            assert!(registered_finish(Surface::ContainerGlass, slot));
            for quality in [IndoorQuality::Auto, IndoorQuality::Portable] {
                let m = material(2, wet, quality);
                assert_eq!(
                    m.specular_transmission > 0.,
                    quality.specular_transmission()
                );
                assert!(m.normal_map_texture.is_none() && m.base_color_texture.is_none());
                assert!(!AnnotationGlass::Through.includes(Some(&m)));
                assert!(AnnotationGlass::Surface.includes(Some(&m)));
                if !quality.specular_transmission() {
                    let rgba = m.base_color.to_linear();
                    assert!(rgba.blue > rgba.red && rgba.alpha > 0. && rgba.alpha < 0.32);
                }
            }
        }
    }
}
