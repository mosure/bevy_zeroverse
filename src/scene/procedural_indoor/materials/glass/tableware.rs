//! Molded table glass and its water-backed inner wall share a small optical
//! palette. A serving adds a water/air interface only when it contains water.
use super::GlassRecipe;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorManifest, ObjectKind},
    materials::{MaterialSelection, Surface},
    preparation::AssetStore,
    IndoorQuality,
};
use bevy::prelude::*;
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 1160;
pub(crate) const IMMERSED: &str = "drink_container#finish1163";

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) enum Tint {
    Flint,
    Aqua,
    Amber,
}
impl Tint {
    pub(crate) fn sample(r: &mut impl Rng) -> Self {
        match r.random_range(0..20) {
            0..=2 => Self::Aqua,
            3..=4 => Self::Amber,
            _ => Self::Flint,
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Flint => "drink_container#finish1160",
            Self::Aqua => "drink_container#finish1161",
            Self::Amber => "drink_container#finish1162",
        }
    }
}
pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::ContainerGlass && (FIRST..=FIRST + 3).contains(&slot)
}

fn material(seed: u64, finish: usize, quality: IndoorQuality) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"TableIOR") ^ finish as u64);
    let glass_ior = 1.52;
    let mut m = StandardMaterial::default();
    let attenuation = match finish {
        1 => [0.49, 0.87, 0.81],
        2 => [0.90, 0.60, 0.29],
        _ => [0.95, 0.99, 0.985],
    };
    let recipe = GlassRecipe {
        // The inner face below the meniscus borders water rather than air.
        ior: if finish == 3 {
            glass_ior / 1.333
        } else {
            glass_ior
        },
        roughness: r.random_range(0.014..0.042),
        thickness_m: 0.002,
        attenuation,
        attenuation_distance_m: r.random_range(0.0028..0.0060),
    };
    recipe.apply(&mut m, quality);
    if !quality.specular_transmission() {
        m.base_color = m.base_color.with_alpha([0.13, 0.29, 0.33, 0.035][finish]);
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
    if selection.is_none()
        && !scene
            .objects
            .iter()
            .any(|o| o.kind == ObjectKind::DrinkingGlass)
    {
        return;
    }
    for finish in 0..4 {
        let key = (Surface::ContainerGlass, FIRST + finish);
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            variants.insert(
                key,
                materials.add(material(scene.material_seed(), finish, quality)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        render::glass::AnnotationGlass,
        scene::procedural_indoor::materials::{finish_slot, registered_finish},
    };
    #[test]
    fn tableware_finishes_follow_portable_and_transmissive_annotation_paths() {
        let labels = [
            Tint::Flint.label(),
            Tint::Aqua.label(),
            Tint::Amber.label(),
            IMMERSED,
        ];
        for (i, label) in labels.into_iter().enumerate() {
            let slot = finish_slot(Surface::ContainerGlass, label).unwrap();
            assert_eq!(slot, FIRST + i);
            assert!(registered_finish(Surface::ContainerGlass, slot));
            for q in [IndoorQuality::Auto, IndoorQuality::Portable] {
                let m = material(17, i, q);
                assert_eq!(m.specular_transmission > 0., q.specular_transmission());
                assert!(
                    m.ior > 1. && m.normal_map_texture.is_none() && m.base_color_texture.is_none()
                );
                assert!(
                    !AnnotationGlass::Through.includes(Some(&m))
                        && AnnotationGlass::Surface.includes(Some(&m))
                );
                assert_eq!(m.ior < 1.2, i == 3);
            }
        }
    }
}
