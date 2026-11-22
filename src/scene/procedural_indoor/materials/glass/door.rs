//! Architectural glass mounted in a door leaf. Tint and frost come from
//! the room glazing recipe; the optical path comes from the installed sheet.
use super::{GlassRecipe, SheetGauge};
use crate::scene::procedural_indoor::{
    layout::IndoorManifest,
    materials::{MaterialSelection, Surface},
    preparation::AssetStore,
    IndoorQuality,
};
use bevy::prelude::*;
use std::collections::BTreeMap;

const GAUGES: [SheetGauge; 2] = [SheetGauge::Six, SheetGauge::Eight];

pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::GlassInterior && GAUGES.iter().any(|g| g.slot() == slot)
}

pub(crate) fn install(
    scene: &IndoorManifest,
    quality: IndoorQuality,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for gauge in GAUGES {
        let key = (Surface::GlassInterior, gauge.slot());
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            let mut recipe = GlassRecipe::sample(scene.material_seed(), key.0);
            recipe.thickness_m = gauge.metres();
            let mut material = StandardMaterial::default();
            recipe.apply(&mut material, quality);
            variants.insert(key, materials.add(material));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{render::glass::AnnotationGlass, scene::procedural_indoor::materials};
    #[test]
    fn installed_gauge_and_through_glass_policy_match_for_selected_and_full_builds() {
        let scene =
            IndoorManifest::generate_with_humans(39, Default::default(), 0.65, 0, 0.).unwrap();
        for gauge in GAUGES {
            let key = (Surface::GlassInterior, gauge.slot());
            assert!(materials::registered_finish(key.0, key.1));
            for quality in [IndoorQuality::Auto, IndoorQuality::Portable] {
                let select = MaterialSelection {
                    surfaces: [key.0].into(),
                    direct_surfaces: Default::default(),
                    finishes: [key].into(),
                };
                for selection in [None, Some(&select)] {
                    let mut assets = Assets::<StandardMaterial>::default();
                    let mut variants = BTreeMap::new();
                    install(&scene, quality, &mut assets, selection, &mut variants);
                    assert_eq!(variants.len(), if selection.is_some() { 1 } else { 2 });
                    let mat = assets.get(&variants[&key]).unwrap();
                    assert_eq!(mat.thickness, gauge.metres());
                    assert_eq!(
                        mat.specular_transmission > 0.,
                        quality.specular_transmission()
                    );
                    assert!(AnnotationGlass::Surface.includes(Some(mat)));
                    assert!(!AnnotationGlass::Through.includes(Some(mat)));
                }
            }
        }
    }
}
