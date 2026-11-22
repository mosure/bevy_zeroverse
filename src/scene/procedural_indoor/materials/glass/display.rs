//! Toughened, clear sheet stock for display cabinets. Raster optics and mesh
//! construction share the sheet gauge. Each used gauge costs one material
//! handle per scene and needs no texture or bake.
use super::GlassRecipe;
use crate::scene::procedural_indoor::{
    layout::IndoorManifest,
    materials::{MaterialSelection, Surface},
    preparation::AssetStore,
    IndoorQuality,
};
use bevy::prelude::*;
use std::collections::BTreeMap;

pub use super::SheetGauge;

impl SheetGauge {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Four => "display_case#finish64",
            Self::Five => "display_case#finish65",
            Self::Six => "display_case#finish66",
            Self::Eight => "display_case#finish67",
        }
    }
    fn recipe(self, seed: u64) -> GlassRecipe {
        let mut r = GlassRecipe::sample(
            seed ^ 0x43415345474c4153 ^ self.slot() as u64,
            Surface::Glass,
        );
        r.thickness_m = self.metres();
        // Staff and customers view small items through clear low-tint stock.
        r.attenuation = r.attenuation.map(|c| 1. - (1. - c) * 0.20);
        r
    }
}
pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::ContainerGlass && (64..=67).contains(&slot)
}

pub(crate) fn install(
    scene: &IndoorManifest,
    quality: IndoorQuality,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for gauge in [
        SheetGauge::Four,
        SheetGauge::Five,
        SheetGauge::Six,
        SheetGauge::Eight,
    ] {
        let key = (Surface::ContainerGlass, gauge.slot());
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            let mut material = StandardMaterial::default();
            gauge
                .recipe(scene.material_seed())
                .apply(&mut material, quality);
            variants.insert(key, materials.add(material));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_cut_sheet_matches_native_and_portable_optics_and_annotation_policy() {
        use crate::{render::glass::AnnotationGlass, scene::procedural_indoor::materials};
        for gauge in [
            SheetGauge::Four,
            SheetGauge::Five,
            SheetGauge::Six,
            SheetGauge::Eight,
        ] {
            assert!(materials::registered_finish(
                Surface::ContainerGlass,
                gauge.slot()
            ));
            assert_eq!(
                materials::finish_slot(Surface::ContainerGlass, gauge.label()),
                Some(gauge.slot())
            );
            for seed in 0..24 {
                let recipe = gauge.recipe(seed);
                assert!((0.85..0.96).contains(&recipe.luminous_transmittance(1.)));
                for quality in [IndoorQuality::Portable, IndoorQuality::Auto] {
                    let mut m = StandardMaterial::default();
                    recipe.apply(&mut m, quality);
                    assert_eq!(m.thickness, gauge.metres());
                    assert!(!m.double_sided);
                    assert_eq!(
                        m.specular_transmission > 0.,
                        quality.specular_transmission()
                    );
                    assert!(!AnnotationGlass::Through.includes(Some(&m)));
                    assert!(AnnotationGlass::Surface.includes(Some(&m)));
                }
            }
        }
    }
}
