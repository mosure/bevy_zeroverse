//! Opaque silver-backed dressing glass. Use the room's reflection probe without
//! another camera render or the normal/roughness maps of brushed hardware.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorManifest, ObjectKind},
    materials::MaterialSelection,
    preparation::AssetStore,
};
use std::collections::BTreeMap;

pub(crate) const FACE: &str = "mirror#finish964";
const KEY: (Surface, usize) = (Surface::Chrome, 964);

pub(in crate::scene::procedural_indoor::materials) fn supports(
    surface: Surface,
    slot: usize,
) -> bool {
    (surface, slot) == KEY
}
fn material(seed: u64) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"BackedAg"));
    let value = r.random_range(0.87..0.95);
    let tint = r.random_range(0.003..0.021);
    StandardMaterial {
        base_color: Color::linear_rgb(value - tint, value, value - 0.7 * tint),
        metallic: 1.,
        perceptual_roughness: r.random_range(0.045..0.088),
        ..default()
    }
}
pub(in crate::scene::procedural_indoor::materials) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if selection.map_or_else(
        || {
            scene
                .objects
                .iter()
                .any(|o| o.kind == ObjectKind::FittingRoom)
        },
        |s| s.finishes.contains(&KEY),
    ) {
        variants.insert(KEY, materials.add(material(scene.material_seed())));
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::glass::AnnotationGlass;
    #[test]
    fn fitting_mirror_is_smooth_and_opaque_under_both_annotation_policies() {
        for seed in 0..20 {
            let m = material(seed);
            assert!(
                m.perceptual_roughness < 0.09 && m.metallic == 1. && m.normal_map_texture.is_none()
            );
            assert!(AnnotationGlass::Through.includes(Some(&m)));
            assert!(AnnotationGlass::Surface.includes(Some(&m)));
        }
    }
}
