//! Opaque backed optics and unlit flash diffusers. Each used stock is shared
//! throughout the room and needs neither an image upload nor a transmission pass.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

pub(crate) const COVER: &str = "phone#finish134";
pub(crate) const LENS: &str = "phone#finish135";
pub(crate) const DIFFUSER: &str = "phone#finish136";

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Ink && (134..=136).contains(&slot)
}

pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if selection.is_none() && !scene.objects.iter().any(|o| o.kind == ObjectKind::Phone) {
        return;
    }
    for slot in 134..=136 {
        let key = (Surface::Ink, slot);
        if selection.is_some_and(|s| !s.finishes.contains(&key)) {
            continue;
        }
        let mut r = stream(
            scene.material_seed(),
            u64::from_le_bytes(*b"Handglas") ^ slot as u64,
        );
        let (color, roughness) = match slot {
            134 => {
                let v = r.random_range(0.036..0.095);
                (Color::srgb(v, v, v * 1.06), r.random_range(0.10..0.24))
            }
            135 => {
                let v = r.random_range(0.065..0.15);
                // Dark optical cavity with a weak green, blue or violet coating tint.
                (
                    Color::srgb(
                        v * r.random_range(0.57..0.93),
                        v * r.random_range(0.65..1.04),
                        v,
                    ),
                    r.random_range(0.065..0.16),
                )
            }
            _ => {
                let v = r.random_range(0.69..0.89);
                (
                    Color::srgb(
                        v,
                        v * r.random_range(0.90..0.98),
                        v * r.random_range(0.66..0.88),
                    ),
                    r.random_range(0.30..0.49),
                )
            }
        };
        variants.insert(
            key,
            materials.add(StandardMaterial {
                base_color: color,
                perceptual_roughness: roughness,
                ..default()
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handset_lens_is_selected_by_tag_and_flash_is_a_passive_dielectric() {
        let scene =
            IndoorManifest::generate_with_humans(17, Default::default(), 0.5, 0, 0.).unwrap();
        let mut materials = Assets::default();
        let mut selected = MaterialSelection {
            surfaces: [Surface::Ink].into(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        for (label, slot) in [(COVER, 134), (LENS, 135), (DIFFUSER, 136)] {
            assert_eq!(super::super::finish_slot(Surface::Ink, label), Some(slot));
            assert!(super::super::registered_finish(Surface::Ink, slot));
            selected.finishes = [(Surface::Ink, slot)].into();
            let mut variants = BTreeMap::new();
            install(&scene, &mut materials, Some(&selected), &mut variants);
            assert_eq!(variants.len(), 1);
            let m = materials.get(&variants[&(Surface::Ink, slot)]).unwrap();
            assert_eq!(m.alpha_mode, AlphaMode::Opaque);
            assert_eq!(m.specular_transmission, 0.);
            assert_eq!(m.metallic, 0.);
            assert_eq!(m.emissive.red + m.emissive.green + m.emissive.blue, 0.);
            assert!(m.base_color_texture.is_none() && m.normal_map_texture.is_none());
            assert_eq!(m.perceptual_roughness < 0.25, slot != 136);
        }
    }
}
