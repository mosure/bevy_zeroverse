//! Bounded manufactured enamel/plastic palette; no procedural image uploads.
use super::*;
use crate::scene::procedural_indoor::{
    layout::stream,
    objects::equipment::signals::fire::{
        detector::{CHAMBER, FIRST_LENS, LAST_LENS},
        FINISH_BASE,
    },
};
use rand::Rng;

/// Translucent-colored indicator resin shared by manufactured signal hardware.
pub(crate) fn indicator_lens_finish(amber: bool) -> usize {
    if amber {
        FIRST_LENS + 1
    } else {
        LAST_LENS
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    (matches!(surface, Surface::Plastic | Surface::Metal)
        && (FINISH_BASE..FINISH_BASE + 5).contains(&slot))
        || (surface == Surface::Plastic && (CHAMBER..=LAST_LENS).contains(&slot))
}

pub(super) fn build(
    scene: &IndoorManifest,
    materials: &mut impl super::super::preparation::AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
) -> std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = std::collections::BTreeMap::new();
    let mut r = stream(scene.seed, 0x4649524546494e);
    let colors = [
        Color::srgb(
            r.random_range(0.65..0.90),
            r.random_range(0.025..0.10),
            r.random_range(0.02..0.06),
        ),
        Color::srgb(0.94, 0.94, 0.91),
        Color::srgb(0.85, 0.82, 0.72),
        Color::srgb(0.12, 0.13, 0.14),
        Color::srgb(0.56, 0.59, 0.61),
    ];
    for surface in [Surface::Plastic, Surface::Metal] {
        for (i, color) in colors.into_iter().enumerate() {
            let slot = FINISH_BASE + i;
            let roughness = r.random_range(0.25..0.52);
            if selection.is_some_and(|s| !s.finishes.contains(&(surface, slot))) {
                continue;
            }
            result.insert(
                (surface, slot),
                materials.add(StandardMaterial {
                    base_color: color,
                    perceptual_roughness: roughness,
                    metallic: if surface == Surface::Metal && i == 4 {
                        0.8
                    } else {
                        0.
                    },
                    ..default()
                }),
            );
        }
    }
    for slot in CHAMBER..=LAST_LENS {
        let key = (Surface::Plastic, slot);
        if selection.is_some_and(|s| !s.finishes.contains(&key)) {
            continue;
        }
        let material = if slot == CHAMBER {
            StandardMaterial {
                base_color: Color::srgb(0.075, 0.069, 0.065),
                perceptual_roughness: 0.84,
                ..default()
            }
        } else {
            let tint = [
                Color::srgb(0.68, 0.047, 0.032),
                Color::srgb(0.84, 0.32, 0.036),
                Color::srgb(0.052, 0.50, 0.19),
            ][slot - FIRST_LENS];
            StandardMaterial {
                base_color: tint,
                perceptual_roughness: 0.22,
                emissive: tint.to_linear() * 0.35,
                ..default()
            }
        };
        result.insert(key, materials.add(material));
    }
    result
}
