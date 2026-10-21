//! Bounded manufactured enamel/plastic palette; no procedural image uploads.
use super::*;
use crate::scene::procedural_indoor::{
    layout::stream, objects::equipment::signals::fire::FINISH_BASE,
};
use rand::Rng;

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(surface, Surface::Plastic | Surface::Metal)
        && (FINISH_BASE..FINISH_BASE + 5).contains(&slot)
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
    result
}
