//! Factory-coated and brushed blind stock. Lighting a painted aluminium blade
//! uses the outer dielectric coating; the metal underneath does not act as a mirror.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 352;

pub(crate) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(surface, Surface::Metal | Surface::Plastic)
        && (FIRST..FIRST + variants::COUNT).contains(&slot)
}

pub(crate) fn slot(surface: Surface, finish: usize) -> Option<usize> {
    matches!(surface, Surface::Metal | Surface::Plastic).then_some(FIRST + finish)
}

fn material(seed: u64, surface: Surface, finish: usize) -> StandardMaterial {
    let mut r = stream(
        seed,
        0x56454e455449414e ^ (surface as u64 * 31 + finish as u64),
    );
    let brushed = surface == Surface::Metal && r.random_bool(0.18);
    let (color, roughness) = if brushed {
        let v = r.random_range(0.63..0.87);
        (
            Color::srgb(v, v * 0.99, v * 0.975),
            r.random_range(0.42..0.60),
        )
    } else {
        let (h, s, l) = match r.random_range(0..10) {
            0..=4 => (
                r.random_range(32.0..56.),
                r.random_range(0.025..0.17),
                r.random_range(0.72..0.92),
            ),
            5..=7 => (
                r.random_range(185.0..235.),
                r.random_range(0.025..0.14),
                r.random_range(0.37..0.72),
            ),
            _ => (
                r.random_range(0.0..360.),
                r.random_range(0.16..0.44),
                r.random_range(0.26..0.60),
            ),
        };
        (Color::hsl(h, s, l), r.random_range(0.48..0.77))
    };
    StandardMaterial {
        base_color: color,
        metallic: if brushed { 1. } else { 0. },
        perceptual_roughness: roughness,
        ..default()
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for surface in [Surface::Metal, Surface::Plastic] {
        for finish in 0..variants::COUNT {
            let key = (surface, FIRST + finish);
            if selection.is_none_or(|s| s.finishes.contains(&key)) {
                variants.insert(
                    key,
                    materials.add(material(scene.material_seed(), surface, finish)),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dielectric_coatings_and_brushed_metal_are_distinct_and_opaque() {
        let mut bare = 0;
        for seed in 0..128 {
            for surface in [Surface::Plastic, Surface::Metal] {
                let finish = seed as usize % variants::COUNT;
                let s = slot(surface, finish).unwrap();
                assert!(super::super::registered_finish(surface, s));
                let m = material(seed, surface, finish);
                assert_eq!(m.alpha_mode, AlphaMode::Opaque);
                assert!((0.42..0.78).contains(&m.perceptual_roughness));
                assert!(m.normal_map_texture.is_none() && m.metallic_roughness_texture.is_none());
                if m.metallic != 0. {
                    assert_eq!((surface, m.metallic), (Surface::Metal, 1.));
                    bare += 1;
                }
            }
        }
        assert!((8..45).contains(&bare));
        assert!(slot(Surface::WoodEdge, 0).is_none());
    }
}
