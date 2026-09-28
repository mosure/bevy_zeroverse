//! A fixed-size per-scene finish palette; cloned materials share all surface maps.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

pub const COUNT: usize = 12;
pub const SURFACES: [Surface; 15] = [
    Surface::Ceramic,
    Surface::Metal,
    Surface::Chrome,
    Surface::Plastic,
    Surface::Fabric,
    Surface::FabricAlt,
    Surface::Art,
    Surface::Wood,
    Surface::WoodEdge,
    Surface::Screen,
    Surface::PhoneScreen,
    Surface::PrintedPaper,
    Surface::Leather,
    Surface::Whiteboard,
    Surface::Television,
];
pub fn supports(surface: Surface) -> bool {
    SURFACES.contains(&surface)
}
pub fn slot(seed: u64) -> usize {
    let mut rng = stream(seed, 845);
    rng.random_range(0..COUNT)
}
pub fn screen_seed(scene_seed: u64, slot: usize) -> u64 {
    let mut rng = stream(scene_seed, 900 + slot as u64);
    rng.random()
}
pub(super) fn build(
    seed: u64,
    base: &[Handle<StandardMaterial>],
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = BTreeMap::new();
    for surface in SURFACES {
        for index in 0..COUNT {
            if used.is_some_and(|keys| !keys.contains(&(surface, index))) {
                continue;
            }
            let mut rng = stream(seed, 1100 + surface as u64 * COUNT as u64 + index as u64);
            let mut mat = materials.get(&base[surface as usize]).unwrap().clone();
            if matches!(surface, Surface::Screen | Surface::PhoneScreen) {
                screens::apply(
                    screen_seed(seed, index),
                    surface == Surface::PhoneScreen,
                    &mut mat,
                    images,
                );
            } else if surface == Surface::Television {
                screens::apply_tv(screen_seed(seed, index), &mut mat, images);
            } else if surface == Surface::Whiteboard {
                boards::apply(screen_seed(seed, index), &mut mat, images);
            } else if surface == Surface::PrintedPaper {
                paper::apply(screen_seed(seed, index), &mut mat, images);
            } else {
                // Upholstery and device plastics remain in the scene palette;
                // small personal accessories can have independently chosen hues.
                let colorful = matches!(surface, Surface::Ceramic | Surface::Art);
                mat.base_color = if colorful {
                    Color::hsl(
                        rng.random_range(0.0..360.0),
                        rng.random_range(0.12..0.72),
                        rng.random_range(0.20..0.78),
                    )
                } else if matches!(surface, Surface::Metal | Surface::Chrome) {
                    let silver = rng.random_range(0.15..0.82);
                    let warm = rng.random_range(0.0..0.10);
                    Color::srgb(silver, silver * (1. - warm), silver * (1. - warm * 1.8))
                } else if surface == Surface::Plastic {
                    let v = rng.random_range(0.035..0.31);
                    Color::srgb(v, v * 1.01, v * 1.02)
                } else {
                    let c = mat.base_color.to_srgba();
                    let scale = rng.random_range(0.72..1.16);
                    Color::srgb(
                        (c.red * scale).min(0.95),
                        (c.green * scale).min(0.95),
                        (c.blue * scale).min(0.95),
                    )
                };
                // Multipliers preserve the shared roughness map's spatial detail.
                mat.perceptual_roughness *= rng.random_range(0.85..1.0);
                if surface == Surface::Ceramic {
                    mat.clearcoat = rng.random_range(0.0..0.65);
                    mat.clearcoat_perceptual_roughness = rng.random_range(0.07..0.24);
                }
            }
            result.insert((surface, index), materials.add(mat));
        }
    }
    result
}
