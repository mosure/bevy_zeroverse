//! Bounded scene palettes. Upholstery shares three independent structure programs
//! per substrate, instead of tinting one texture for every object in the room.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

pub const COUNT: usize = 12;
pub const SURFACES: [Surface; 18] = [
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
    Surface::Leaf,
    Surface::LeafLight,
    Surface::LeafVariegated,
];
pub fn structure_count(surface: Surface) -> usize {
    match surface {
        Surface::Fabric | Surface::FabricAlt | Surface::Leather => 3,
        Surface::Wood
        | Surface::WoodEdge
        | Surface::Leaf
        | Surface::LeafLight
        | Surface::LeafVariegated => 2,
        _ => 1,
    }
}
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
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let seed = scene.material_seed();
    let structures = structures(scene, images, used);
    let mut result = BTreeMap::new();
    for surface in SURFACES {
        for index in 0..COUNT {
            if used.is_some_and(|keys| !keys.contains(&(surface, index))) {
                continue;
            }
            let mut rng = stream(seed, 1100 + surface as u64 * COUNT as u64 + index as u64);
            let mut mat = materials.get(&base[surface as usize]).unwrap().clone();
            if let Some((recipe, maps)) =
                structures.get(&(surface, index % structure_count(surface)))
            {
                mat.base_color_texture = Some(maps[0].clone());
                mat.normal_map_texture = Some(maps[1].clone());
                mat.metallic_roughness_texture = Some(maps[2].clone());
                mat.occlusion_texture = Some(maps[2].clone());
                mat.uv_transform = bevy::math::Affine2::from_scale(recipe.period_uv().recip());
                if recipe.absolute_color(scene.floor_style) {
                    mat.base_color = Color::WHITE;
                }
                recipe.apply_pbr(&mut mat);
            }
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
                    let silver = if surface == Surface::Chrome {
                        rng.random_range(0.78..0.94)
                    } else {
                        rng.random_range(0.08..0.55)
                    };
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
                if surface == Surface::Ceramic && scene.program.is_none() {
                    mat.clearcoat = rng.random_range(0.0..0.65);
                    mat.clearcoat_perceptual_roughness = rng.random_range(0.07..0.24);
                }
            }
            result.insert((surface, index), materials.add(mat));
        }
    }
    result
}

type Structure = (program::MaterialRecipe, [Handle<Image>; 3]);
fn structures(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> BTreeMap<(Surface, usize), Structure> {
    let Some(p) = &scene.program else {
        return BTreeMap::new();
    };
    let jobs: Vec<_> = [
        Surface::Fabric,
        Surface::FabricAlt,
        Surface::Leather,
        Surface::Wood,
        Surface::WoodEdge,
        Surface::Leaf,
        Surface::LeafLight,
        Surface::LeafVariegated,
    ]
    .into_iter()
    .flat_map(|s| {
        (1..structure_count(s))
            .filter(move |g| {
                used.is_none_or(|keys| {
                    keys.iter()
                        .any(|(surface, slot)| *surface == s && slot % structure_count(s) == *g)
                })
            })
            .map(move |g| (s, g, p.materials[s as usize].variant(g)))
    })
    .collect();
    let prepare = |(s, g, r): &(Surface, usize, program::MaterialRecipe)| {
        let maps = r.maps(scene.floor_style);
        ((*s, *g), r.clone(), maps)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let ready = super::super::preparation::workers::pool().scope(|scope| {
        for job in &jobs {
            let prepare = &prepare;
            scope.spawn(async move { prepare(job) });
        }
    });
    #[cfg(target_arch = "wasm32")]
    let ready: Vec<_> = jobs.iter().map(prepare).collect();
    ready
        .into_iter()
        .map(|(key, r, maps)| (key, (r, maps.map(|map| images.add(map)))))
        .collect()
}
