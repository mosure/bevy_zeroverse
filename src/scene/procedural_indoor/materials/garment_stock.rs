//! Personal outerwear can share a woven or leather texture with other objects
//! without inheriting the colour of the room's furniture.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 672;
const STOCK: [Surface; 3] = [Surface::Fabric, Surface::FabricAlt, Surface::Leather];

pub(crate) fn sample_slot(rng: &mut impl Rng) -> usize {
    FIRST + rng.random_range(0..variants::COUNT)
}

pub(super) fn parent(surface: Surface, slot: usize) -> Option<usize> {
    if (FIRST..FIRST + variants::COUNT).contains(&slot) && STOCK.contains(&surface) {
        Some(slot - FIRST)
    } else {
        None
    }
}

fn dye(seed: u64, surface: Surface, slot: usize) -> Color {
    let mut r = stream(
        seed,
        u64::from_le_bytes(*b"coat dye") ^ ((surface as u64) << 16) ^ slot as u64,
    );
    let (h, s, l) = match (surface, r.random_range(0..12)) {
        (_, 0..=2) => (205., 0.045, r.random_range(0.12..0.39)), // charcoal through slate
        (Surface::Leather, _) => (r.random_range(13.0..34.), 0.34, r.random_range(0.16..0.43)),
        (_, 3..=4) => (
            r.random_range(202.0..228.),
            0.43,
            r.random_range(0.19..0.39),
        ),
        (_, 5..=6) => (r.random_range(22.0..46.), 0.34, r.random_range(0.47..0.72)),
        (_, 7) => (r.random_range(63.0..133.), 0.27, r.random_range(0.23..0.48)),
        (_, 8) => (
            r.random_range(329.0..355.),
            0.43,
            r.random_range(0.24..0.47),
        ),
        (_, 9) => (r.random_range(24.0..62.), 0.11, r.random_range(0.68..0.86)),
        _ => (
            r.random_range(0.0..360.),
            r.random_range(0.30..0.64),
            r.random_range(0.29..0.60),
        ),
    };
    Color::hsl(h, s, l)
}

/// A store's coordinated colorways include undyed cotton, deep neutrals and
/// brighter seasonal stock, on the same filtered weave and leather maps.
fn shop_dye(seed: u64, surface: Surface, slot: usize) -> Color {
    if surface == Surface::Leather {
        return dye(seed, surface, slot);
    }
    let mut r = stream(seed, u64::from_le_bytes(*b"ShopDyes") ^ slot as u64);
    let family = stream(seed, u64::from_le_bytes(*b"Seasonal")).random_range(0.0_f32..360.);
    let hue =
        (family + ((slot - FIRST) % 3) as f32 * 109. + r.random_range(-32.0..32.)).rem_euclid(360.);
    let (sat, light) = match (slot - FIRST) % 6 {
        0 => (r.random_range(0.07..0.20), r.random_range(0.73..0.91)),
        1 => (r.random_range(0.06..0.27), r.random_range(0.16..0.37)),
        _ => (r.random_range(0.34..0.73), r.random_range(0.43..0.72)),
    };
    Color::hsl(hue, sat, light)
}

pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < super::super::objects::utilities::racks::dressing::INTRODUCED {
        return;
    }
    for surface in STOCK {
        let range = (surface, FIRST)..(surface, FIRST + variants::COUNT);
        let mut add = |slot: usize| {
            let stock = &variants[&(surface, slot - FIRST)];
            let mut m = materials
                .get(stock)
                .expect("prepared garment stock")
                .clone();
            m.base_color = if scene.layout
                == crate::scene::procedural_indoor::layout::IndoorLayout::ClothingStore
            {
                shop_dye(scene.material_seed(), surface, slot)
            } else {
                dye(scene.material_seed(), surface, slot)
            };
            // All rack garments have a separate physical inner surface and
            // turned edges, including the scarves. Shade each cut face once.
            m.double_sided = false;
            m.cull_mode = Some(bevy::render::render_resource::Face::Back);
            variants.insert((surface, slot), materials.add(m));
        };
        if let Some(s) = selection {
            for &(_, slot) in s.finishes.range(range) {
                add(slot);
            }
        } else {
            for slot in FIRST..FIRST + variants::COUNT {
                add(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn used_dyes_share_the_exact_texture_handles_and_roughness_of_their_own_stock() {
        let scene =
            IndoorManifest::generate_with_humans(73, Default::default(), 0.65, 0, 0.).unwrap();
        let mut used = MaterialSelection {
            surfaces: STOCK.into_iter().collect(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let mut storage = Assets::<StandardMaterial>::default();
        let mut vars = BTreeMap::new();
        for s in STOCK {
            for index in [0, 7] {
                used.select_finish(s, FIRST + index);
                vars.insert(
                    (s, index),
                    storage.add(StandardMaterial {
                        base_color_texture: Some(Handle::default()),
                        normal_map_texture: Some(Handle::default()),
                        metallic_roughness_texture: Some(Handle::default()),
                        perceptual_roughness: 0.76,
                        ..default()
                    }),
                );
            }
        }
        install(&scene, &mut storage, Some(&used), &mut vars);
        assert_eq!(vars.len(), 12);
        let mut colors = std::collections::BTreeSet::new();
        for s in STOCK {
            for i in [0, 7] {
                let source = storage.get(&vars[&(s, i)]).unwrap();
                let m = storage.get(&vars[&(s, FIRST + i)]).unwrap();
                assert_eq!(source.base_color_texture, m.base_color_texture);
                assert_eq!(source.normal_map_texture, m.normal_map_texture);
                assert_eq!(
                    source.metallic_roughness_texture,
                    m.metallic_roughness_texture
                );
                assert_eq!(source.perceptual_roughness, m.perceptual_roughness);
                assert!(
                    !m.double_sided
                        && m.cull_mode == Some(bevy::render::render_resource::Face::Back)
                );
                colors.insert(m.base_color.to_srgba().to_u8_array());
            }
        }
        assert!(colors.len() >= 5);
    }
}
