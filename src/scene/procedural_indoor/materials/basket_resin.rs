//! Dyed polypropylene stock shares the room's molded-polymer relief and
//! roughness maps. Geometry supplies the basket openings and highlights.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 1072;
const COLORS: usize = 6;
pub(crate) fn sample_slot(r: &mut impl Rng) -> usize {
    FIRST + r.random_range(0..COLORS)
}
pub(super) fn parent(surface: Surface, slot: usize) -> Option<usize> {
    (surface == Surface::Plastic && (FIRST..FIRST + COLORS).contains(&slot)).then_some(0)
}
fn resin(stock: &StandardMaterial, seed: u64, slot: usize) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"BasketPP") ^ slot as u64);
    let (h, s, l) = match slot - FIRST {
        0 => (
            r.random_range(-12_f32..14.).rem_euclid(360.),
            r.random_range(0.60..0.83),
            r.random_range(0.32..0.49),
        ),
        1 => (
            r.random_range(198.0..235.),
            r.random_range(0.54..0.82),
            r.random_range(0.29..0.47),
        ),
        2 => (
            r.random_range(124.0..172.),
            r.random_range(0.38..0.72),
            r.random_range(0.22..0.39),
        ),
        3 => (
            r.random_range(18.0..53.),
            r.random_range(0.68..0.91),
            r.random_range(0.44..0.60),
        ),
        4 => (
            r.random_range(152.0..238.),
            r.random_range(0.03..0.15),
            r.random_range(0.09..0.24),
        ),
        _ => (
            r.random_range(30.0..220.),
            r.random_range(0.035..0.16),
            r.random_range(0.60..0.84),
        ),
    };
    let mut m = stock.clone();
    m.base_color = Color::hsl(h, s, l);
    m.base_color_texture = None;
    m.metallic = 0.;
    m.perceptual_roughness = r.random_range(0.38..0.60);
    m
}
pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    use crate::scene::procedural_indoor::{
        layout::ObjectKind,
        objects::public::{baskets, retail},
    };
    if scene.generator_version < baskets::VERSION {
        return;
    }
    let mut add = |slot: usize| {
        let key = (Surface::Plastic, slot);
        if !variants.contains_key(&key) {
            let stock = materials
                .get(&variants[&(Surface::Plastic, 0)])
                .expect("prepared molded basket stock");
            let m = resin(stock, scene.material_seed(), slot);
            variants.insert(key, materials.add(m));
        }
    };
    if let Some(s) = selection {
        for &(_, slot) in s
            .finishes
            .range((Surface::Plastic, FIRST)..(Surface::Plastic, FIRST + COLORS))
        {
            add(slot)
        }
    } else {
        for o in &scene.objects {
            match o.kind {
                ObjectKind::BasketStack => {
                    let p = baskets::BasketStack::for_object(o);
                    if p.shell == baskets::BasketShell::Molded || p.coated_wire {
                        add(p.dye_slot);
                    }
                }
                ObjectKind::GroceryDisplay
                    if scene.generator_version >= retail::PRODUCE_VERSION && o.size.y < 1.15 =>
                {
                    // Eager material callers need the same dye as selective preparation.
                    // One stock per stand, with shared relief and no new texture synthesis.
                    add(retail::produce_resin(o));
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basket_pigments_share_stock_relief_and_remain_dielectric() {
        let stock = StandardMaterial {
            normal_map_texture: Some(Handle::default()),
            metallic_roughness_texture: Some(Handle::default()),
            ..default()
        };
        let mut colors = std::collections::BTreeSet::new();
        for slot in FIRST..FIRST + COLORS {
            let m = resin(&stock, 412, slot);
            assert!(super::super::registered_finish(Surface::Plastic, slot));
            assert_eq!(m.normal_map_texture, stock.normal_map_texture);
            assert_eq!(
                m.metallic_roughness_texture,
                stock.metallic_roughness_texture
            );
            assert_eq!(m.metallic, 0.);
            colors.insert(m.base_color.to_srgba().to_u8_array());
        }
        assert_eq!(colors.len(), COLORS);
    }
}
