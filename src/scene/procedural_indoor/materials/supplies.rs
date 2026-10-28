//! Bounded product palettes reusing already requested PBR structure maps.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(surface, Surface::Plastic | Surface::Fabric) && (320..328).contains(&slot)
}
pub(super) fn build(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
) -> std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = std::collections::BTreeMap::new();
    for surface in [Surface::Plastic, Surface::Fabric] {
        for i in 0..8 {
            let slot = 320 + i;
            if selection.is_some_and(|s| !s.finishes.contains(&(surface, slot))) {
                continue;
            }
            let mut r = stream(scene.material_seed(), 0x53504f4e47454d41 + i as u64);
            let mut m = materials
                .get(&base[surface as usize])
                .expect("supply template")
                .clone();
            let hue = [49., 186., 29., 328., 204., 83., 38., 0.][i] + r.random_range(-9.0..9.);
            let foam = surface == Surface::Plastic;
            m.base_color = Color::hsl(
                hue,
                if i == 7 {
                    r.random_range(0.0..0.05)
                } else {
                    r.random_range(0.32..0.67)
                },
                if foam {
                    r.random_range(0.50..0.78)
                } else {
                    r.random_range(0.12..0.34)
                },
            );
            // Existing micro-normal and occlusion maps supply structure; do not tint
            // sponges with the furniture fabric's patterned albedo.
            m.base_color_texture = None;
            m.metallic_roughness_texture = None; // glossy plastic maps must not override foam roughness
            m.perceptual_roughness = r.random_range(0.78..0.98);
            m.metallic = 0.;
            m.clearcoat = 0.;
            m.reflectance = 0.3;
            m.uv_transform =
                bevy::math::Affine2::from_scale(Vec2::splat(if foam { 3.5 } else { 2.0 }))
                    * m.uv_transform;
            result.insert((surface, slot), materials.add(m));
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supply_bindings_reuse_structure_and_are_rough_dielectrics() {
        let scene =
            IndoorManifest::generate_with_humans(0, Default::default(), 0.5, 0, 0.).unwrap();
        let mut assets = Assets::<StandardMaterial>::default();
        let base: Vec<_> = Surface::ALL
            .iter()
            .map(|_| {
                assets.add(StandardMaterial {
                    normal_map_texture: Some(Handle::default()),
                    ..default()
                })
            })
            .collect();
        let all = build(&scene, &base, &mut assets, None);
        assert_eq!(all.len(), 16);
        let mut colors = std::collections::BTreeSet::new();
        for ((s, slot), h) in all {
            let m = assets.get(&h).unwrap();
            assert!(supports(s, slot));
            assert!(m.base_color_texture.is_none());
            assert_eq!(
                m.normal_map_texture,
                assets.get(&base[s as usize]).unwrap().normal_map_texture
            );
            assert_eq!(m.metallic, 0.);
            assert!(m.perceptual_roughness >= 0.78);
            colors.insert(m.base_color.to_srgba().to_u8_array());
        }
        assert_eq!(colors.len(), 16);
    }
}
