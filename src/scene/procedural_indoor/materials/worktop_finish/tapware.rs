//! Optical response of plated, brushed and powder-coated tap bodies.
use super::*;

pub(super) fn material(seed: u64, surface: Surface, mut m: StandardMaterial) -> StandardMaterial {
    let mut r = stream(seed, 0x5441_504f_5054 ^ surface as u64);
    match surface {
        Surface::Chrome => {
            let v = r.random_range(0.53..0.67);
            m.base_color = Color::linear_rgb(v * 0.98, v, v * 1.02);
            m.metallic = 1.;
            m.perceptual_roughness = r.random_range(0.10..0.22);
        }
        Surface::Metal => {
            // Satin nickel, stainless steel and warm PVD finishes inherit the brushed stock.
            let warm = if r.random_bool(0.33) {
                r.random_range(0.42..1.)
            } else {
                r.random_range(0.0..0.16)
            };
            let tint =
                Vec3::splat(r.random_range(0.52..0.63)) + warm * Vec3::new(0.22, -0.015, -0.27);
            m.base_color = Color::linear_rgb(tint.x, tint.y, tint.z);
        }
        Surface::Ink => {
            let v = r.random_range(0.023..0.082);
            m.base_color = Color::linear_rgb(v * 1.02, v, v * 0.97);
            m.perceptual_roughness = r.random_range(0.38..0.62);
            m.clearcoat = r.random_range(0.025..0.12);
            m.clearcoat_perceptual_roughness = m.perceptual_roughness * 0.86;
        }
        _ => unreachable!("tap stock"),
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_and_archival_hardware_have_distinct_brdfs_and_share_brush_maps() {
        use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};
        let mut scene =
            IndoorManifest::generate_with_humans(7, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
        let mut fixtures = Vec::new();
        for finish in [Surface::Chrome, Surface::Metal, Surface::Ink] {
            let o = (0..128)
                .map(|i| {
                    let mut o = audit::fixture(ObjectKind::KitchenSink, i);
                    o.finish_seed = Some(i);
                    o
                })
                .find(|o| kitchen::sink::program::finishes(o) == (Surface::Metal, finish))
                .unwrap();
            fixtures.push(o);
            assert_eq!(
                super::super::super::finish_slot(finish, TAP_LABEL),
                Some(TAP)
            );
        }
        scene.objects = fixtures;
        let selected = MaterialSelection {
            surfaces: Default::default(),
            direct_surfaces: Default::default(),
            finishes: [
                (Surface::Metal, BASIN),
                (Surface::Metal, TAP),
                (Surface::Ink, TAP),
                (Surface::Chrome, TAP),
            ]
            .into(),
        };
        for selection in [Some(&selected), None] {
            let mut images = Assets::default();
            let mut materials = Assets::default();
            let mut variants = BTreeMap::new();
            install(
                &scene,
                &mut images,
                &mut materials,
                selection,
                &mut variants,
            );
            let get = |s, slot| materials.get(&variants[&(s, slot)]).unwrap();
            let metal = get(Surface::Metal, TAP);
            let chrome = get(Surface::Chrome, TAP);
            let coat = get(Surface::Ink, TAP);
            assert_eq!(metal.metallic, 1.);
            assert!(metal.metallic_roughness_texture.is_some());
            assert_eq!(
                metal.metallic_roughness_texture,
                get(Surface::Metal, BASIN).metallic_roughness_texture
            );
            assert!(chrome.metallic == 1. && chrome.perceptual_roughness < 0.23);
            assert!(coat.metallic == 0. && coat.perceptual_roughness > 0.37 && coat.clearcoat > 0.);
        }
        scene.generator_version = kitchen::sink::FITTED_PLUMBING_VERSION - 1;
        let mut variants = BTreeMap::new();
        install(
            &scene,
            &mut Assets::default(),
            &mut Assets::default(),
            None,
            &mut variants,
        );
        assert!(!variants.keys().any(|&(_, slot)| slot == TAP));
    }
}
