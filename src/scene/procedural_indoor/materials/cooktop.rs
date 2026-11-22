//! Opaque glass-ceramic and seasoned iron. One mipmapped printed-zone swatch
//! serves every electric hob in the room regardless of its burner layout.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

pub(crate) const GLASS_LABEL: &str = "stove#finish107";
pub(crate) const IRON_LABEL: &str = "stove#finish108";
const GLASS: (Surface, usize) = (Surface::Ceramic, 107);
const IRON: (Surface, usize) = (Surface::Metal, 108);
/// Radius of the outer printed track in the shared [0, 1] chart.
pub(crate) const PRINT_RADIUS: f32 = 0.40;

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    (surface, slot) == GLASS || (surface, slot) == IRON
}

fn print_image(seed: u64) -> Image {
    const SIDE: u32 = 128;
    let mut rng = stream(seed, 0x484f_4250_5249_4e54);
    let substrate = rng.random_range(21.0..40.0);
    let print = rng.random_range(116.0..184.0);
    let width = rng.random_range(0.006..0.013);
    let inner = rng
        .random_bool(0.45)
        .then(|| rng.random_range(0.245..0.325));
    let mut pixels = Vec::with_capacity((SIDE * SIDE * 4) as usize);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let p = (Vec2::new(x as f32, y as f32) + 0.5) / SIDE as f32 - 0.5;
            let radius = p.length();
            let mut distance = (radius - PRINT_RADIUS).abs();
            if let Some(inner) = inner {
                distance = distance.min((radius - inner).abs());
            }
            let coverage = ((width * 0.5 - distance) * SIDE as f32 + 0.5).clamp(0., 1.);
            let c = substrate + (print - substrate) * coverage;
            pixels.extend([c as u8, c as u8, (c + 2.) as u8, 255]);
        }
    }
    let mut image = mip_image(pixels, SIDE, MapType::Color);
    if let ImageSampler::Descriptor(s) = &mut image.sampler {
        s.address_mode_u = ImageAddressMode::ClampToEdge;
        s.address_mode_v = ImageAddressMode::ClampToEdge;
    }
    image
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let mut rng = stream(scene.material_seed(), 0x484f_4246_494e_4953);
    for key in [GLASS, IRON] {
        let used = selection.map_or_else(
            || {
                scene.objects.iter().any(|o| {
                    matches!(
                        o.kind,
                        super::super::layout::ObjectKind::Cooktop
                            | super::super::layout::ObjectKind::Range
                    )
                })
            },
            |s| s.finishes.contains(&key),
        );
        if !used {
            continue;
        }
        let mat = if key == GLASS {
            StandardMaterial {
                base_color_texture: Some(images.add(print_image(scene.material_seed()))),
                perceptual_roughness: rng.random_range(0.12..0.24),
                reflectance: 0.5,
                ..default()
            }
        } else {
            // Oxidized or enamelled cast iron has a dark, diffusely reflecting skin.
            let mut iron = stream(scene.material_seed(), 0x4752_4154_4543_4f4c);
            let v = iron.random_range(0.10..0.21);
            StandardMaterial {
                base_color: Color::srgb(v, v * 0.97, v * 0.92),
                perceptual_roughness: iron.random_range(0.62..0.78),
                ..default()
            }
        };
        variants.insert(key, materials.add(mat));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    #[test]
    fn glass_and_iron_share_stove_semantics_and_only_bake_the_requested_print() {
        let scene =
            IndoorManifest::generate_with_humans(2601, IndoorLayout::Kitchen, 0.65, 0, 0.).unwrap();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let mut variants = BTreeMap::new();
        let mut selection = MaterialSelection {
            surfaces: Default::default(),
            finishes: [IRON].into(),
            direct_surfaces: Default::default(),
        };
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut variants,
        );
        assert_eq!(variants.len(), 1);
        assert_eq!(images.len(), 0);
        selection.finishes = [GLASS].into();
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut variants,
        );
        assert_eq!(images.len(), 1);
        for (key, label) in [(IRON, IRON_LABEL), (GLASS, GLASS_LABEL)] {
            assert!(super::super::registered_finish(key.0, key.1));
            assert_eq!(super::super::finish_slot(key.0, label), Some(key.1));
            let m = materials.get(&variants[&key]).unwrap();
            assert_eq!(m.alpha_mode, AlphaMode::Opaque);
            assert_eq!(m.specular_transmission, 0.);
            assert_eq!(m.metallic, 0.);
            assert!(m.normal_map_texture.is_none() && m.emissive_texture.is_none());
            if key == GLASS {
                assert!((0.12..0.24).contains(&m.perceptual_roughness));
                let im = images.get(m.base_color_texture.as_ref().unwrap()).unwrap();
                assert_eq!(im.texture_descriptor.mip_level_count, 8);
                let ImageSampler::Descriptor(s) = &im.sampler else {
                    panic!("missing filter")
                };
                assert_eq!(s.address_mode_u, ImageAddressMode::ClampToEdge);
                assert_eq!(s.address_mode_v, ImageAddressMode::ClampToEdge);
                assert_eq!(s.mipmap_filter, ImageFilterMode::Linear);
                let data = im.data.as_ref().unwrap();
                let at = |x: usize, y: usize| data[4 * (128 * y + x)];
                assert_eq!(at(0, 0), at(64, 64));
                assert!((112..119).map(|x| at(x, 64)).max().unwrap() > at(0, 0) + 40);
            } else {
                assert!(m.perceptual_roughness > 0.6);
            }
        }
    }
}
