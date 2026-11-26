//! A checkout belt is a fine-grained, pigmented carrying surface. Scanner
//! windows are backed optical glass and have no transmissive path through the unit.
use super::*;
pub(crate) const BELT: &str = "counter#finish988";
pub(crate) const SCANNER: &str = "control_panel#finish989";
const BELT_KEY: (Surface, usize) = (Surface::Rubber, 988);
const GLASS_KEY: (Surface, usize) = (Surface::Ink, 989);

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    (s, slot) == BELT_KEY || (s, slot) == GLASS_KEY
}

fn belt_map() -> Image {
    let size = 32_u32;
    let mut grain = stream(0, u64::from_le_bytes(*b"BeltSkin"));
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..size * size {
        pixels.extend_from_slice(&[255, grain.random_range(209..=248), 0, 255]);
    }
    mip_image(pixels, size, MapType::Data)
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let (belt, glass) = if let Some(s) = selection {
        (
            s.finishes.contains(&BELT_KEY),
            s.finishes.contains(&GLASS_KEY),
        )
    } else {
        let mut belt = false;
        let mut glass = false;
        for f in scene.objects.iter().filter_map(CheckoutForm::for_object) {
            belt |= f == CheckoutForm::ConveyorLane;
            glass |= f != CheckoutForm::ApparelWrap;
        }
        (belt, glass)
    };
    if !belt && !glass {
        return;
    }
    let mut r = stream(scene.material_seed(), u64::from_le_bytes(*b"TillDeck"));
    // The shop's installed equipment determines its finish lot. Inspecting only
    // the scan head of an installed lane must reproduce the full assembly.
    let installed_belt = belt
        || (selection.is_some()
            && scene
                .objects
                .iter()
                .any(|o| CheckoutForm::for_object(o) == Some(CheckoutForm::ConveyorLane)));
    if installed_belt {
        let v = r.random_range(0.072_f32..0.175);
        let green = r.random_range(0.98..1.10);
        let blue = r.random_range(0.92..1.06);
        let roughness = r.random_range(0.60..0.86);
        let period = r.random_range(0.075..0.14);
        if belt {
            static GRAIN: std::sync::OnceLock<Image> = std::sync::OnceLock::new();
            let material = StandardMaterial {
                base_color: Color::srgb(v, v * green, v * blue),
                metallic: 0.,
                perceptual_roughness: roughness,
                metallic_roughness_texture: Some(images.add(GRAIN.get_or_init(belt_map).clone())),
                uv_transform: bevy::math::Affine2::from_scale(Vec2::splat(1. / period)),
                ..default()
            };
            variants.insert(BELT_KEY, materials.add(material));
        }
    }
    if glass {
        let v = r.random_range(0.047_f32..0.112);
        let material = StandardMaterial {
            base_color: Color::srgb(v * 0.82, v * 0.94, v),
            metallic: 0.,
            perceptual_roughness: r.random_range(0.095..0.19),
            reflectance: 0.5,
            ..default()
        };
        variants.insert(GLASS_KEY, materials.add(material));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::ObjectKind, objects::audit};

    #[test]
    fn checkout_scanner_finish_replays_when_rendering_just_the_window() {
        let mut shop =
            IndoorManifest::generate_with_humans(51, Default::default(), 0.65, 0, 0.).unwrap();
        for seed in [12, 13] {
            shop.objects = vec![audit::fixture(ObjectKind::CheckoutCounter, seed)];
            let subset = MaterialSelection {
                surfaces: default(),
                direct_surfaces: default(),
                finishes: [GLASS_KEY].into_iter().collect(),
            };
            let render = |selection| {
                let mut images = Assets::default();
                let mut materials = Assets::default();
                let mut variants = BTreeMap::new();
                install(&shop, &mut images, &mut materials, selection, &mut variants);
                let window = materials.get(&variants[&GLASS_KEY]).unwrap();
                (
                    (window.base_color, window.perceptual_roughness),
                    images.len(),
                )
            };
            let (full, _) = render(None);
            let (partial, images) = render(Some(&subset));
            assert_eq!(full, partial);
            assert_eq!(images, 0);
        }
    }
}
