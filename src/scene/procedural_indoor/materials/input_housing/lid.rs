//! Opaque dielectric finishes for the inactive border, webcam and privacy blade.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LidStock {
    Molded,
    Bonded,
    Lens,
    Shutter,
}

impl LidStock {
    const FIRST: usize = 812;
    const ALL: [Self; 4] = [Self::Molded, Self::Bonded, Self::Lens, Self::Shutter];

    pub(crate) fn from_slot(slot: usize) -> Option<Self> {
        Self::ALL.get(slot.checked_sub(Self::FIRST)?).copied()
    }
    pub(crate) fn slot(self) -> usize {
        Self::FIRST + self as usize
    }
    pub(crate) fn label(self) -> &'static str {
        [
            "laptop#finish812",
            "laptop#finish813",
            "laptop#finish814",
            "laptop#finish815",
        ][self as usize]
    }
    pub(super) fn material(self, seed: u64) -> StandardMaterial {
        let mut r = stream(seed, u64::from_le_bytes(*b"LidStock") + self as u64);
        let (tint, shade, roughness) = match self {
            Self::Molded => (Vec3::new(0.95, 0.97, 1.), 0.08..0.20, 0.43..0.65),
            Self::Bonded => (Vec3::new(0.87, 0.96, 1.), 0.026..0.075, 0.09..0.23),
            Self::Lens => (Vec3::new(0.44, 0.62, 1.), 0.065..0.16, 0.10..0.21),
            Self::Shutter => (
                if r.random_bool(0.55) {
                    Vec3::new(0.92, 0.35, 0.24)
                } else {
                    Vec3::new(0.91, 0.94, 1.)
                },
                0.31..0.52,
                0.42..0.61,
            ),
        };
        let rgb = tint * r.random_range(shade);
        StandardMaterial {
            base_color: Color::srgb(rgb.x, rgb.y, rgb.z),
            perceptual_roughness: r.random_range(roughness),
            ..default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::IndoorLayout,
        materials::finish_slot,
        objects::{audit, build_manifest_object, computers},
    };

    #[test]
    fn selected_and_full_builds_bind_the_visible_opaque_lid_finishes() {
        let mut scene =
            IndoorManifest::generate_with_humans(310, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.objects = (0..64)
            .map(|s| audit::fixture(ObjectKind::Laptop, s))
            .collect();
        let mut selection = MaterialSelection {
            surfaces: default(),
            direct_surfaces: default(),
            finishes: default(),
        };
        for o in &scene.objects {
            for (surface, label) in build_manifest_object(&scene, o).parts.keys() {
                if let Some(slot) =
                    finish_slot(*surface, label).filter(|&s| LidStock::from_slot(s).is_some())
                {
                    selection.select_finish(*surface, slot);
                }
            }
        }
        assert_eq!(selection.finishes.len(), 4);
        let mut assets = Assets::<StandardMaterial>::default();
        let mut images = Assets::<Image>::default();
        let mut selected = BTreeMap::new();
        let mut full = BTreeMap::new();
        install(
            &scene,
            &mut images,
            &mut assets,
            Some(&selection),
            &mut selected,
        );
        assert!(images.is_empty());
        install(&scene, &mut images, &mut assets, None, &mut full);
        for stock in LidStock::ALL {
            let key = (Surface::Plastic, stock.slot());
            let a = assets.get(&selected[&key]).unwrap();
            let b = assets.get(&full[&key]).unwrap();
            assert_eq!(a.base_color, b.base_color);
            assert_eq!(a.perceptual_roughness, b.perceptual_roughness);
            assert_eq!(a.alpha_mode, AlphaMode::Opaque);
            assert_eq!(a.specular_transmission, 0.);
            assert_eq!(a.metallic, 0.);
            assert!(a.base_color_texture.is_none() && a.normal_map_texture.is_none());
            assert_eq!(a.emissive, LinearRgba::BLACK);
        }
        assert!(
            assets
                .get(&selected[&(Surface::Plastic, LidStock::Molded.slot())])
                .unwrap()
                .perceptual_roughness
                > assets
                    .get(&selected[&(Surface::Plastic, LidStock::Bonded.slot())])
                    .unwrap()
                    .perceptual_roughness
        );
        scene.generator_version = computers::lid::INTRODUCED - 1;
        let mut legacy = BTreeMap::new();
        install(&scene, &mut images, &mut assets, None, &mut legacy);
        assert!(legacy
            .keys()
            .all(|(_, s)| LidStock::from_slot(*s).is_none()));
    }
}
