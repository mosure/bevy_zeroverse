//! Small manufactured fittings use a shared optical stock and no image uploads.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::{sprinkler::FITTED_HEADS_VERSION, CeilingSystem},
    layout::stream,
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    Brass,
    Nickel,
    Enamel,
    Ruby,
    Amber,
}

impl Finish {
    const ALL: [Self; 5] = [
        Self::Brass,
        Self::Nickel,
        Self::Enamel,
        Self::Ruby,
        Self::Amber,
    ];
    pub(crate) fn key(self) -> (Surface, &'static str) {
        match self {
            Self::Brass => (Surface::Metal, "ceiling#finish796"),
            Self::Nickel => (Surface::Metal, "ceiling#finish797"),
            Self::Enamel => (Surface::Metal, "ceiling#finish798"),
            Self::Ruby => (Surface::Plastic, "ceiling#finish799"),
            Self::Amber => (Surface::Plastic, "ceiling#finish800"),
        }
    }
    fn slot(self) -> usize {
        796 + self as usize
    }
    fn material(self, seed: u64) -> StandardMaterial {
        let mut r = stream(seed, u64::from_le_bytes(*b"HeadCoat") ^ self.slot() as u64);
        let (tint, metallic, roughness) = match self {
            Self::Brass => (Vec3::new(0.87, 0.69, 0.39), 1., 0.24..0.42),
            Self::Nickel => (Vec3::new(0.81, 0.82, 0.82), 1., 0.15..0.32),
            Self::Enamel => (Vec3::new(0.89, 0.88, 0.85), 0., 0.31..0.48),
            // At room-view scales an opaque optical proxy keeps the small liquid
            // column aligned in RGB, depth and semantics without a transmission pass.
            Self::Ruby => (Vec3::new(0.72, 0.065, 0.044), 0., 0.12..0.24),
            Self::Amber => (Vec3::new(0.93, 0.47, 0.092), 0., 0.12..0.24),
        };
        let tint = tint * r.random_range(0.86..1.03);
        StandardMaterial {
            base_color: Color::srgb(tint.x, tint.y, tint.z),
            metallic,
            perceptual_roughness: r.random_range(roughness),
            ..default()
        }
    }
}
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(
        (surface, slot),
        (Surface::Metal, 796..=798) | (Surface::Plastic, 799..=800)
    )
}
pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < FITTED_HEADS_VERSION
        || (selection.is_none()
            && !scene.ceiling_installation.as_ref().is_some_and(|c| {
                c.modules
                    .iter()
                    .any(|m| m.system == CeilingSystem::SprinklerMain)
            }))
    {
        return;
    }
    for finish in Finish::ALL {
        let key = (finish.key().0, finish.slot());
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            variants.insert(key, materials.add(finish.material(scene.material_seed())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fittings_resolve_shared_ceiling_finishes_and_keep_bulbs_nonemissive() {
        for f in Finish::ALL {
            let (surface, label) = f.key();
            assert!(supports(surface, f.slot()));
            assert_eq!(super::super::finish_slot(surface, label), Some(f.slot()));
            assert_eq!(
                crate::scene::procedural_indoor::objects::part_label(label),
                "ceiling"
            );
            let m = f.material(71);
            assert_eq!(
                m.metallic,
                if matches!(f, Finish::Brass | Finish::Nickel) {
                    1.
                } else {
                    0.
                }
            );
            assert_eq!(m.alpha_mode, AlphaMode::Opaque);
            assert_eq!(m.emissive, LinearRgba::BLACK);
            assert!(m.base_color_texture.is_none() && m.normal_map_texture.is_none());
        }
    }
}
