//! Small, shared finishes for paperboard jackets and liquid surfaces.
use super::super::preparation::AssetStore;
use super::*;
use rand::Rng;

pub(crate) const KRAFT_SLOT: usize = 400;
const FIRST_BREW: usize = 400;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum Brew {
    Coffee,
    Tea,
    MilkCoffee,
    Cocoa,
}

impl Brew {
    const ALL: [Self; 4] = [Self::Coffee, Self::Tea, Self::MilkCoffee, Self::Cocoa];

    pub(crate) fn sample(r: &mut impl Rng) -> Self {
        match r.random_range(0..10) {
            0..=3 => Self::Coffee,
            4..=5 => Self::Tea,
            6..=8 => Self::MilkCoffee,
            _ => Self::Cocoa,
        }
    }

    pub(crate) fn slot(self) -> usize {
        FIRST_BREW + self as usize
    }

    fn color(self) -> [f32; 3] {
        match self {
            Self::Coffee => [0.12, 0.066, 0.031],
            Self::Tea => [0.36, 0.16, 0.049],
            Self::MilkCoffee => [0.59, 0.38, 0.22],
            Self::Cocoa => [0.34, 0.19, 0.12],
        }
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    (surface == Surface::Paper && slot == KRAFT_SLOT)
        || (surface == Surface::Drink && (FIRST_BREW..FIRST_BREW + Brew::ALL.len()).contains(&slot))
}

pub(super) fn install(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let stock = std::iter::once((Surface::Paper, KRAFT_SLOT, [0.67, 0.49, 0.32]))
        .chain(Brew::ALL.map(|b| (Surface::Drink, b.slot(), b.color())));
    for (surface, slot, rgb) in stock {
        if selection.is_some_and(|s| !s.finishes.contains(&(surface, slot))) {
            continue;
        }
        let mut r = super::super::layout::stream(
            scene.material_seed(),
            0x43555046494e + slot as u64 + ((surface as u64) << 16),
        );
        let gain = r.random_range(0.82..1.15);
        let [red, green, blue] = rgb.map(|x| x * gain);
        // Paperboard inherits the actual paper fibre and roughness maps.
        let mut m = materials.get(&base[surface as usize]).unwrap().clone();
        m.base_color = Color::srgb(red, green, blue);
        if surface == Surface::Drink {
            m.perceptual_roughness = r.random_range(0.13..0.24);
            m.metallic = 0.;
        }
        variants.insert((surface, slot), materials.add(m));
    }
}
