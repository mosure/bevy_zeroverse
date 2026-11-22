//! Opaque pigmented polyethylene liners. The mesh supplies the gathered folds.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum FilmFinish {
    Charcoal,
    Pearl,
}

impl FilmFinish {
    fn slot(self) -> usize {
        match self {
            Self::Charcoal => 112,
            Self::Pearl => 113,
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Charcoal => "trash_can#finish112",
            Self::Pearl => "trash_can#finish113",
        }
    }
    fn material(self, seed: u64) -> StandardMaterial {
        let mut r = stream(seed, u64::from_le_bytes(*b"BinFilm ") ^ self.slot() as u64);
        let value = match self {
            Self::Charcoal => r.random_range(0.075..0.24),
            Self::Pearl => r.random_range(0.60..0.82),
        };
        let warmth = r.random_range(-0.025..0.035);
        StandardMaterial {
            base_color: Color::srgb(value * (1. + warmth), value, value * (1. - warmth)),
            perceptual_roughness: r.random_range(0.31..0.57),
            ..default()
        }
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Plastic && matches!(slot, 112 | 113)
}

pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    for finish in [FilmFinish::Charcoal, FilmFinish::Pearl] {
        let key = (Surface::Plastic, finish.slot());
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            variants.insert(key, materials.add(finish.material(scene.material_seed())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_two_film_stocks_are_opaque_dielectrics_and_require_no_maps() {
        for seed in 0..32 {
            for stock in [FilmFinish::Charcoal, FilmFinish::Pearl] {
                let m = stock.material(seed);
                assert_eq!(
                    super::super::finish_slot(Surface::Plastic, stock.label()),
                    Some(stock.slot())
                );
                assert_eq!(m.alpha_mode, AlphaMode::Opaque);
                assert_eq!(m.metallic, 0.);
                assert!(m.base_color_texture.is_none() && m.normal_map_texture.is_none());
                assert!((0.31..0.57).contains(&m.perceptual_roughness));
                let green = m.base_color.to_srgba().green;
                assert!((if stock == FilmFinish::Charcoal {
                    0.075..0.24
                } else {
                    0.60..0.82
                })
                .contains(&green));
            }
        }
    }
}
