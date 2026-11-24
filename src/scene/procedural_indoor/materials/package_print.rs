//! Printed drink packaging, drawn aluminium ends and moulded closures. Product lots
//! share one mipmapped label per room. Ink and substrate keep distinct optical responses.
use super::*;
use crate::scene::procedural_indoor::{
    clutter::beverages,
    layout::{stream, IndoorObject, ObjectKind},
    preparation::AssetStore,
};
mod artwork;
use artwork::label_pixels;
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

const FIRST: usize = 496;
const LOTS: usize = 6;
const ALLOY: usize = 504;
const CAP: usize = 505;
const SODA: usize = 508;
pub(crate) const ALUMINUM: &str = "drink_container#finish504";
pub(crate) const DRINK: &str = "drink_container#finish508";

#[derive(Clone, Copy)]
pub(crate) struct Print {
    bottle: bool,
    slot: usize,
}
impl Print {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"PackLot "),
        );
        Self {
            bottle: o.kind == ObjectKind::WaterBottle,
            slot: FIRST + r.random_range(0..LOTS),
        }
    }
    pub fn surface(self) -> Surface {
        if self.bottle {
            Surface::Plastic
        } else {
            Surface::Metal
        }
    }
    pub fn label(self) -> String {
        format!("drink_container#finish{}#uv_local", self.slot)
    }
}
fn cap_slot(o: &IndoorObject) -> usize {
    CAP + stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"CapDye  "),
    )
    .random_range(0..3)
}
pub(crate) fn cap(o: &IndoorObject) -> &'static str {
    [
        "drink_container#finish505",
        "drink_container#finish506",
        "drink_container#finish507",
    ][cap_slot(o) - CAP]
}
pub(super) fn supports(s: Surface, i: usize) -> bool {
    ((FIRST..FIRST + LOTS).contains(&i) && matches!(s, Surface::Metal | Surface::Plastic))
        || (s == Surface::Metal && i == ALLOY)
        || (s == Surface::Drink && i == SODA)
        || (s == Surface::Plastic && (CAP..CAP + 3).contains(&i))
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let keys: BTreeSet<_> = if let Some(s) = selection {
        s.finishes
            .iter()
            .filter(|&&(s, i)| supports(s, i))
            .copied()
            .collect()
    } else {
        let mut keys = BTreeSet::new();
        for o in scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::SodaCan | ObjectKind::WaterBottle))
        {
            let printed = if o.kind == ObjectKind::SodaCan {
                let p = beverages::can::parameters(o);
                keys.insert((Surface::Metal, ALLOY));
                if p.opened && p.end.fill_fraction.is_some() {
                    keys.insert((Surface::Drink, SODA));
                }
                !p.bare
            } else {
                keys.insert((Surface::Plastic, cap_slot(o)));
                beverages::parameters(o).style < 2
            };
            if printed {
                let p = Print::for_object(o);
                keys.insert((p.surface(), p.slot));
            }
        }
        keys
    };
    for (s, i) in keys {
        let mut r = stream(
            scene.material_seed(),
            u64::from_le_bytes(*b"PackCoat") ^ (s as u64).rotate_left(20) ^ i as u64,
        );
        let mat = if (FIRST..FIRST + LOTS).contains(&i) {
            let bottle = s == Surface::Plastic;
            let mut texture = mip_image(label_pixels(&mut r, bottle), 256, MapType::Color);
            if let ImageSampler::Descriptor(sampler) = &mut texture.sampler {
                sampler.address_mode_v = ImageAddressMode::ClampToEdge;
            }
            StandardMaterial {
                base_color_texture: Some(images.add(texture)),
                perceptual_roughness: r.random_range(if bottle { 0.37..0.64 } else { 0.27..0.50 }),
                clearcoat: r.random_range(0.16..0.43),
                clearcoat_perceptual_roughness: r.random_range(0.19..0.43),
                ..default()
            }
        } else {
            let (color, roughness, metallic) = match (s, i) {
                (Surface::Metal, ALLOY) => (
                    Color::srgb(0.88, 0.89, 0.90),
                    r.random_range(0.25..0.43),
                    0.97,
                ),
                (Surface::Drink, SODA) => {
                    let rgb = [
                        [0.16, 0.072, 0.032],
                        [0.33, 0.15, 0.043],
                        [0.52, 0.34, 0.11],
                        [0.29, 0.15, 0.18],
                    ][r.random_range(0..4)];
                    (
                        Color::srgb(rgb[0], rgb[1], rgb[2]),
                        r.random_range(0.10..0.20),
                        0.,
                    )
                }
                (_, CAP) => (
                    Color::hsl(r.random_range(0.0..360.), 0.09, r.random_range(0.74..0.93)),
                    r.random_range(0.44..0.66),
                    0.,
                ),
                (_, 506) => (
                    Color::hsl(r.random_range(0.0..360.), 0.15, r.random_range(0.10..0.24)),
                    r.random_range(0.47..0.68),
                    0.,
                ),
                _ => (
                    Color::hsl(
                        r.random_range(0.0..360.),
                        r.random_range(0.50..0.77),
                        r.random_range(0.29..0.52),
                    ),
                    r.random_range(0.42..0.61),
                    0.,
                ),
            };
            StandardMaterial {
                base_color: color,
                perceptual_roughness: roughness,
                metallic,
                ..default()
            }
        };
        variants.insert((s, i), materials.add(mat));
    }
}

#[cfg(test)]
mod tests;
