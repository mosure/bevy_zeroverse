//! Court paint, net stock and pigmented seating. The openings between the cords are empty
//! geometry in both the optical and annotation passes.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;
pub(crate) const LINE: &str = "floor#finish965";
pub(crate) const TAPE: &str = "sports_equipment#finish966";
pub(crate) const CORD: &str = "sports_equipment#finish967";
const STOCK: [(Surface, usize); 3] = [
    (Surface::Paint, 965),
    (Surface::Fabric, 966),
    (Surface::Rubber, 967),
];
const PAD_FIRST: usize = 968;
pub(crate) fn pad_slot(r: &mut impl Rng) -> usize {
    PAD_FIRST + r.random_range(0..variants::COUNT)
}
pub(super) fn stock_parent(s: Surface, slot: usize) -> Option<usize> {
    (matches!(s, Surface::Leather | Surface::Plastic)
        && (PAD_FIRST..PAD_FIRST + variants::COUNT).contains(&slot))
    .then(|| slot - PAD_FIRST)
}
pub(super) fn supports(s: Surface, slot: usize) -> bool {
    STOCK.contains(&(s, slot)) || stock_parent(s, slot).is_some()
}
pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if selection.map_or_else(
        || {
            scene
                .program
                .as_ref()
                .is_none_or(|p| p.volleyball.is_none())
        },
        |s| {
            !STOCK.iter().any(|key| s.finishes.contains(key))
                && ![Surface::Leather, Surface::Plastic]
                    .into_iter()
                    .any(|surface| {
                        s.finishes
                            .range((surface, PAD_FIRST)..(surface, PAD_FIRST + variants::COUNT))
                            .next()
                            .is_some()
                    })
        },
    ) {
        return;
    }
    let mut r = stream(scene.material_seed(), u64::from_le_bytes(*b"VBFinish"));
    let warm = r.random_range(0.75..0.94);
    let line = if r.random_bool(0.73) {
        Color::srgb(warm, warm, warm * 0.97)
    } else {
        Color::srgb(
            r.random_range(0.13..0.28),
            r.random_range(0.25..0.41),
            r.random_range(0.43..0.62),
        )
    };
    let tape = r.random_range(0.78..0.95);
    let cord = r.random_range(0.048..0.15);
    for (key, color, roughness) in [
        (STOCK[0], line, 0.41),
        (STOCK[1], Color::srgb(tape, tape, tape * 0.98), 0.81),
        (STOCK[2], Color::srgb(cord, cord * 1.05, cord * 1.08), 0.84),
    ] {
        if selection.is_none_or(|s| s.finishes.contains(&key)) {
            variants.insert(
                key,
                materials.add(StandardMaterial {
                    base_color: color,
                    perceptual_roughness: roughness,
                    ..default()
                }),
            );
        }
    }
    // Vinyl pole padding is pigmented independently of furniture. Reuse the
    // installed grain and normal/roughness maps from the same leather stock.
    for surface in [Surface::Leather, Surface::Plastic] {
        if selection.is_none()
            && surface == Surface::Plastic
            && !scene.objects.iter().any(|o| {
                o.kind == crate::scene::procedural_indoor::layout::ObjectKind::BleacherBank
            })
        {
            continue;
        }
        for slot in PAD_FIRST..PAD_FIRST + variants::COUNT {
            let key = (surface, slot);
            if selection.is_some_and(|s| !s.finishes.contains(&key)) {
                continue;
            }
            let Some(base) = variants
                .get(&(surface, slot - PAD_FIRST))
                .and_then(|h| materials.get(h))
                .cloned()
            else {
                continue;
            };
            let mut pigment = stream(
                scene.material_seed(),
                (if surface == Surface::Leather {
                    u64::from_le_bytes(*b"VBPadDye")
                } else {
                    u64::from_le_bytes(*b"SeatDyes")
                }) ^ slot as u64,
            );
            let hue = [215., 202., 353., 30., 145., 248.][pigment.random_range(0..6)]
                + pigment.random_range(-9.0..9.0);
            let sat = if pigment.random_bool(0.84) {
                pigment.random_range(0.45..0.78)
            } else {
                pigment.random_range(0.03..0.19)
            };
            variants.insert(
                key,
                materials.add(StandardMaterial {
                    base_color: Color::hsl(hue, sat, pigment.random_range(0.20..0.43)),
                    perceptual_roughness: pigment.random_range(0.44..0.66),
                    ..base
                }),
            );
        }
    }
}
