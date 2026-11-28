//! Court paint, net stock and pigmented seating. The openings between the cords are empty
//! geometry in both the optical and annotation passes.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::BTreeMap;
mod court_paint;
pub(crate) const LINE: &str = "floor#finish965";
pub(crate) const TAPE: &str = "sports_equipment#finish966";
pub(crate) const CORD: &str = "sports_equipment#finish967";
pub(crate) const RIM: &str = "sports_equipment#finish1008";
pub(crate) const ENAMEL: &str = "sports_equipment#finish1009";
pub(crate) const PRACTICE: &str = "floor#finish1010";
const PRACTICE_STOCK: (Surface, usize) = (Surface::Paint, 1010);
const GOAL: [(Surface, usize); 2] = [(Surface::Metal, 1008), (Surface::Paint, 1009)];
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
    (s, slot) == PRACTICE_STOCK
        || STOCK.contains(&(s, slot))
        || GOAL.contains(&(s, slot))
        || stock_parent(s, slot).is_some()
}
pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let goal = selection.map_or_else(
        || {
            scene.objects.iter().any(|o| {
                o.kind == crate::scene::procedural_indoor::layout::ObjectKind::BasketballGoal
            })
        },
        |s| GOAL.iter().any(|key| s.finishes.contains(key)),
    );
    if goal {
        let mut r = stream(scene.material_seed(), u64::from_le_bytes(*b"GoalCoat"));
        let white = r.random_range(0.79..0.95);
        for (key, color, roughness) in [
            (
                GOAL[0],
                Color::srgb(
                    r.random_range(0.76..0.97),
                    r.random_range(0.20..0.39),
                    0.037,
                ),
                r.random_range(0.28..0.45),
            ),
            (
                GOAL[1],
                Color::srgb(white, white * 0.987, white * 0.95),
                r.random_range(0.35..0.55),
            ),
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
    }
    if !goal
        && selection.map_or_else(
            || {
                scene
                    .program
                    .as_ref()
                    .is_none_or(|p| p.volleyball.is_none())
            },
            |s| {
                !s.finishes.contains(&PRACTICE_STOCK)
                    && !STOCK.iter().any(|key| s.finishes.contains(key))
                    && ![Surface::Leather, Surface::Plastic]
                        .into_iter()
                        .any(|surface| {
                            s.finishes
                                .range((surface, PAD_FIRST)..(surface, PAD_FIRST + variants::COUNT))
                                .next()
                                .is_some()
                        })
            },
        )
    {
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
    let line = scene
        .program
        .as_ref()
        .filter(|p| {
            p.volleyball.is_some()
                && scene.generator_version
                    >= crate::scene::procedural_indoor::objects::public::basketball::VERSION
        })
        .map_or(line, |p| {
            court_paint::on_floor(
                scene.material_seed(),
                p.materials[Surface::Floor as usize].color,
                line,
            )
        });
    let tape = r.random_range(0.78..0.95);
    let cord = r.random_range(0.048..0.15);
    if let Some(p) = scene.program.as_ref().filter(|p| {
        selection.map_or_else(
            || !p.free_throw_marks.is_empty(),
            |s| s.finishes.contains(&PRACTICE_STOCK),
        )
    }) {
        let floor = p.materials[Surface::Floor as usize].color;
        let mut pigment = stream(scene.material_seed(), u64::from_le_bytes(*b"HoopDyes"));
        variants.insert(
            PRACTICE_STOCK,
            materials.add(StandardMaterial {
                base_color: court_paint::secondary(floor, line, &mut pigment),
                perceptual_roughness: pigment.random_range(0.34..0.53),
                ..default()
            }),
        );
    }
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
