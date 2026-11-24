//! Die-cut paperboard, its unprinted cut edge, and shared embossed tissue stock.
//! Only realized prints allocate a filtered tile; a solid carton needs no image.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::containers::tissue::{self, Case, Draw},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};
mod print;
#[cfg(test)]
mod tests;

const FIRST: usize = 512;
const LOTS: usize = 4;
const CUT: usize = 516;
const FIBER: usize = 517;
pub(crate) const KRAFT: &str = "box#finish516";
pub(crate) const PLY: &str = "paper#finish517";
fn slot(o: &IndoorObject) -> usize {
    FIRST
        + stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"TissLot "),
        )
        .random_range(0..LOTS)
}
pub(crate) fn carton_label(o: &IndoorObject) -> String {
    format!("box#finish{}#uv_local", slot(o))
}
pub(crate) fn pattern(o: &IndoorObject, material_seed: u64) -> print::Pattern {
    print::pattern(material_seed, slot(o))
}
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Paper && (FIRST..=FIBER).contains(&slot)
}
fn ply(seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"PlyPress"));
    let value = r.random_range(0.875..0.97);
    let warmth = r.random_range(-0.006..0.030);
    let pulp = Vec3::new(value + 0.35 * warmth, value, value - warmth);
    let pitch = Vec2::new(r.random_range(0.015..0.028), r.random_range(0.013..0.025));
    let emboss = r.random_range(0.000025..0.00009);
    let salt = r.random();
    sheet::sheet_material::<32>(images, pitch, 0., 1., |uv| {
        let row = (uv.y * 2.).floor();
        let dot = Vec2::new(
            (uv.x * 2. + row * 0.5).fract() - 0.5,
            (uv.y * 2.).fract() - 0.5,
        );
        let pit = (1. - (dot * Vec2::new(3.8, 5.4)).length_squared()).clamp(0., 1.);
        let grain = periodic_noise(uv.x, uv.y, 13, 7, salt);
        (
            pulp * (0.990 + 0.010 * grain),
            grain * 0.000020 - emboss * pit * pit,
            0.87 + 0.10 * grain,
        )
    })
}
pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    mats: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let mut slots = BTreeSet::new();
    if let Some(s) = selection {
        slots.extend(
            s.finishes
                .iter()
                .filter_map(|&(surface, i)| supports(surface, i).then_some(i)),
        );
    } else {
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::TissueBox)
        {
            let Some(d) = tissue::parameters(o).dispenser else {
                continue;
            };
            if d.case == Case::FoldingCarton {
                slots.insert(slot(o));
            }
            if d.draw != Draw::Sealed {
                slots.insert(FIBER);
                if d.case == Case::FoldingCarton {
                    slots.insert(CUT);
                }
            }
        }
    }
    let seed = scene.material_seed();
    for i in slots {
        let mat = match i {
            FIBER => ply(seed, images),
            CUT => {
                let k = stream(seed, u64::from_le_bytes(*b"CutBoard")).random_range(0.57..0.72);
                StandardMaterial {
                    base_color: Color::srgb(k, k * 0.84, k * 0.64),
                    perceptual_roughness: 0.94,
                    ..default()
                }
            }
            _ => print::material(seed, i, images),
        };
        variants.insert((Surface::Paper, i), mats.add(mat));
    }
}
