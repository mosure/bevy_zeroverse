//! Anodized or molded cases and their color-matched, dielectric touch surfaces.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

const CASE: usize = 126;
const PAD: usize = 130;
const LOTS: usize = 4;

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    matches!(s, Surface::Metal | Surface::Plastic) && (CASE..CASE + LOTS).contains(&slot)
        || s == Surface::Plastic && (PAD..PAD + LOTS).contains(&slot)
}

pub(crate) struct Housing {
    pub surface: Surface,
    case_slot: usize,
}
impl Housing {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.finish_seed.unwrap_or(o.seed), 0x434153454c4f5453);
        let lot = match r.random_range(0..12) {
            0..=4 => 0,
            5..=8 => 1,
            9 => 2,
            _ => 3,
        };
        let plastic = if o.kind == ObjectKind::Laptop {
            o.variant % 2 == 1
        } else {
            r.random_bool(if o.kind == ObjectKind::Mouse {
                0.94
            } else {
                0.65
            })
        };
        Self {
            surface: if plastic {
                Surface::Plastic
            } else {
                Surface::Metal
            },
            case_slot: CASE + lot,
        }
    }
    pub(crate) fn case_label(&self, o: &IndoorObject) -> String {
        format!("{}#finish{}", o.kind.class_name(), self.case_slot)
    }
    pub(crate) fn pad_label(&self) -> String {
        format!("laptop#finish{}", self.case_slot - CASE + PAD)
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let keys: BTreeSet<_> = if let Some(selection) = selection {
        selection
            .finishes
            .iter()
            .filter(|&&(s, i)| supports(s, i))
            .copied()
            .collect()
    } else {
        let mut keys = BTreeSet::new();
        for o in scene.objects.iter().filter(|o| {
            matches!(
                o.kind,
                ObjectKind::Laptop | ObjectKind::Keyboard | ObjectKind::PcTower | ObjectKind::Mouse
            )
        }) {
            let p = Housing::for_object(o);
            keys.insert((p.surface, p.case_slot));
            if o.kind == ObjectKind::Laptop {
                keys.insert((Surface::Plastic, p.case_slot - CASE + PAD));
            }
        }
        keys
    };
    for (s, slot) in keys {
        let pad = slot >= PAD;
        let lot = slot - if pad { PAD } else { CASE };
        let mut r = stream(scene.material_seed(), 0x4b42434841530000 | lot as u64);
        let v = match lot {
            0 => r.random_range(0.15..0.32),
            1 => r.random_range(0.52..0.75),
            2 => r.random_range(0.75..0.90),
            _ => r.random_range(0.31..0.51),
        };
        let color = if lot == 3 {
            [
                Vec3::new(0.76, 0.86, 1.),
                Vec3::new(0.80, 1., 0.90),
                Vec3::new(1., 0.83, 0.75),
            ][r.random_range(0..3)]
                * v
        } else {
            Vec3::new(
                v,
                v * r.random_range(0.96..1.0),
                v * r.random_range(0.94..1.0),
            )
        };
        let roughness = if pad {
            r.random_range(0.31..0.51)
        } else {
            r.random_range(0.37..0.63)
        };
        let color = color * if pad { r.random_range(0.82..0.96) } else { 1. };
        variants.insert(
            (s, slot),
            materials.add(StandardMaterial {
                base_color: Color::srgb(color.x, color.y, color.z),
                perceptual_roughness: roughness,
                metallic: if s == Surface::Metal { 0.88 } else { 0. },
                ..default()
            }),
        );
    }
}
