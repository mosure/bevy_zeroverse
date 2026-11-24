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
const BALL: usize = 801;

/// Polished dielectric stock for an optical trackball.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BallFinish {
    Garnet,
    Cobalt,
    Graphite,
    Pearl,
}

impl BallFinish {
    const ALL: [Self; 4] = [Self::Garnet, Self::Cobalt, Self::Graphite, Self::Pearl];
    pub(crate) fn sample(r: &mut impl Rng) -> Self {
        Self::ALL[r.random_range(0..4)]
    }
    pub(crate) fn label(self) -> &'static str {
        [
            "mouse#finish801",
            "mouse#finish802",
            "mouse#finish803",
            "mouse#finish804",
        ][self as usize]
    }
    fn material(self, seed: u64) -> StandardMaterial {
        let mut r = stream(seed, u64::from_le_bytes(*b"OptiBall") ^ self as u64);
        let tint = match self {
            Self::Garnet => Vec3::new(0.55, 0.084, 0.11),
            Self::Cobalt => Vec3::new(0.085, 0.26, 0.53),
            Self::Graphite => Vec3::new(0.15, 0.17, 0.19),
            Self::Pearl => Vec3::new(0.72, 0.74, 0.76),
        } * r.random_range(0.84..1.14);
        StandardMaterial {
            base_color: Color::srgb(tint.x, tint.y, tint.z),
            perceptual_roughness: r.random_range(0.14..0.29),
            ..default()
        }
    }
}

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    matches!(s, Surface::Metal | Surface::Plastic) && (CASE..CASE + LOTS).contains(&slot)
        || s == Surface::Plastic
            && ((PAD..PAD + LOTS).contains(&slot) || (BALL..BALL + LOTS).contains(&slot))
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
            r.random_bool(
                if matches!(o.kind, ObjectKind::Mouse | ObjectKind::Printer) {
                    0.94
                } else {
                    0.65
                },
            )
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
    fn trim_slot(&self) -> usize {
        if self.case_slot == CASE {
            CASE + 3
        } else {
            CASE
        }
    }
    pub(crate) fn trim_label(&self, o: &IndoorObject) -> String {
        format!("{}#finish{}", o.kind.class_name(), self.trim_slot())
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
            ) || (o.kind == ObjectKind::Printer
                && scene.generator_version
                    >= crate::scene::procedural_indoor::objects::utilities::printer::OUTPUT_VERSION)
        }) {
            let p = Housing::for_object(o);
            keys.insert((p.surface, p.case_slot));
            use crate::scene::procedural_indoor::objects::desktop::{self, pointer};
            if o.kind == ObjectKind::Mouse && scene.generator_version >= pointer::INTRODUCED {
                keys.insert((Surface::Plastic, p.trim_slot()));
                let input = desktop::parameters(o);
                if let Some(finish) = input
                    .trackball
                    .then(|| pointer::PointerProgram::sample(o, &input).control.ball())
                    .flatten()
                {
                    keys.insert((Surface::Plastic, BALL + finish as usize));
                }
            } else if o.kind == ObjectKind::Printer {
                keys.insert((Surface::Plastic, p.trim_slot()));
            }
            if o.kind == ObjectKind::Laptop {
                keys.insert((Surface::Plastic, p.case_slot - CASE + PAD));
            }
        }
        keys
    };
    for (s, slot) in keys {
        if (BALL..BALL + LOTS).contains(&slot) {
            let finish = BallFinish::ALL[slot - BALL];
            variants.insert(
                (s, slot),
                materials.add(finish.material(scene.material_seed())),
            );
            continue;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::IndoorLayout,
        objects::{audit, build_manifest_object, desktop},
    };

    #[test]
    fn polished_trackballs_resolve_identically_in_selected_and_full_materials() {
        let mut scene =
            IndoorManifest::generate_with_humans(93401, IndoorLayout::Mixed, 0.65, 1, 0.).unwrap();
        scene.objects = (0..160)
            .map(|seed| audit::fixture(ObjectKind::Mouse, seed))
            .collect();
        let mut selection = MaterialSelection {
            surfaces: default(),
            finishes: default(),
            direct_surfaces: default(),
        };
        for o in &scene.objects {
            let assembly = build_manifest_object(&scene, o);
            for (surface, label) in assembly.parts.keys() {
                if let Some(slot) = label.split_once("#finish").map(|(_, s)| s.parse().unwrap()) {
                    selection.select_finish(*surface, slot);
                }
            }
        }
        let mut assets = Assets::<StandardMaterial>::default();
        let (mut all, mut selected) = (BTreeMap::new(), BTreeMap::new());
        install(&scene, &mut assets, None, &mut all);
        install(&scene, &mut assets, Some(&selection), &mut selected);
        assert!(all.keys().eq(selected.keys()));
        for slot in BALL..BALL + 4 {
            let key = (Surface::Plastic, slot);
            let full = assets.get(&all[&key]).unwrap();
            let partial = assets.get(&selected[&key]).unwrap();
            assert_eq!(full.base_color, partial.base_color);
            assert_eq!(full.perceptual_roughness, partial.perceptual_roughness);
            assert!((0.14..0.29).contains(&full.perceptual_roughness));
            assert_eq!(full.metallic, 0.);
        }
        scene.generator_version = desktop::pointer::INTRODUCED - 1;
        let mut legacy = BTreeMap::new();
        install(&scene, &mut assets, None, &mut legacy);
        assert!(legacy.keys().all(|&(_, slot)| slot < BALL));
    }
}
