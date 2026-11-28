//! Pigmented closed-cell foam and rubber. Metric relief, pigment and roughness
//! share the same tooling pattern. Each used stock is baked once per room.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::public::mats,
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;
mod edge;
pub(crate) use edge::CHART_TURNS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Stock {
    Pebbled,
    Ribbed,
    Diamond,
    NaturalCork,
    ToastedCork,
}

impl Stock {
    pub(crate) fn rubber(i: usize) -> Self {
        [Self::Pebbled, Self::Ribbed, Self::Diamond][i]
    }
    pub(crate) fn for_mat(o: &IndoorObject, folded: bool) -> Self {
        let mut r = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"MatStock"),
        );
        // Many studios purchase a common finish; personal mats vary within a class.
        let mut own = stream(o.seed, u64::from_le_bytes(*b"OwnSheet"));
        let lot = if own.random_bool(0.32) {
            &mut own
        } else {
            &mut r
        };
        if !folded && lot.random_bool(0.18) {
            if lot.random_bool(0.65) {
                Self::NaturalCork
            } else {
                Self::ToastedCork
            }
        } else {
            Self::rubber(lot.random_range(0..3))
        }
    }
    pub(crate) fn edge_label(self) -> &'static str {
        match self {
            Self::Pebbled => "floormat#finish1005#uv_local",
            Self::Ribbed => "floormat#finish1006#uv_local",
            Self::Diamond => "floormat#finish1007#uv_local",
            _ => unreachable!("rack stocks are rollable elastomers"),
        }
    }
    pub(crate) fn key(self) -> (Surface, &'static str) {
        match self {
            Self::Pebbled => (Surface::Rubber, "floormat#finish1002#uv_local"),
            Self::Ribbed => (Surface::Rubber, "floormat#finish1003#uv_local"),
            Self::Diamond => (Surface::Rubber, "floormat#finish1004#uv_local"),
            Self::NaturalCork => (Surface::FabricAlt, "floormat#finish652#uv_local"),
            Self::ToastedCork => (Surface::FabricAlt, "floormat#finish653#uv_local"),
        }
    }
    /// An equipment instance can use the same baked cut as a mat while its
    /// pixels retain their own class. The material key ignores the class prefix.
    pub(crate) fn equipment_key(self) -> (Surface, &'static str) {
        match self {
            Self::Pebbled => (Surface::Rubber, "fitness_equipment#finish1002#uv_local"),
            Self::Ribbed => (Surface::Rubber, "fitness_equipment#finish1003#uv_local"),
            Self::Diamond => (Surface::Rubber, "fitness_equipment#finish1004#uv_local"),
            Self::NaturalCork => (Surface::FabricAlt, "fitness_equipment#finish652#uv_local"),
            Self::ToastedCork => (Surface::FabricAlt, "fitness_equipment#finish653#uv_local"),
        }
    }
}

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    s == Surface::Rubber && (1002..=1007).contains(&slot)
}

fn pigment(r: &mut impl Rng) -> Vec3 {
    let dyes = [
        [0.15, 0.18, 0.19],
        [0.19, 0.33, 0.37],
        [0.34, 0.23, 0.37],
        [0.21, 0.40, 0.33],
        [0.53, 0.30, 0.25],
        [0.24, 0.31, 0.48],
        [0.62, 0.54, 0.43],
        [0.47, 0.36, 0.49],
    ];
    Vec3::from_array(dyes[r.random_range(0..dyes.len())]) * r.random_range(0.84..1.14)
}

fn material(seed: u64, index: usize, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"MatPress") + index as u64);
    let dye = pigment(&mut r);
    let period = Vec2::new(r.random_range(0.034..0.047), r.random_range(0.034..0.047));
    let mut budget = 16 * 1024;
    let pores = field::PreparedPeriodic::new([23, 19], r.random(), &mut budget);
    let grain = field::PreparedPeriodic::new([9, 11], r.random(), &mut budget);
    let depth = r.random_range(0.000075..0.00022);
    sheet::sheet_material::<64>(images, period, 0., 1., |p| {
        let micro = pores.sample(p.x, p.y);
        let patch = grain.sample(p.x, p.y);
        let ridge = |phase: f32| 0.5 + 0.5 * (std::f32::consts::TAU * phase).cos();
        let tool = match index {
            0 => field::smooth((patch + 0.30 * micro - 0.35) * 1.65),
            1 => ridge(8. * p.y + 0.11 * patch).powi(2),
            _ => ridge(5. * p.x + 5. * p.y).min(ridge(5. * p.x - 5. * p.y)),
        };
        (
            dye * (0.94 + 0.045 * patch + 0.035 * tool),
            depth * (tool + 0.23 * micro),
            0.83 - 0.12 * tool + 0.044 * micro,
        )
    })
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < mats::VERSION {
        return;
    }
    let mut used = [false; 6];
    if let Some(s) = selection {
        for (i, flag) in used.iter_mut().enumerate() {
            *flag = s.finishes.contains(&(Surface::Rubber, 1002 + i));
        }
    } else {
        let mut mark = |stock: Stock, rolled: bool| {
            if let Some(i) = [Stock::Pebbled, Stock::Ribbed, Stock::Diamond]
                .iter()
                .position(|s| *s == stock)
            {
                used[i] = true;
                if rolled {
                    used[3 + i] = true;
                }
            }
        };
        for o in &scene.objects {
            match o.kind {
                ObjectKind::ExerciseMat => mark(mats::PracticeMat::for_object(o).facing, false),
                ObjectKind::YogaBlock => {
                    let b = mats::props::Block::for_object(o);
                    mark(b.facing, false);
                    if let Some(stock) = b.inlay {
                        mark(stock, false);
                    }
                }
                ObjectKind::ExerciseMatRack => {
                    for roll in mats::MatRack::for_object(o).rolls {
                        mark(roll.stock, true);
                    }
                }
                _ => {}
            }
        }
    }
    let mut edge_maps = None;
    for (i, _) in used.into_iter().enumerate().filter(|(_, used)| *used) {
        let stock = if i < 3 {
            material(scene.material_seed(), i, images)
        } else {
            let [normal, orm] = edge_maps.get_or_insert_with(|| edge::maps(images));
            let mut r = stream(
                scene.material_seed(),
                u64::from_le_bytes(*b"MatPress") + (i - 3) as u64,
            );
            let core = pigment(&mut r).lerp(Vec3::splat(0.72), 0.13);
            StandardMaterial {
                base_color: Color::srgb(core.x, core.y, core.z),
                metallic: 0.,
                perceptual_roughness: 1.,
                normal_map_texture: Some(normal.clone()),
                metallic_roughness_texture: Some(orm.clone()),
                ..default()
            }
        };
        variants.insert((Surface::Rubber, 1002 + i), materials.add(stock));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn practice_grip_maps_are_metric_dielectric_repeatable_and_installed_on_demand() {
        fn baked(seed: u64, slot: usize) -> [Vec<u8>; 3] {
            let mut images = Assets::<Image>::default();
            let m = material(seed, slot, &mut images);
            assert_eq!((m.metallic, m.perceptual_roughness), (0., 1.));
            let period = m.uv_transform.matrix2.x_axis.length().recip();
            assert!((0.033..0.048).contains(&period));
            let maps = [
                (m.base_color_texture, true),
                (m.normal_map_texture, false),
                (m.metallic_roughness_texture, false),
            ]
            .map(|(h, srgb)| {
                let i = images.remove(h.unwrap().id()).unwrap();
                assert_eq!(i.texture_descriptor.format.is_srgb(), srgb);
                assert_eq!(i.texture_descriptor.mip_level_count, 7);
                i.data.unwrap()
            });
            assert!(maps[2]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 255 && p[2] == 0 && (179..=224).contains(&p[1])));
            let normals = &maps[1][..64 * 64 * 4];
            assert!(normals
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0].abs_diff(128) > 3 || p[1].abs_diff(128) > 3));
            assert!(maps[1].as_chunks::<4>().0.iter().all(|p| {
                let n = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - 1.;
                (n.length() - 1.).abs() < 0.015 && n.z > 0.80
            }));
            maps
        }
        for i in 0..3 {
            let maps = baked(78, i);
            assert!(maps == baked(78, i) && maps != baked(79, i) && maps != baked(78, (i + 1) % 3));
        }
        let scene =
            IndoorManifest::generate_with_humans(1, IndoorLayout::YogaStudio, 0.65, 0, 0.).unwrap();
        let key = (Surface::Rubber, 1003);
        let mut selection = MaterialSelection {
            surfaces: [key.0].into(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let mut vars = BTreeMap::new();
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut vars,
        );
        assert!(vars.is_empty() && images.is_empty());
        selection.finishes.insert(key);
        assert!(registered_finish(key.0, key.1) && selection.omits_parent_maps(&scene, key.0));
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut vars,
        );
        assert_eq!((vars.len(), materials.len(), images.len()), (1, 1, 3));
        selection.finishes = (1005..=1007).map(|i| (Surface::Rubber, i)).collect();
        install(
            &scene,
            &mut images,
            &mut materials,
            Some(&selection),
            &mut vars,
        );
        assert_eq!((vars.len(), materials.len(), images.len()), (4, 4, 5));
        let edge_maps = |i| {
            let m = materials.get(&vars[&(Surface::Rubber, i)]).unwrap();
            (
                m.normal_map_texture.as_ref().unwrap().id(),
                m.metallic_roughness_texture.as_ref().unwrap().id(),
            )
        };
        assert_eq!(edge_maps(1005), edge_maps(1006));
        assert_eq!(edge_maps(1005), edge_maps(1007));
        let (normal, orm) = edge_maps(1005);
        assert_eq!(
            images
                .get(normal)
                .unwrap()
                .texture_descriptor
                .mip_level_count,
            8
        );
        assert!(!images
            .get(normal)
            .unwrap()
            .texture_descriptor
            .format
            .is_srgb());
        let data = images.get(orm).unwrap().data.as_ref().unwrap();
        assert!(data
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[0] == 255 && p[2] == 0 && (174..=213).contains(&p[1])));
    }
}
