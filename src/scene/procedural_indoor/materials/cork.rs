//! Cut agglomerated cork. The same irregular chips drive pigment, binder seams
//! and metric relief. A room bakes only the stock referenced by its boards.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::equipment::board_section::{Facing, TACKBOARD_VERSION},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const NATURAL: usize = 652;
const TOASTED: usize = 653;

pub(crate) fn slot(facing: Facing) -> Option<usize> {
    match facing {
        Facing::Textile => None,
        Facing::NaturalCork => Some(NATURAL),
        Facing::ToastedCork => Some(TOASTED),
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::FabricAlt && matches!(slot, NATURAL | TOASTED)
}

fn stock(seed: u64, slot: usize, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"CorkChip") ^ slot as u64);
    let period = r.random_range(0.105..0.19);
    let spacing = r.random_range(0.0032..0.008);
    let cells = field::PreparedCellular::new(
        [
            field::frequency(period, spacing),
            field::frequency(period, spacing * r.random_range(0.82..1.18)),
        ],
        r.random(),
    );
    let mut budget = 48 * 1024;
    let [warp_x, warp_y, pores] = [[11, 9], [9, 13], [73, 69]]
        .map(|count| field::PreparedPeriodic::new(count, r.random(), &mut budget));
    let toast = if slot == TOASTED {
        r.random_range(0.54..0.92)
    } else {
        r.random_range(0.02..0.49)
    };
    let pigment = Vec3::new(0.84, 0.69, 0.48).lerp(Vec3::new(0.48, 0.32, 0.21), toast);
    let contrast = r.random_range(0.17..0.38);
    let relief = r.random_range(0.00018..0.00049);
    sheet::sheet_material::<128>(images, Vec2::splat(period), 0., 1., |p| {
        let warp = Vec2::new(warp_x.sample(p.x, p.y), warp_y.sample(p.x, p.y)) - 0.5;
        let cell = cells.sample_edges((p + warp * 0.033).to_array());
        let pore = pores.sample(p.x, p.y);
        let seam = 1. - field::smooth(cell.edge / 0.22);
        let fleck = cell.dye - 0.5;
        let pit = field::smooth((0.36 - pore) * 3.8);
        (
            pigment * (1. + contrast * fleck - 0.085 * seam - 0.14 * pit + 0.08 * (pore - 0.5)),
            relief * (fleck - 0.52 * seam - 0.70 * pit),
            0.78 + 0.08 * cell.dye + 0.065 * pit + 0.045 * seam,
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
    if scene.generator_version < TACKBOARD_VERSION {
        return;
    }
    for (facing, i) in [
        (Facing::NaturalCork, NATURAL),
        (Facing::ToastedCork, TOASTED),
    ] {
        let key = (Surface::FabricAlt, i);
        let used = match selection {
            Some(s) => s.finishes.contains(&key),
            None => scene.objects.iter().any(|o| {
                o.kind == ObjectKind::NoticeBoard
                    && Facing::for_board(o, scene.generator_version) == facing
            }),
        };
        if used {
            variants.insert(key, materials.add(stock(scene.material_seed(), i, images)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn cork_installs_once_per_used_stock_and_never_for_archived_textile() {
        let mut scene =
            IndoorManifest::generate_with_humans(41, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let key = (Surface::FabricAlt, NATURAL);
        let mut selection = MaterialSelection {
            surfaces: [key.0].into(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let mut images = Assets::<Image>::default();
        let mut mats = Assets::<StandardMaterial>::default();
        let mut vars = BTreeMap::new();
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert!(vars.is_empty() && images.is_empty());
        selection.finishes.insert(key);
        scene.generator_version = TACKBOARD_VERSION - 1;
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert!(vars.is_empty() && images.is_empty());
        scene.generator_version = TACKBOARD_VERSION;
        assert!(registered_finish(key.0, key.1) && selection.omits_parent_maps(&scene, key.0));
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert_eq!((vars.len(), mats.len(), images.len()), (1, 1, 3));
        assert!(vars.contains_key(&key));
    }

    #[test]
    fn cork_replays_metric_color_relief_and_roughness_through_every_mip() {
        fn bake(seed: u64, slot: usize) -> [Vec<u8>; 3] {
            let mut images = Assets::<Image>::default();
            let m = stock(seed, slot, &mut images);
            assert_eq!((m.metallic, m.perceptual_roughness), (0., 1.));
            let period = m.uv_transform.matrix2.x_axis.length().recip();
            assert!((0.104..0.191).contains(&period));
            let [color, normal, orm] = [
                (m.base_color_texture, true),
                (m.normal_map_texture, false),
                (m.metallic_roughness_texture, false),
            ]
            .map(|(h, srgb)| {
                let image = images.remove(h.unwrap().id()).unwrap();
                assert_eq!(image.texture_descriptor.format.is_srgb(), srgb);
                assert_eq!(image.texture_descriptor.mip_level_count, 8);
                image.data.unwrap()
            });
            assert!(orm
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 255 && (198..=255).contains(&p[1]) && p[2] == 0 && p[3] == 255));
            let base = &normal.as_chunks::<4>().0[..128 * 128];
            assert!(
                base.iter()
                    .filter(|p| p[0].abs_diff(128) > 3 || p[1].abs_diff(128) > 3)
                    .count()
                    > 1024
            );
            assert!(normal.as_chunks::<4>().0.iter().all(|p| {
                let n = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - 1.;
                (n.length() - 1.).abs() < 0.015 && n.z > 0.70
            }));
            [color, normal, orm]
        }
        let a = bake(2, NATURAL);
        assert!(a == bake(2, NATURAL));
        for b in [bake(3, NATURAL), bake(2, TOASTED)] {
            assert!(a.iter().zip(b).all(|(x, y)| *x != y));
        }
    }
}
