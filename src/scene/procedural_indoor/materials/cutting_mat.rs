//! Replaceable self-healing cutting sheet. Scale is metric in the bench chart.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::public::workbench::{Bench, WorkingSurface, VERSION},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const SLOT: usize = 2150;
pub(crate) const LABEL: &str = "workbench#finish2150#uv_local";
pub(super) fn supports(s: Surface, slot: usize) -> bool {
    s == Surface::Rubber && slot == SLOT
}

fn material(seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"CutSheet"));
    let colors = [
        [0.16, 0.33, 0.29],
        [0.17, 0.27, 0.34],
        [0.31, 0.34, 0.30],
        [0.31, 0.24, 0.31],
        [0.20, 0.35, 0.38],
        [0.19, 0.20, 0.21],
    ];
    let dye =
        Vec3::from_array(colors[r.random_range(0..colors.len())]) * r.random_range(0.87..1.14);
    let grid = if r.random_bool(0.69) { 0.050 } else { 0.0635 }; // 10 mm or half-inch subdivisions
    let print = r.random_range(0.115..0.27);
    let mut budget = 8192;
    let grain = field::PreparedPeriodic::new([21, 23], r.random(), &mut budget);
    let stops: [(Vec2, Vec2); 3] = std::array::from_fn(|_| {
        (
            Vec2::new(r.random(), r.random()),
            Vec2::new(r.random_range(0.09..0.42), r.random_range(-0.39..0.39)),
        )
    });
    sheet::sheet_material::<64>(images, Vec2::splat(grid), 0., 1., |p| {
        let edge = |x: f32| (x - x.round()).abs();
        let at = |q: Vec2, w: f32| (1. - edge(q.x).min(edge(q.y)) / w).clamp(0., 1.);
        let line = at(p, 0.021).max(0.44 * at(p * 5., 0.063));
        let mut cut = 0_f32;
        for (a, d) in stops {
            let q = (p - a + Vec2::splat(0.5)).rem_euclid(Vec2::ONE) - Vec2::splat(0.5);
            let t = (q.dot(d) / d.length_squared()).clamp(0., 1.);
            cut = cut.max((1. - (q - t * d).length() / 0.012).clamp(0., 1.));
        }
        let noise = grain.sample(p.x, p.y);
        (
            dye * (0.98 + noise * 0.038 - cut * 0.055) + Vec3::splat(print * line),
            0.000028 * noise - 0.000078 * cut,
            0.79 + 0.063 * noise - 0.11 * cut - 0.045 * line,
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
    if scene.generator_version < VERSION {
        return;
    }
    let key = (Surface::Rubber, SLOT);
    let used = selection.map_or_else(
        || {
            scene.objects.iter().any(|o| {
                o.kind == ObjectKind::Workbench
                    && Bench::sample(o).surface == WorkingSurface::CuttingMat
            })
        },
        |s| s.finishes.contains(&key),
    );
    if used {
        variants.insert(key, materials.add(material(scene.material_seed(), images)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cutting_mat_produces_metric_mipmapped_dielectric_with_relief_and_ink() {
        let mut images = Assets::<Image>::default();
        let m = material(133, &mut images);
        let [color, normal, orm] = [
            m.base_color_texture,
            m.normal_map_texture,
            m.metallic_roughness_texture,
        ]
        .map(|h| images.get(h.as_ref().unwrap()).unwrap());
        assert!(color.texture_descriptor.format.is_srgb());
        assert!(
            !normal.texture_descriptor.format.is_srgb() && !orm.texture_descriptor.format.is_srgb()
        );
        assert_eq!(m.metallic, 0.);
        for map in [color, normal, orm] {
            assert_eq!(map.texture_descriptor.mip_level_count, 7);
        }
        let level = 64 * 64 * 4;
        let colors: std::collections::BTreeSet<_> = color.data.as_ref().unwrap()[..level]
            .as_chunks::<4>()
            .0
            .iter()
            .collect();
        assert!(colors.len() > 20);
        assert!(normal.data.as_ref().unwrap()[..level]
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[0].abs_diff(128) > 2 || p[1].abs_diff(128) > 2));
        let data = orm.data.as_ref().unwrap()[..level].as_chunks::<4>().0;
        assert!(data.iter().all(|p| p[2] == 0 && p[1] > 136));
        assert!(
            data.iter().map(|p| p[1]).max().unwrap() - data.iter().map(|p| p[1]).min().unwrap()
                > 15
        );
    }

    #[test]
    fn cutting_mat_is_allocated_once_only_for_visible_stock() {
        use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Workshop, 0.6, 0, 0.).unwrap();
        let run = |scene: &IndoorManifest, selection: Option<&MaterialSelection>| {
            let (mut images, mut materials) = (Assets::default(), Assets::default());
            let mut variants = BTreeMap::new();
            install(scene, &mut images, &mut materials, selection, &mut variants);
            (images.len(), materials.len(), variants.len())
        };
        let fixture = |kind| {
            (0..64)
                .map(|s| audit::fixture(ObjectKind::Workbench, s))
                .find(|o| Bench::sample(o).surface == kind)
                .unwrap()
        };
        scene.objects = vec![fixture(WorkingSurface::DogHoles)];
        assert_eq!(run(&scene, None), (0, 0, 0));
        scene.objects.extend([
            fixture(WorkingSurface::CuttingMat),
            fixture(WorkingSurface::CuttingMat),
        ]);
        assert_eq!(run(&scene, None), (3, 1, 1));
        let mut selected = MaterialSelection {
            surfaces: Default::default(),
            direct_surfaces: Default::default(),
            finishes: Default::default(),
        };
        assert_eq!(run(&scene, Some(&selected)), (0, 0, 0));
        selected.select_finish(Surface::Rubber, SLOT);
        assert_eq!(run(&scene, Some(&selected)), (3, 1, 1));
    }
}
