//! Metric plain-weave flax and gessoed canvas. Only occupied finishes are baked.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::public::easel::{Canvas, Easel, VERSION},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 2100;
pub(crate) const RAW: &str = "picture#finish2100#uv_local";
pub(crate) const PRIMED: &str = "picture#finish2101#uv_local";

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Paper && (FIRST..=FIRST + 1).contains(&slot)
}

fn material(seed: u64, primed: bool, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"FlaxLoom"));
    let period = Vec2::new(
        r.random_range(0.0035..0.0065),
        r.random_range(0.0038..0.0068),
    );
    let flax = Vec3::new(0.72, 0.62, 0.48) * r.random_range(0.81..1.13);
    let gesso = Vec3::new(0.94, 0.92, 0.88) * r.random_range(0.92..1.02);
    let mut budget = 1024;
    let slubs = field::PreparedPeriodic::new([4, 3], r.random(), &mut budget);
    let phase = Vec2::new(r.random(), r.random());
    sheet::sheet_material::<32>(images, period, 0., 1., |p| {
        let slub = slubs.sample(p.x, p.y);
        let threads = p * Vec2::new(6., 8.) + phase + Vec2::new(0.10, -0.08) * slub;
        let warp = (std::f32::consts::PI * threads.x).sin().abs();
        let weft = (std::f32::consts::PI * threads.y).sin().abs();
        let crossing = (threads.x.floor() as i32 + threads.y.floor() as i32) & 1;
        let (over, under) = if crossing == 0 {
            (warp, weft)
        } else {
            (weft, warp)
        };
        let crown = over * (0.60 + 0.40 * under) + (1. - over) * 0.57 * under;
        if primed {
            (
                gesso * (0.976 + 0.023 * crown),
                0.000034 * (crown + 0.13 * slub),
                0.87 - 0.057 * crown,
            )
        } else {
            (
                flax * (0.86 + 0.12 * crown + 0.042 * slub),
                0.000085 * (crown + 0.20 * slub),
                0.94 - 0.096 * crown,
            )
        }
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
    let mut used = [false; 2];
    if let Some(s) = selection {
        for (i, flag) in used.iter_mut().enumerate() {
            *flag = s.finishes.contains(&(Surface::Paper, FIRST + i));
        }
    } else {
        for object in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::ArtistEasel)
        {
            let canvas = Easel::sample(object).canvas;
            used[0] |= canvas != Canvas::Empty;
            used[1] |= canvas == Canvas::Primed;
        }
    }
    for (i, present) in used.into_iter().enumerate() {
        if present {
            let handle = materials.add(material(scene.material_seed(), i == 1, images));
            variants.insert((Surface::Paper, FIRST + i), handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canvas_stock_has_metric_relief_filtered_dielectric_maps_and_distinct_primer() {
        let mut images = Assets::<Image>::default();
        let mut swatches = Vec::new();
        for primed in [false, true] {
            let m = material(21, primed, &mut images);
            let [color, normal, orm] = [
                m.base_color_texture,
                m.normal_map_texture,
                m.metallic_roughness_texture,
            ]
            .map(|h| images.get(h.as_ref().unwrap()).unwrap());
            assert_eq!(m.metallic, 0.);
            assert_eq!(normal.texture_descriptor.mip_level_count, 6);
            assert!(color.texture_descriptor.format.is_srgb());
            assert!(
                !normal.texture_descriptor.format.is_srgb()
                    && !orm.texture_descriptor.format.is_srgb()
            );
            let texels = 32 * 32 * 4;
            let normal = &normal.data.as_ref().unwrap()[..texels];
            let orm = &orm.data.as_ref().unwrap()[..texels];
            assert!(normal
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0].abs_diff(128) > 5 || p[1].abs_diff(128) > 5));
            assert!(orm
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[2] == 0 && p[1] > 180));
            let mut roughness = orm
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[1])
                .collect::<Vec<_>>();
            roughness.sort_unstable();
            assert!(roughness.last().unwrap() - roughness[0] > 5);
            swatches.push(
                color.data.as_ref().unwrap()[..texels]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| u32::from(p[2]))
                    .sum::<u32>()
                    / 1024,
            );
        }
        assert!(swatches[1] > swatches[0] + 65);
    }
    #[test]
    fn canvas_stock_bakes_only_consumed_faces_once_per_room() {
        use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::Studio, 0.5, 0, 0.).unwrap();
        let fixture = |kind: Canvas| {
            (0..128)
                .map(|s| audit::fixture(ObjectKind::ArtistEasel, s))
                .find(|o| Easel::sample(o).canvas == kind)
                .unwrap()
        };
        let bake = |s: &IndoorManifest, selection: Option<&MaterialSelection>| {
            let mut images = Assets::default();
            let mut materials = Assets::default();
            let mut finishes = BTreeMap::new();
            install(s, &mut images, &mut materials, selection, &mut finishes);
            (images.len(), finishes.into_keys().collect::<Vec<_>>())
        };
        scene.objects = vec![fixture(Canvas::Empty)];
        assert_eq!(bake(&scene, None), (0, vec![]));
        scene
            .objects
            .extend([fixture(Canvas::Painted), fixture(Canvas::Linen)]);
        assert_eq!(bake(&scene, None), (3, vec![(Surface::Paper, FIRST)]));
        scene
            .objects
            .extend([fixture(Canvas::Primed), fixture(Canvas::Primed)]);
        assert_eq!(
            bake(&scene, None),
            (
                6,
                vec![(Surface::Paper, FIRST), (Surface::Paper, FIRST + 1)]
            )
        );
        let selected = MaterialSelection {
            surfaces: Default::default(),
            direct_surfaces: Default::default(),
            finishes: [(Surface::Paper, FIRST + 1)].into_iter().collect(),
        };
        assert_eq!(
            bake(&scene, Some(&selected)),
            (3, vec![(Surface::Paper, FIRST + 1)])
        );
    }
}
