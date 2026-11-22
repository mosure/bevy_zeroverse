//! Finished sheet and slab stock for installed worktops and furniture.
use super::super::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::{kitchen, tables, ObjectFamily},
    preparation::AssetStore,
};
use super::sheet::sheet_material;
use super::*;
use rand::Rng;
use std::collections::BTreeMap;

mod basin;
mod laminate;
const FIRST: usize = 109;
const BASIN: usize = 111;
pub(crate) const BASIN_LABEL: &str = "sink#finish111";

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    sheet_surface(surface) && (FIRST..FIRST + 2).contains(&slot)
        || slot == BASIN
            && matches!(
                surface,
                Surface::Ceramic | Surface::Concrete | Surface::Metal
            )
}

fn sheet_surface(surface: Surface) -> bool {
    matches!(
        surface,
        Surface::Concrete | Surface::Ceramic | Surface::Plastic
    )
}

pub(crate) fn table_label(o: &IndoorObject, surface: Surface) -> std::borrow::Cow<'static, str> {
    let class = o.kind.class_name();
    if sheet_surface(surface) {
        format!("{class}#finish{}", FIRST + stock(o)).into()
    } else {
        class.into()
    }
}

fn stock(o: &IndoorObject) -> usize {
    stream(o.finish_seed.unwrap_or(o.seed), 0x534c_4142_4c4f_5453).random_range(0..2)
}
pub(crate) fn label(o: &IndoorObject, surface: Surface) -> &'static str {
    if matches!(surface, Surface::Concrete | Surface::Ceramic) {
        ["counter#finish109", "counter#finish110"][stock(o)]
    } else {
        "counter"
    }
}

struct Stock {
    seed: u64,
    matrix: Vec3,
    inclusion: Vec3,
    grain: [u32; 2],
    ribbon: [u32; 2],
    chips: f32,
    veins: f32,
    period: Vec2,
    roughness: f32,
    relief: f32,
    glaze: f32,
}

impl Stock {
    fn sample(seed: u64, surface: Surface, slot: usize) -> Self {
        let mut rng = stream(
            seed,
            0x574f_524b_544f_5053 ^ ((surface as u64) << 16) ^ slot as u64,
        );
        let ceramic = surface == Surface::Ceramic;
        let value = if rng.random_bool(0.26) {
            rng.random_range(0.20..0.51)
        } else {
            rng.random_range(0.52..0.89)
        };
        let warmth = rng.random_range(-0.045..0.085);
        let matrix = Vec3::new(value + warmth, value, value - 0.78 * warmth);
        let contrast = if rng.random_bool(0.64) {
            rng.random_range(-0.37..-0.10)
        } else {
            rng.random_range(0.12..0.37)
        };
        let polish = rng.random_range(0.0f32..1.);
        let chips = rng.random_range(0.10f32..1.) * if ceramic { 0.68 } else { 1. };
        Self {
            seed: rng.random(),
            matrix,
            inclusion: (matrix + Vec3::splat(contrast) + Vec3::new(0.023, 0., -0.02))
                .clamp(Vec3::splat(0.06), Vec3::splat(0.94)),
            grain: [rng.random_range(23..54), rng.random_range(23..55)],
            ribbon: [rng.random_range(2..5), rng.random_range(4..10)],
            chips,
            veins: rng.random_range(0.04f32..0.95).powi(2) * if ceramic { 1. } else { 0.76 },
            period: Vec2::new(rng.random_range(0.45..1.2), rng.random_range(0.42..0.90)),
            roughness: 0.53 - 0.31 * polish,
            relief: (0.00035 - 0.00025 * polish) * if ceramic { 0.56 } else { 1. },
            glaze: if ceramic { 0.08 + 0.24 * polish } else { 0. },
        }
    }

    fn texel(&self, p: Vec2) -> (Vec3, f32, f32) {
        let noise = |p: Vec2, cells: [u32; 2], salt| {
            periodic_noise(p.x, p.y, cells[0], cells[1], self.seed ^ salt)
        };
        let warp = Vec2::new(noise(p, [3, 3], 1), noise(p, [4, 2], 2)) - 0.5;
        let mineral = noise(p + warp * 0.14, self.grain, 3);
        let grit = noise(p, [61, 57], 4);
        let chip = ((mineral * 0.79 + grit * 0.21 - 0.55) * 5.5).clamp(0., 1.) * self.chips;
        let band = noise(p + warp * 0.24, self.ribbon, 5);
        let vein = (1. - ((band - 0.53).abs() * 16.).min(1.)).powi(2) * self.veins;
        let pigment = self
            .matrix
            .lerp(self.inclusion, (chip + vein * (1. - chip)).clamp(0., 1.))
            + Vec3::splat((grit - 0.5) * 0.027 + warp.y * 0.020);
        let relief = self.relief * (chip * 0.48 + grit * 0.38 + mineral * 0.14);
        (
            pigment,
            relief,
            self.roughness + 0.055 * (grit - 0.5) + 0.10 * chip,
        )
    }

    fn material(&self, images: &mut impl AssetStore<Image>) -> StandardMaterial {
        sheet_material::<128>(
            images,
            self.period,
            self.glaze,
            self.roughness * 0.88,
            |p| self.texel(p),
        )
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let used = if let Some(s) = selection {
        s.finishes
            .iter()
            .copied()
            .filter(|&(surface, slot)| supports(surface, slot))
            .collect()
    } else {
        let mut keys = std::collections::BTreeSet::new();
        for o in &scene.objects {
            match o.kind {
                ObjectKind::KitchenCounter
                | ObjectKind::KitchenSink
                | ObjectKind::KitchenCupboard => {
                    let program = kitchen::parameters(o);
                    if matches!(program.top, Surface::Concrete | Surface::Ceramic) {
                        keys.insert((program.top, FIRST + stock(o)));
                    }
                    if o.kind == ObjectKind::KitchenSink {
                        keys.insert((
                            kitchen::sink::program::parameters(o, &program).basin_finish,
                            BASIN,
                        ));
                    }
                }
                kind if kind.family() == ObjectFamily::Tables => {
                    let top = tables::parameters(o).top_surface;
                    if sheet_surface(top) {
                        keys.insert((top, FIRST + stock(o)));
                    }
                }
                _ => {}
            }
        }
        keys
    };
    for (surface, slot) in used {
        let material = if slot == BASIN {
            basin::material(surface, scene.material_seed(), images)
        } else if surface == Surface::Plastic {
            laminate::Sheet::sample(scene.material_seed(), slot).material(images)
        } else {
            Stock::sample(scene.material_seed(), surface, slot).material(images)
        };
        variants.insert((surface, slot), materials.add(material));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scene_and_selected_install_cover_every_manufactured_table_family() {
        use crate::scene::procedural_indoor::objects;
        let mut scene =
            IndoorManifest::generate_with_humans(51, Default::default(), 0.65, 0, 0.).unwrap();
        for kind in ObjectKind::ALL
            .into_iter()
            .filter(|k| k.family() == ObjectFamily::Tables)
        {
            let o = (0..128)
                .map(|seed| objects::audit::fixture(kind, seed))
                .find(|o| tables::parameters(o).top_surface == Surface::Plastic)
                .unwrap();
            let key = (Surface::Plastic, FIRST + stock(&o));
            let label = table_label(&o, key.0);
            let parts = objects::build_object(&o).parts;
            assert!(parts.contains_key(&(key.0, label.into_owned())), "{kind:?}");
            scene.objects = vec![o];
            let used = MaterialSelection {
                surfaces: [key.0].into(),
                direct_surfaces: Default::default(),
                finishes: [key].into(),
            };
            let mut images = Assets::default();
            let mut materials = Assets::default();
            let mut response = None;
            for selection in [None, Some(&used)] {
                let mut variants = BTreeMap::new();
                install(
                    &scene,
                    &mut images,
                    &mut materials,
                    selection,
                    &mut variants,
                );
                assert_eq!(variants.len(), 1);
                let handle = materials
                    .get(&variants[&key])
                    .unwrap()
                    .metallic_roughness_texture
                    .as_ref()
                    .unwrap();
                let map = images.get(handle).unwrap().data.as_ref().unwrap();
                if let Some(expected) = &response {
                    assert_eq!(expected, map);
                } else {
                    response = Some(map.clone());
                }
            }
        }
    }

    #[test]
    fn mineral_stock_replays_with_filtered_metric_normal_and_roughness_maps() {
        for surface in [Surface::Ceramic, Surface::Concrete] {
            let a = Stock::sample(72, surface, FIRST);
            let b = Stock::sample(72, surface, FIRST);
            let c = Stock::sample(73, surface, FIRST);
            let p = Vec2::new(0.23, 0.42);
            assert_eq!(a.texel(p), b.texel(p));
            assert_ne!(a.texel(p), c.texel(p));
            assert!(a.texel(p).0.distance(a.texel(p + Vec2::new(1., -1.)).0) < 0.0001);
            assert!(super::super::registered_finish(surface, FIRST));
            let mut images = Assets::<Image>::default();
            let m = a.material(&mut images);
            assert_eq!(images.len(), 3);
            for (handle, srgb) in [
                (&m.base_color_texture, true),
                (&m.normal_map_texture, false),
                (&m.metallic_roughness_texture, false),
            ] {
                let image = images.get(handle.as_ref().unwrap()).unwrap();
                assert_eq!(image.texture_descriptor.mip_level_count, 8);
                assert_eq!(image.texture_descriptor.format.is_srgb(), srgb);
            }
            let image = images
                .get(m.metallic_roughness_texture.as_ref().unwrap())
                .unwrap();
            let data = image.data.as_ref().unwrap();
            let rough: std::collections::BTreeSet<_> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    assert_eq!((p[0], p[2], p[3]), (255, 0, 255));
                    p[1]
                })
                .collect();
            assert!(rough.len() > 8);
            let normals = images
                .get(m.normal_map_texture.as_ref().unwrap())
                .unwrap()
                .data
                .as_ref()
                .unwrap();
            assert!(normals
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[2] >= 250 && p[3] == 255));
            assert!(normals
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0].abs_diff(128) > 1 || p[1].abs_diff(128) > 1));
        }
    }
}
