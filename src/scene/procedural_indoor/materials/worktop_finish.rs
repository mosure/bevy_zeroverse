//! Finished sheet and slab stock for installed worktops and furniture.
use super::super::{
    layout::{stream, IndoorObject, ObjectKind},
    objects::{kitchen, tables, ObjectFamily},
    preparation::AssetStore,
};
use super::field::PreparedPeriodic;
use super::sheet::sheet_material;
use super::*;
use rand::Rng;
use std::collections::BTreeMap;

mod basin;
mod laminate;
mod tapware;
const FIRST: usize = 109;
const BASIN: usize = 111;
const TAP: usize = 114;
pub(crate) const TAP_LABEL: &str = "faucet#finish114";
pub(crate) const BASIN_LABEL: &str = "sink#finish111";

/// Installed bare metalwork uses the same metre-scale brushed roughness map
/// as stainless bowls and satin-plated fittings. Both stocks are conductive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BrushedHardware {
    Stainless,
    SatinPlated,
}
impl BrushedHardware {
    pub(crate) fn for_landing(o: &IndoorObject) -> Self {
        if stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"SillStok"),
        )
        .random_bool(0.68)
        {
            Self::Stainless
        } else {
            Self::SatinPlated
        }
    }
    fn slot(self) -> usize {
        match self {
            Self::Stainless => BASIN,
            Self::SatinPlated => TAP,
        }
    }
    pub(crate) fn label(self, class: &str) -> String {
        format!("{class}#finish{}#uv_local", self.slot())
    }
}

pub(crate) fn is_sheet_slot(slot: usize) -> bool {
    (FIRST..FIRST + 2).contains(&slot)
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    slot == TAP && matches!(surface, Surface::Chrome | Surface::Metal | Surface::Ink)
        || sheet_surface(surface) && is_sheet_slot(slot)
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

// Temporary per atlas, shared by all texels and released before the next stock.
const LATTICE_BUDGET: usize = 32 * 1024;

fn sheet_fields<const N: usize>(
    seed: u64,
    grids: [([u32; 2], u64); N],
    mut budget: usize,
) -> [PreparedPeriodic; N] {
    grids.map(|(count, salt)| PreparedPeriodic::new(count, seed ^ salt, &mut budget))
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

    fn fields(&self, budget: usize) -> [PreparedPeriodic; 5] {
        sheet_fields(
            self.seed,
            [
                ([3, 3], 1),
                ([4, 2], 2),
                (self.grain, 3),
                ([61, 57], 4),
                (self.ribbon, 5),
            ],
            budget,
        )
    }

    #[cfg(test)]
    fn texel(&self, p: Vec2) -> (Vec3, f32, f32) {
        self.sample_fields(p, &self.fields(0))
    }

    fn sample_fields(&self, p: Vec2, fields: &[PreparedPeriodic; 5]) -> (Vec3, f32, f32) {
        let [wx, wy, mineral_field, grit_field, band_field] = fields;
        let warp = Vec2::new(wx.sample(p.x, p.y), wy.sample(p.x, p.y)) - 0.5;
        let q = p + warp * 0.14;
        let mineral = mineral_field.sample(q.x, q.y);
        let grit = grit_field.sample(p.x, p.y);
        let chip = ((mineral * 0.79 + grit * 0.21 - 0.55) * 5.5).clamp(0., 1.) * self.chips;
        let q = p + warp * 0.24;
        let band = band_field.sample(q.x, q.y);
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
        self.material_with_budget(images, LATTICE_BUDGET)
    }

    fn material_with_budget(
        &self,
        images: &mut impl AssetStore<Image>,
        budget: usize,
    ) -> StandardMaterial {
        let fields = self.fields(budget);
        sheet_material::<128>(
            images,
            self.period,
            self.glaze,
            self.roughness * 0.88,
            |p| self.sample_fields(p, &fields),
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
                ObjectKind::Elevator
                    if scene.generator_version
                        >= super::super::objects::public::elevator::LANDING_VERSION =>
                {
                    keys.insert((Surface::Metal, BrushedHardware::for_landing(o).slot()));
                }
                ObjectKind::KitchenCounter
                | ObjectKind::KitchenSink
                | ObjectKind::KitchenCupboard => {
                    let program = kitchen::parameters(o);
                    if matches!(program.top, Surface::Concrete | Surface::Ceramic) {
                        keys.insert((program.top, FIRST + stock(o)));
                    }
                    if o.kind == ObjectKind::KitchenSink {
                        let (bowl, tap) = kitchen::sink::program::finishes(o);
                        keys.insert((bowl, BASIN));
                        if scene.generator_version >= kitchen::sink::FITTED_PLUMBING_VERSION {
                            keys.insert((tap, TAP));
                        }
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
    let mut brushed = None;
    for (surface, slot) in used {
        let material = if surface == Surface::Metal && matches!(slot, BASIN | TAP) {
            // The brushing has a common metric scale. Reuse its mip chain when
            // the bowl and the hardware in this scene both need one.
            let stock = brushed
                .get_or_insert_with(|| basin::material(surface, scene.material_seed(), images))
                .clone();
            if slot == TAP {
                tapware::material(scene.material_seed(), surface, stock)
            } else {
                stock
            }
        } else if slot == TAP {
            tapware::material(scene.material_seed(), surface, StandardMaterial::default())
        } else if slot == BASIN {
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
    fn elevator_sill_stock_is_conductive_and_shares_one_filtered_roughness_map() {
        use crate::scene::procedural_indoor::{
            layout::IndoorLayout,
            objects::{audit, public::elevator},
        };
        let mut scene =
            IndoorManifest::generate_with_humans(95500, IndoorLayout::ElevatorLanding, 0.65, 0, 0.)
                .unwrap();
        scene.objects = [BrushedHardware::Stainless, BrushedHardware::SatinPlated]
            .map(|stock| {
                (0..128)
                    .map(|seed| {
                        let mut car = audit::fixture(ObjectKind::Elevator, seed);
                        car.finish_seed = Some(seed);
                        car
                    })
                    .find(|o| BrushedHardware::for_landing(o) == stock)
                    .unwrap()
            })
            .into();
        let selected = MaterialSelection {
            finishes: [(Surface::Metal, BASIN), (Surface::Metal, TAP)].into(),
            surfaces: Default::default(),
            direct_surfaces: Default::default(),
        };
        for selection in [None, Some(&selected)] {
            let mut images = Assets::default();
            let mut materials = Assets::default();
            let mut variants = BTreeMap::new();
            install(
                &scene,
                &mut images,
                &mut materials,
                selection,
                &mut variants,
            );
            let stock_material = |slot| materials.get(&variants[&(Surface::Metal, slot)]).unwrap();
            let stainless = stock_material(BASIN);
            let plated = stock_material(TAP);
            assert_eq!(stainless.metallic, 1.);
            assert_eq!(plated.metallic, 1.);
            assert_eq!(
                stainless.metallic_roughness_texture,
                plated.metallic_roughness_texture
            );
            let map = images
                .get(stainless.metallic_roughness_texture.as_ref().unwrap())
                .unwrap();
            assert!(map.texture_descriptor.mip_level_count > 1);
        }
        scene.generator_version = elevator::LANDING_VERSION - 1;
        let mut variants = BTreeMap::new();
        install(
            &scene,
            &mut Assets::default(),
            &mut Assets::default(),
            None,
            &mut variants,
        );
        assert!(variants.is_empty());
    }

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
    fn cached_sheet_fields_preserve_every_byte_of_every_pbr_mip() {
        let mut elapsed = [std::time::Duration::ZERO; 2];
        for seed in [0, 359, u64::MAX] {
            for surface in [Surface::Ceramic, Surface::Concrete, Surface::Plastic] {
                for slot in FIRST..FIRST + 2 {
                    let mut bake = |budget| {
                        let start = std::time::Instant::now();
                        let mut images = Assets::<Image>::default();
                        let mat = if surface == Surface::Plastic {
                            laminate::Sheet::sample(seed, slot)
                                .material_with_budget(&mut images, budget)
                        } else {
                            Stock::sample(seed, surface, slot)
                                .material_with_budget(&mut images, budget)
                        };
                        let maps = [
                            mat.base_color_texture,
                            mat.normal_map_texture,
                            mat.metallic_roughness_texture,
                        ]
                        .map(|h| {
                            let image = images.remove(h.unwrap().id()).unwrap();
                            (image.texture_descriptor, image.data.unwrap())
                        });
                        elapsed[usize::from(budget > 0)] += start.elapsed();
                        (mat.uv_transform, maps)
                    };
                    // Alternate the order of the two bakes for the timing diagnostic.
                    let (scalar_result, cached_result) = if slot == FIRST {
                        (bake(0), bake(LATTICE_BUDGET))
                    } else {
                        let cached = bake(LATTICE_BUDGET);
                        (bake(0), cached)
                    };
                    let (uv, cached) = cached_result;
                    let (expected_uv, scalar) = scalar_result;
                    assert_eq!(uv, expected_uv);
                    for ((d, a), (reference, b)) in cached.iter().zip(&scalar) {
                        assert_eq!(d, reference);
                        assert!(a == b, "{surface:?} {seed} {slot}");
                    }
                }
            }
        }
        eprintln!(
            "18 worktop bakes, scalar {:?}, cached {:?}",
            elapsed[0], elapsed[1]
        );
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
