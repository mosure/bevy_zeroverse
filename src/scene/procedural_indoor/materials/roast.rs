//! Cooked stock shares its pigment, dielectric roughness and metric relief fields.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::public::hot_food::{Contents, HotFoodWell},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Stock {
    Potato,
    Carrot,
    Gratin,
}
const STOCK: [Stock; 3] = [Stock::Potato, Stock::Carrot, Stock::Gratin];
impl Stock {
    fn slot(self) -> usize {
        999 + self as usize
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Potato => "food#finish999#uv_local",
            Self::Carrot => "food#finish1000#uv_local",
            Self::Gratin => "food#finish1001#uv_local",
        }
    }
    fn material(self, seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
        let mut r = stream(seed, self.slot() as u64);
        let (skin, toast, rough, relief) = match self {
            Self::Carrot => ([216., 110., 40.], [142., 54., 22.], 0.36, 0.00028),
            Self::Potato => ([226., 178., 96.], [125., 68., 34.], 0.49, 0.00053),
            Self::Gratin => ([232., 198., 117.], [137., 69., 28.], 0.43, 0.0012),
        };
        let base = Vec3::from_array(skin) / 255.;
        let crust = Vec3::from_array(toast) / 255.;
        let doneness = r.random_range(0.36..0.55);
        let tint = r.random_range(0.88..1.06);
        let key = r.random::<u64>();
        sheet::sheet_material::<32>(images, Vec2::splat(0.076), 0., rough, |uv| {
            let field = periodic_noise(uv.x, uv.y, 5, 4, key);
            let pores = periodic_noise(uv.x, uv.y, 13, 15, key.rotate_left(29));
            let brown = ((field + 0.22 * pores - doneness) * 2.8).clamp(0., 1.);
            (
                base.lerp(crust, brown) * tint * (0.95 + 0.08 * pores),
                relief * (field * field + 0.45 * pores),
                rough + brown * 0.23 + pores * 0.05,
            )
        })
    }
}
pub(super) fn supports(s: Surface, slot: usize) -> bool {
    s == Surface::Terracotta && (999..=1001).contains(&slot)
}
pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let mut used = [false; STOCK.len()];
    if let Some(s) = selection {
        used = STOCK.map(|v| s.finishes.contains(&(Surface::Terracotta, v.slot())));
    } else {
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::HotFoodWell)
        {
            for p in HotFoodWell::for_object(o).pans {
                let stock = match p.contents {
                    Contents::RoastPotatoes => Stock::Potato,
                    Contents::GlazedCarrots => Stock::Carrot,
                    Contents::PotatoGratin => Stock::Gratin,
                    _ => continue,
                };
                used[stock as usize] = true;
            }
        }
    }
    for (stock, used) in STOCK.into_iter().zip(used) {
        if used {
            variants.insert(
                (Surface::Terracotta, stock.slot()),
                materials.add(stock.material(scene.material_seed(), images)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hot_food_well_finishes_have_dielectric_roughness_relief_and_complete_mips() {
        let mut images = Assets::<Image>::default();
        for stock in STOCK {
            let sample = stock.material(119, &mut images);
            let again = stock.material(119, &mut images);
            let different = stock.material(121, &mut images);
            let color = images.get(sample.base_color_texture.unwrap().id()).unwrap();
            let orm = images
                .get(sample.metallic_roughness_texture.unwrap().id())
                .unwrap();
            let normals = images.get(sample.normal_map_texture.unwrap().id()).unwrap();
            assert!(color.texture_descriptor.format.is_srgb());
            assert!(
                !orm.texture_descriptor.format.is_srgb()
                    && !normals.texture_descriptor.format.is_srgb()
            );
            assert_eq!(color.texture_descriptor.mip_level_count, 6);
            let pixels = orm.data.as_ref().unwrap();
            assert_eq!(pixels.len(), (1024 + 256 + 64 + 16 + 4 + 1) * 4);
            assert!(pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[2] == 0 && p[3] == 255));
            let (lo, hi) = pixels[..4096]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[1])
                .fold((255, 0), |(lo, hi), v| (lo.min(v), hi.max(v)));
            assert!(lo >= 80 && hi <= 210 && hi - lo > 24);
            assert!(normals.data.as_ref().unwrap()[..4096]
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0].abs_diff(128) > 2 && p[2] > 200));
            assert_eq!(
                images
                    .get(again.base_color_texture.unwrap().id())
                    .unwrap()
                    .data,
                color.data
            );
            assert_ne!(
                images
                    .get(different.base_color_texture.unwrap().id())
                    .unwrap()
                    .data,
                color.data
            );
        }
    }
}
