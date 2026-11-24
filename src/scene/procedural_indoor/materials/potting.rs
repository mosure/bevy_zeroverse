//! The fine fraction remains visible between larger geometric grains. Seed one
//! repeatable, metric PBR stock per installed medium; both grain tints reuse its
//! filtered normal and roughness maps, including on the cooperative renderer.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    plants::soil::{self, Layer, Medium},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Soil
        && (soil::FIRST_SLOT..soil::FIRST_SLOT + 3 * Medium::GRADED.len()).contains(&slot)
}

struct Stock {
    medium: Medium,
    period: Vec2,
    cells: field::PreparedCellular,
    pores: field::PreparedPeriodic,
    matrix: Vec3,
    grain: Vec3,
    roughness: f32,
    relief: f32,
    grain_radius: f32,
}

impl Stock {
    fn sample(seed: u64, medium: Medium) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"MediaPBR") ^ medium as u64);
        let (matrix, grain, spacing, aspect, rough, relief, radius) = match medium {
            Medium::PerliteMix => (
                Vec3::new(0.225, 0.154, 0.099),
                Vec3::new(0.86, 0.84, 0.75),
                0.0035..0.007,
                0.81..1.13,
                0.88,
                0.0008,
                0.25..0.40,
            ),
            Medium::BarkChips => (
                Vec3::new(0.16, 0.095, 0.055),
                Vec3::new(0.48, 0.31, 0.18),
                0.0035..0.007,
                1.45..2.65,
                0.87,
                0.0013,
                0.46..0.64,
            ),
            Medium::ExpandedClay => (
                Vec3::new(0.26, 0.15, 0.096),
                Vec3::new(0.62, 0.38, 0.235),
                0.006..0.012,
                0.85..1.13,
                0.84,
                0.0023,
                0.37..0.51,
            ),
            Medium::MineralGravel => (
                Vec3::new(0.26, 0.24, 0.21),
                Vec3::new(0.62, 0.61, 0.54),
                0.004..0.009,
                0.78..1.45,
                0.73,
                0.0015,
                0.40..0.55,
            ),
            Medium::Loam => unreachable!("fine soil uses the scene soil material"),
        };
        let length = rng.random_range(0.085..0.135);
        let period = Vec2::new(length * rng.random_range(0.89..1.16), length);
        let spacing = rng.random_range(spacing);
        let frequency = [
            field::frequency(period.x, spacing * rng.random_range(aspect)),
            field::frequency(period.y, spacing),
        ];
        let mut budget = 32 * 1024;
        Self {
            medium,
            period,
            cells: field::PreparedCellular::new(frequency, rng.random()),
            pores: field::PreparedPeriodic::new([37, 43], rng.random(), &mut budget),
            matrix: matrix * rng.random_range(0.77..1.18),
            grain: grain
                * Vec3::new(
                    rng.random_range(0.91..1.05),
                    1.,
                    rng.random_range(0.88..1.07),
                )
                * rng.random_range(0.86..1.09),
            roughness: rough,
            relief: relief * rng.random_range(0.67..1.28),
            grain_radius: rng.random_range(radius),
        }
    }
    fn texel(&self, uv: Vec2) -> (Vec3, f32, f32) {
        let cell = self.cells.sample_edges(uv.to_array());
        let pore = self.pores.sample(uv.x, uv.y);
        let radius = self.grain_radius * (0.64 + 0.68 * cell.shape);
        // Give each piece its own orientation and perimeter. Filling the
        // Voronoi cells all the way to their edges makes clay resemble paving.
        let [x, y] = cell.offset;
        let tilt = (cell.dye - 0.5) * 1.4;
        let u = (x + tilt * y).abs();
        let v = (y - tilt * x).abs();
        let distance = match self.medium {
            Medium::ExpandedClay => cell.radius * (0.97 + 0.08 * cell.shape),
            Medium::BarkChips => (0.66 * u + 0.52 * v).max(1.14 * v),
            _ => 0.78 * u.max(v) + 0.21 * (u + v),
        };
        let radius = radius * (0.93 + 0.14 * pore);
        let crown = (1. - (distance / radius).powi(2)).clamp(0., 1.);
        let outline = field::smooth(0.45 + (radius - distance) * 13.);
        let separation = field::smooth(cell.edge * 8.);
        let mask = outline
            * separation
            * match self.medium {
                Medium::PerliteMix => field::smooth((cell.dye - 0.28) * 7.),
                Medium::BarkChips => field::smooth((cell.dye - 0.06) * 4.5),
                _ => 1.,
            };
        let pigment = self.grain * (0.60 + 0.34 * cell.dye + 0.15 * pore + 0.07 * crown);
        let color = (self.matrix * (0.66 + 0.44 * pore)).lerp(pigment, mask);
        let raised = mask * (0.30 + 0.70 * crown) * (0.62 + 0.38 * cell.dye);
        (
            color,
            self.relief * (raised + 0.16 * (pore - 0.5)),
            (self.roughness + 0.09 * (1. - mask) + 0.10 * (pore - 0.5)).clamp(0.55, 1.),
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
    if scene.generator_version < soil::INTRODUCED {
        return;
    }
    let mut present = [false; 4];
    if selection.is_none() && !scene.objects.is_empty() {
        let mut include = |o: &super::super::layout::IndoorObject| {
            if let Some(slot) = Medium::sample(o).slot(Layer::Bed) {
                present[(slot - soil::FIRST_SLOT) / 3] = true;
            }
        };
        for o in scene.objects.iter().filter(|o| o.kind == ObjectKind::Plant) {
            include(o);
        }
        for o in super::super::architecture::outdoors::pergola_plants(scene) {
            include(&o);
        }
    }
    for (i, medium) in Medium::GRADED.into_iter().enumerate() {
        let slots =
            [Layer::Bed, Layer::Grain, Layer::Accent].map(|layer| medium.slot(layer).unwrap());
        let used = slots.map(|slot| {
            selection.map_or(present[i], |s| s.finishes.contains(&(Surface::Soil, slot)))
        });
        if !used.iter().any(|&u| u) {
            continue;
        }
        let stock = Stock::sample(scene.material_seed(), medium);
        // A fine perlite chip can be under four millimetres. Keep several
        // texels across it at the authored physical repeat, then filter for distance.
        let bed = sheet::sheet_material::<128>(images, stock.period, 0., 1., |uv| stock.texel(uv));
        // The geometry supplies the larger grain's silhouette. Resolve its
        // dielectric pigment separately while retaining fine pores and roughness.
        for j in 0..3 {
            if !used[j] {
                continue;
            }
            let mut m = bed.clone();
            if j > 0 {
                let rgb = stock.grain * if j == 1 { 0.94 } else { 0.71 };
                m.base_color = Color::srgb(rgb.x, rgb.y, rgb.z);
                m.base_color_texture = None;
                m.uv_transform = bevy::math::Affine2::from_scale(Vec2::splat(3.8)) * m.uv_transform;
            }
            variants.insert((Surface::Soil, slots[j]), materials.add(m));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    #[test]
    fn complete_palette_includes_the_planted_exterior_when_the_room_has_no_pots() {
        use super::super::super::architecture::outdoors;
        let (scene, slots) = (0..48)
            .find_map(|seed| {
                let mut scene =
                    IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.)
                        .unwrap();
                scene.objects.retain(|o| o.kind != ObjectKind::Plant);
                let slots: std::collections::BTreeSet<_> = outdoors::pergola_plants(&scene)
                    .flat_map(|o| {
                        let medium = Medium::sample(&o);
                        [Layer::Bed, Layer::Grain, Layer::Accent]
                            .into_iter()
                            .filter_map(move |layer| medium.slot(layer))
                    })
                    .collect();
                (!slots.is_empty()).then_some((scene, slots))
            })
            .expect("sample includes a furnished pergola");
        let mut images = Assets::<Image>::default();
        let mut mats = Assets::<StandardMaterial>::default();
        let mut vars = BTreeMap::new();
        install(&scene, &mut images, &mut mats, None, &mut vars);
        assert_eq!(
            vars.keys()
                .map(|(_, slot)| *slot)
                .collect::<std::collections::BTreeSet<_>>(),
            slots
        );
        assert_eq!(images.len(), slots.len());
    }
    #[test]
    fn map_sets_are_lazy_opaque_and_shared_by_both_grain_tints() {
        let mut scene =
            IndoorManifest::generate_with_humans(41, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut selection = MaterialSelection {
            surfaces: [Surface::Soil].into(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let mut images = Assets::<Image>::default();
        let mut mats = Assets::<StandardMaterial>::default();
        let mut vars = BTreeMap::new();
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert!(images.is_empty() && mats.is_empty());
        for layer in [Layer::Bed, Layer::Grain, Layer::Accent] {
            let slot = Medium::PerliteMix.slot(layer).unwrap();
            assert!(registered_finish(Surface::Soil, slot));
            selection.finishes.insert((Surface::Soil, slot));
        }
        scene.generator_version = soil::INTRODUCED - 1;
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert!(images.is_empty() && vars.is_empty());
        scene.generator_version = soil::INTRODUCED;
        assert!(selection.omits_parent_maps(&scene, Surface::Soil));
        install(&scene, &mut images, &mut mats, Some(&selection), &mut vars);
        assert_eq!((images.len(), vars.len()), (3, 3));
        let get = |l| {
            mats.get(&vars[&(Surface::Soil, Medium::PerliteMix.slot(l).unwrap())])
                .unwrap()
        };
        let bed = get(Layer::Bed);
        for layer in [Layer::Grain, Layer::Accent] {
            let f = get(layer);
            assert!(f.base_color_texture.is_none() && bed.base_color_texture.is_some());
            assert_eq!(f.normal_map_texture, bed.normal_map_texture);
            assert_eq!(f.metallic_roughness_texture, bed.metallic_roughness_texture);
            assert_eq!(f.alpha_mode, AlphaMode::Opaque);
        }
        for h in [
            &bed.normal_map_texture,
            &bed.base_color_texture,
            &bed.metallic_roughness_texture,
        ] {
            let image = images.get(h.as_ref().unwrap()).unwrap();
            assert_eq!(image.texture_descriptor.mip_level_count, 8);
        }
    }
    #[test]
    fn metric_relief_and_pigment_repeat_at_the_tile_seam_and_vary_with_stock() {
        for m in Medium::GRADED {
            let a = Stock::sample(28, m);
            let b = Stock::sample(53, m);
            let mut lo = 1_f32;
            let mut hi = 0_f32;
            let mut lo_h = 1_f32;
            let mut hi_h = 0_f32;
            let mut changed = 0;
            let mut fines = 0;
            let mut raised = 0;
            for x in 0..16 {
                for y in 0..16 {
                    let uv = Vec2::new(x as f32 + 0.25, y as f32 + 0.50) / 16.;
                    let (c, h, r) = a.texel(uv);
                    let seam = a.texel(uv + Vec2::new(1., -1.));
                    assert!(
                        c.abs_diff_eq(seam.0, 5e-5)
                            && (h - seam.1).abs() < 1e-7
                            && (r - seam.2).abs() < 5e-5
                    );
                    assert!(c.is_finite() && (0.55..=1.).contains(&r));
                    lo = lo.min(c.y);
                    hi = hi.max(c.y);
                    lo_h = lo_h.min(h);
                    hi_h = hi_h.max(h);
                    fines += usize::from(h < a.relief * 0.1);
                    raised += usize::from(h > a.relief * 0.45);
                    changed += usize::from(!c.abs_diff_eq(b.texel(uv).0, 0.01));
                }
            }
            assert!(hi - lo > 0.06 && hi_h - lo_h > 0.00016 && changed > 100);
            assert!(
                fines >= 25 && raised >= 5,
                "{m:?}: {fines} fines, {raised} raised"
            );
        }
    }
}
