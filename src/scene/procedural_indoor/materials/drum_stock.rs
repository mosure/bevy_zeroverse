//! Physical kettle and membrane stock. Each fitted kit consumes one bowl
//! finish and one matte film, shared by all of its drums and camera views.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::public::music::{timpani::VERSION, BowlStock},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const FIRST: usize = 2190;
pub(crate) const HEAD_LABEL: &str = "musical_instrument#finish2194#uv_local";
const STOCK: [BowlStock; 4] = [
    BowlStock::HammeredCopper,
    BowlStock::SpunCopper,
    BowlStock::SatinAluminium,
    BowlStock::LacqueredComposite,
];
pub(crate) fn bowl_label(s: BowlStock) -> (Surface, &'static str) {
    match s {
        BowlStock::HammeredCopper => (Surface::Metal, "musical_instrument#finish2190#uv_local"),
        BowlStock::SpunCopper => (Surface::Metal, "musical_instrument#finish2191#uv_local"),
        BowlStock::SatinAluminium => (Surface::Metal, "musical_instrument#finish2192#uv_local"),
        BowlStock::LacqueredComposite => {
            (Surface::Plastic, "musical_instrument#finish2193#uv_local")
        }
    }
}
pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    matches!(
        (surface, slot),
        (Surface::Metal, 2190..=2192) | (Surface::Plastic, 2193..=2194)
    )
}

fn material(
    seed: u64,
    stock: Option<BowlStock>,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    use BowlStock::*;
    let mut r = stream(
        seed,
        u64::from_le_bytes(*b"DrumSkin") ^ stock.map_or(4, |s| s as u64),
    );
    let (color, rough, metal) = match stock {
        Some(HammeredCopper | SpunCopper) => (
            Vec3::new(0.97, r.random_range(0.71..0.82), r.random_range(0.58..0.67)),
            r.random_range(0.22..0.37),
            true,
        ),
        Some(SatinAluminium) => (
            Vec3::new(0.91, 0.92, 0.94),
            r.random_range(0.34..0.51),
            true,
        ),
        Some(LacqueredComposite) => (
            [
                Vec3::new(0.41, 0.18, 0.10),
                Vec3::new(0.29, 0.11, 0.12),
                Vec3::new(0.18, 0.19, 0.21),
            ][r.random_range(0..3)],
            r.random_range(0.29..0.47),
            false,
        ),
        None => (
            Vec3::new(0.92, 0.89, r.random_range(0.78..0.88)) * r.random_range(0.91..1.01),
            r.random_range(0.61..0.78),
            false,
        ),
    };
    let period = Vec2::splat(r.random_range(0.092..0.142));
    let mut budget = 4096;
    let grain = field::PreparedPeriodic::new([14, 19], r.random(), &mut budget);
    // Irregular overlapping hammer impressions. Wrap the lattice so the
    // metric cylindrical chart and every filtered mip join across its seam.
    const CELLS: i32 = 6;
    let dents: Option<[Vec2; 36]> = (stock == Some(HammeredCopper)).then(|| {
        std::array::from_fn(|_| Vec2::new(r.random_range(0.10..0.90), r.random_range(0.10..0.90)))
    });
    let hammer_depth = r.random_range(0.00090..0.00125);
    // A spinning tool follows the circumference. The shell's V chart measures
    // distance down the kettle, so the marks wrap around it without a seam.
    let tool_marks = (stock == Some(SpunCopper)).then(|| {
        (
            r.random_range(8..=13) as f32,
            r.random_range(0.00006..0.00011),
        )
    });
    let (base, normal, mut orm) = sheet::metric_maps::<64>(period, |uv| {
        let n = grain.sample(uv.x, uv.y) - 0.5;
        let relief = if let Some(dents) = &dents {
            let p = uv * CELLS as f32;
            let cell = p.floor().as_ivec2();
            let frac = p.fract();
            let mut d = 1_f32;
            for y in -1..=1 {
                for x in -1..=1 {
                    let ix = (cell.x + x).rem_euclid(CELLS);
                    let iy = (cell.y + y).rem_euclid(CELLS);
                    let delta =
                        Vec2::new(x as f32, y as f32) + dents[(iy * CELLS + ix) as usize] - frac;
                    d = d.min(delta.length_squared());
                }
            }
            -hammer_depth * (1. - d).powi(2)
        } else if let Some((turns, depth)) = tool_marks {
            depth * (std::f32::consts::TAU * (uv.y * turns + n * 0.19)).sin()
        } else {
            0.
        };
        let height = relief + 0.000022 * n;
        (
            color * (1. + n * if metal { 0.012 } else { 0.052 }),
            height,
            rough + n * 0.085 - 20. * relief,
        )
    });
    if metal {
        for pixel in orm.as_chunks_mut::<4>().0 {
            pixel[2] = 255;
        }
    }
    StandardMaterial {
        base_color_texture: Some(images.add(mip_image(base, 64, MapType::Color))),
        normal_map_texture: Some(images.add(mip_image(normal, 64, MapType::Normal))),
        metallic_roughness_texture: Some(images.add(mip_image(orm, 64, MapType::Data))),
        metallic: if metal { 1. } else { 0. },
        perceptual_roughness: 1.,
        uv_transform: bevy::math::Affine2::from_scale(period.recip()),
        ..default()
    }
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
    let mut used = [false; 5];
    if let Some(s) = selection {
        for (i, flag) in used.iter_mut().enumerate() {
            let surface = STOCK.get(i).map_or(Surface::Plastic, |&s| bowl_label(s).0);
            *flag = s.finishes.contains(&(surface, FIRST + i));
        }
    } else {
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Timpani)
        {
            used[BowlStock::sample(o.finish_seed.unwrap_or(o.seed)) as usize] = true;
            used[4] = true;
        }
    }
    for (i, _) in used.iter().enumerate().filter(|(_, on)| **on) {
        let stock = STOCK.get(i).copied();
        let surface = stock.map_or(Surface::Plastic, |s| bowl_label(s).0);
        variants.insert(
            (surface, FIRST + i),
            materials.add(material(scene.material_seed(), stock, images)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};

    #[test]
    fn bowl_metallicity_and_skin_roughness_survive_every_mip() {
        for stock in STOCK.into_iter().map(Some).chain([None]) {
            let mut images = Assets::default();
            let m = material(37, stock, &mut images);
            let [color, normal, orm] = [
                m.base_color_texture,
                m.normal_map_texture,
                m.metallic_roughness_texture,
            ]
            .map(|h| images.get(h.as_ref().unwrap()).unwrap());
            let metal = matches!(
                stock,
                Some(BowlStock::HammeredCopper | BowlStock::SpunCopper | BowlStock::SatinAluminium)
            );
            assert_eq!(m.metallic, if metal { 1. } else { 0. });
            assert!(color.texture_descriptor.format.is_srgb());
            assert!(
                !normal.texture_descriptor.format.is_srgb()
                    && !orm.texture_descriptor.format.is_srgb()
            );
            for map in [normal, color, orm] {
                assert_eq!(map.texture_descriptor.mip_level_count, 7);
            }
            let data = orm.data.as_ref().unwrap().as_chunks::<4>().0;
            assert!(data
                .iter()
                .all(|v| v[2] == if metal { 255 } else { 0 } && v[1] >= 40 && v[1] <= 225));
            if stock.is_none() {
                assert!(data.iter().all(|v| v[1] > 125));
            }
            let n = normal.data.as_ref().unwrap();
            let tail = &n[n.len() - 4..];
            assert!(tail[2] > 249);
            if stock == Some(BowlStock::HammeredCopper) {
                assert!(n[..64 * 64 * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|p| p[0].abs_diff(128) > 5));
            }
            if stock == Some(BowlStock::SpunCopper) {
                let gradient = |channel: usize| -> usize {
                    n[..64 * 64 * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|p| usize::from(p[channel].abs_diff(128)))
                        .sum()
                };
                assert!(gradient(1) > 2 * gradient(0));
            }
        }
    }

    #[test]
    fn demand_binds_one_bowl_stock_and_one_film_for_a_matched_set() {
        let mut scene =
            IndoorManifest::generate_with_humans(0, IndoorLayout::RehearsalRoom, 0.65, 0, 0.)
                .unwrap();
        let install_count = |scene: &IndoorManifest, selected: Option<&MaterialSelection>| {
            let (mut images, mut materials) = (Assets::default(), Assets::default());
            let mut bound = BTreeMap::new();
            install(scene, &mut images, &mut materials, selected, &mut bound);
            (bound.len(), images.len())
        };
        scene.objects.clear();
        assert_eq!(install_count(&scene, None), (0, 0));
        for seed in 0..3 {
            let mut drum = audit::fixture(ObjectKind::Timpani, seed);
            drum.finish_seed = Some(39);
            scene.objects.push(drum);
        }
        assert_eq!(install_count(&scene, None), (2, 6));
        let mut selected = MaterialSelection {
            surfaces: Default::default(),
            direct_surfaces: Default::default(),
            finishes: Default::default(),
        };
        assert_eq!(install_count(&scene, Some(&selected)), (0, 0));
        selected.select_finish(Surface::Plastic, FIRST + 4);
        assert_eq!(install_count(&scene, Some(&selected)), (1, 3));
    }
}
