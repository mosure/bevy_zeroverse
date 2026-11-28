//! Dyed woven tablecloth. Metric repeat, warp/weft relief and dielectric
//! roughness share the same filtered stock in every eligible material path.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::tables::{self, LinenWeave},
    preparation::AssetStore,
};
use rand::Rng;
use std::{collections::BTreeMap, f32::consts::TAU};

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Fabric && (880..884).contains(&slot)
}

fn stock(seed: u64, weave: LinenWeave, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"dyeLinen") ^ weave.slot() as u64);
    let ground = match r.random_range(0..7) {
        0 => Vec3::new(0.72, 0.74, 0.68),
        1 => Vec3::new(0.78, 0.72, 0.64),
        _ => Vec3::new(0.92, 0.90, 0.85),
    } * r.random_range(0.92..1.06);
    let dyed = [
        Vec3::new(0.47, 0.13, 0.16),
        Vec3::new(0.17, 0.30, 0.39),
        Vec3::new(0.23, 0.35, 0.27),
        Vec3::new(0.40, 0.27, 0.19),
        Vec3::new(0.38, 0.36, 0.41),
    ][r.random_range(0..5)]
        * r.random_range(0.82..1.18);
    let plain = if r.random_bool(0.32) { dyed } else { ground };
    let pitch = match weave {
        LinenWeave::Plain | LinenWeave::Herringbone => r.random_range(0.007..0.020),
        _ => r.random_range(0.034..0.108),
    };
    let period = Vec2::new(pitch, pitch * r.random_range(0.83..1.18));
    let width = r.random_range(0.22..0.62);
    let rough = r.random_range(0.76..0.91);
    let relief = (pitch * r.random_range(0.0022_f32..0.0036)).clamp(0.000020, 0.000180);
    let salt = r.random();
    let maps = sheet::metric_maps::<32>(period, |uv| {
        let warp = (TAU * 8. * uv.x).cos();
        let weft = (TAU * 8. * uv.y).cos();
        let crossing = warp * weft;
        let slub = periodic_noise(uv.x, uv.y, 5, 3, salt) - 0.5;
        let band = |x: f32| ((width - (x - 0.5).abs() * 2.) * 14. + 0.5).clamp(0., 1.);
        let herring = (TAU * (4. * (2. * uv.x - 1.).abs() + 4. * uv.y)).cos();
        let pigment = match weave {
            LinenWeave::Plain => plain,
            LinenWeave::Herringbone => plain * (1.0 + 0.043 * herring),
            LinenWeave::AwningStripe => {
                ground.lerp(dyed, 0.88 * band(uv.x) + 0.09 * band((uv.x + 0.40).fract()))
            }
            LinenWeave::BistroCheck => ground.lerp(dyed, 0.46 * (band(uv.x) + band(uv.y))),
        };
        (
            pigment * (1. + 0.016 * crossing + 0.024 * slub),
            relief * (0.70 * crossing + 0.24 * warp + 0.28 * slub),
            rough + 0.033 * crossing + 0.024 * slub,
        )
    });
    // Keep unresolved fibre relief in the specular lobe. Reducing quantized
    // normals independently at each mip would also bias a level tabletop.
    let [albedo, normal, roughness] = filter::images(maps, 32);
    StandardMaterial {
        base_color_texture: Some(images.add(albedo)),
        normal_map_texture: Some(images.add(normal)),
        metallic_roughness_texture: Some(images.add(roughness)),
        metallic: 0.,
        perceptual_roughness: 1.,
        reflectance: 0.31,
        uv_transform: bevy::math::Affine2::from_scale(period.recip()),
        ..default()
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    mats: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    if scene.generator_version < tables::LINEN_VERSION {
        return;
    }
    let mut used = [false; 4];
    if let Some(s) = selection {
        for weave in LinenWeave::ALL {
            used[weave as usize] = s.finishes.contains(&(Surface::Fabric, weave.slot()));
        }
    } else {
        for o in &scene.objects {
            if o.kind == ObjectKind::TableNapkin {
                used[super::super::clutter::napkin::weave(o) as usize] = true;
            } else if let Some(linen) = tables::dining_linen(o) {
                used[linen.weave as usize] = true;
            }
        }
    }
    for weave in LinenWeave::ALL {
        if used[weave as usize] {
            variants.insert(
                (Surface::Fabric, weave.slot()),
                mats.add(stock(scene.material_seed(), weave, images)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dining_linen_stocks_have_bounded_mipmapped_dielectric_maps() {
        let mut appearance = std::collections::BTreeSet::new();
        for w in LinenWeave::ALL {
            assert!(registered_finish(Surface::Fabric, w.slot()));
            let mut images = Assets::<Image>::default();
            let m = stock(740, w, &mut images);
            assert!(!m.double_sided && m.cull_mode == StandardMaterial::default().cull_mode);
            assert_eq!(m.metallic, 0.);
            assert_eq!(m.emissive, LinearRgba::BLACK);
            let map = |h: &Option<Handle<Image>>| images.get(h.as_ref().unwrap()).unwrap();
            for tex in [
                &m.base_color_texture,
                &m.normal_map_texture,
                &m.metallic_roughness_texture,
            ] {
                assert_eq!(map(tex).texture_descriptor.mip_level_count, 6);
            }
            let albedo = map(&m.base_color_texture).data.as_ref().unwrap();
            assert!(appearance.insert(albedo.clone()));
            let orm = map(&m.metallic_roughness_texture).data.as_ref().unwrap();
            assert!(orm
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| v[1] > 160 && v[2] == 0));
            let normal = map(&m.normal_map_texture).data.as_ref().unwrap();
            assert!(normal[..32 * 32 * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .any(|v| v[0].abs_diff(128) > 3 || v[1].abs_diff(128) > 3));
            // Filtering folds all the yarn into a nearly flat surface at distance.
            let last = normal.last_chunk::<4>().unwrap();
            let v = Vec3::new(last[0] as f32, last[1] as f32, last[2] as f32) / 127.5 - Vec3::ONE;
            assert!(
                v.normalize().z > 0.9999 && (v.length() - 1.).abs() < 0.005,
                "{w:?}: {last:?}"
            );
            let mut other = Assets::<Image>::default();
            let next = stock(741, w, &mut other);
            assert_ne!(
                albedo,
                other
                    .get(next.base_color_texture.as_ref().unwrap())
                    .unwrap()
                    .data
                    .as_ref()
                    .unwrap()
            );
        }
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn dining_linen_builds_one_shared_material_and_skips_unused_stock() {
        let scene =
            IndoorManifest::generate_with_humans(55100, IndoorLayout::Restaurant, 0.65, 0, 0.)
                .unwrap();
        let used: Vec<_> = scene
            .objects
            .iter()
            .filter_map(tables::dining_linen)
            .collect();
        assert!(used.len() >= 2);
        let key = (Surface::Fabric, used[0].weave.slot());
        assert!(used.iter().all(|cloth| cloth.weave.slot() == key.1));
        for mode in 0..3 {
            let mut images = Assets::<Image>::default();
            let mut materials = Assets::<StandardMaterial>::default();
            let mut variants = BTreeMap::new();
            let mut selection = MaterialSelection {
                surfaces: Default::default(),
                direct_surfaces: Default::default(),
                finishes: Default::default(),
            };
            if mode == 1 {
                selection.select_finish(key.0, key.1);
            }
            install(
                &scene,
                &mut images,
                &mut materials,
                (mode != 0).then_some(&selection),
                &mut variants,
            );
            let expected = if mode == 0 {
                let mut slots = std::collections::BTreeSet::from([key.1]);
                slots.extend(
                    scene
                        .objects
                        .iter()
                        .filter(|o| o.kind == ObjectKind::TableNapkin)
                        .map(|o| super::super::super::clutter::napkin::weave(o).slot()),
                );
                slots.len()
            } else {
                usize::from(mode == 1)
            };
            assert_eq!(images.len(), 3 * expected);
            assert_eq!(materials.len(), expected);
            assert_eq!(variants.contains_key(&key), mode != 2);
        }
    }
}
