//! One mineral surface per installed batch. Kiln or casting shades share its
//! filtered color, normal and roughness maps across all the units in a room.
use super::*;
use crate::scene::procedural_indoor::{layout::stream, preparation::AssetStore};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

const FIRST: usize = 576;
pub(crate) const MORTAR: usize = 3;
const PER_LOT: usize = MORTAR + 1;

fn slot(lot: usize, tone: usize) -> usize {
    FIRST + lot * PER_LOT + tone
}

pub(crate) fn label(lot: usize, tone: usize) -> String {
    format!("wall#finish{}", slot(lot, tone))
}

pub(crate) fn tone_at(col: i32, row: i32, phase: f32) -> usize {
    let h = hash(col as u32, row as u32, u64::from(phase.to_bits()));
    if h < 0.23 {
        0
    } else if h < 0.80 {
        1
    } else {
        2
    }
}

pub(super) fn supports(surface: Surface, i: usize) -> bool {
    (FIRST..FIRST + variants::COUNT * PER_LOT).contains(&i)
        && (surface == Surface::Concrete
            || (surface == Surface::Terracotta && (i - FIRST) % PER_LOT != MORTAR))
}

fn stock(
    seed: u64,
    surface: Surface,
    lot: usize,
    mortar: bool,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut r = stream(
        seed,
        u64::from_le_bytes(*b"ClayFace")
            ^ slot(lot, if mortar { MORTAR } else { 0 }) as u64
            ^ ((surface as u64) << 24),
    );
    let fired = surface == Surface::Terracotta;
    let pigment = if fired {
        let (a, b) = match r.random_range(0..10) {
            0..=4 => (Vec3::new(0.43, 0.22, 0.15), Vec3::new(0.77, 0.47, 0.30)),
            5..=7 => (Vec3::new(0.57, 0.40, 0.28), Vec3::new(0.84, 0.72, 0.53)),
            _ => (Vec3::new(0.30, 0.235, 0.22), Vec3::new(0.59, 0.37, 0.32)),
        };
        a.lerp(b, r.random())
    } else {
        let v = if mortar {
            r.random_range(0.48..0.81)
        } else {
            r.random_range(0.41..0.73)
        };
        let warm = r.random_range(-0.015..0.055);
        Vec3::new(v + warm, v, v - warm * 0.8)
    };
    let period = r.random_range(0.125..0.228);
    let pitch = Vec2::new(
        period * if fired { r.random_range(1.1..1.7) } else { 1. },
        period,
    );
    let relief = if mortar {
        r.random_range(0.00016..0.00056)
    } else {
        r.random_range(0.0004..0.0017)
    };
    let salt = r.random();
    sheet::sheet_material::<64>(images, pitch, 0., 1., |uv| {
        let fine = periodic_noise(uv.x, uv.y, 29, 31, salt) - 0.5;
        let grain = periodic_noise(uv.x, uv.y, 19, 21, salt ^ 1);
        let cloud = periodic_noise(uv.x, uv.y, if fired { 3 } else { 6 }, 8, salt ^ 2) - 0.5;
        let pore = ((0.40 - grain) * 3.1).clamp(0., 1.);
        (
            pigment * (1. + cloud * 0.18 + fine * 0.18 - pore * 0.22),
            relief * (fine * 0.42 + cloud * 0.26 - pore),
            (0.84 + 0.13 * pore + fine * 0.11).clamp(0., 1.),
        )
    })
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    mats: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    finishes: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let Some(e) = &scene.envelope else { return };
    let mut lots = BTreeSet::new();
    for s in e
        .walls
        .iter()
        .filter_map(|w| w.lining)
        .filter(|s| s.stock.is_masonry())
    {
        lots.insert((s.stock.surface(), s.finish_slot, false));
        lots.insert((Surface::Concrete, s.finish_slot, true));
    }
    for (surface, lot, mortar) in lots {
        let mut base = None;
        let seed = scene.material_seed();
        let contrast = stream(seed, u64::from_le_bytes(*b"KilnTone") ^ lot as u64).random_range(
            0.055..if surface == Surface::Terracotta {
                0.26
            } else {
                0.17
            },
        );
        for tone in if mortar { MORTAR..PER_LOT } else { 0..MORTAR } {
            let key = (surface, slot(lot, tone));
            if selection.is_some_and(|s| !s.finishes.contains(&key)) {
                continue;
            }
            let mut m = base
                .get_or_insert_with(|| stock(seed, surface, lot, mortar, images))
                .clone();
            if !mortar {
                let reflectance = 1. + (tone as f32 - 1.) * contrast;
                m.base_color = Color::linear_rgb(reflectance, reflectance, reflectance);
            }
            finishes.insert(key, mats.add(m));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        envelope::lining::{LiningStock, WallCourses, WallLining},
        layout::IndoorLayout,
    };

    #[test]
    fn only_installed_batches_allocate_maps_and_tones_share_them() {
        let mut scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let mut images = Assets::default();
        let mut mats = Assets::default();
        let mut finishes = BTreeMap::new();
        for w in &mut scene.envelope.as_mut().unwrap().walls {
            w.lining = None;
        }
        install(&scene, &mut images, &mut mats, None, &mut finishes);
        assert_eq!(images.len(), 0);
        for stock in [LiningStock::FiredBrick, LiningStock::ConcreteBlock] {
            scene.envelope.as_mut().unwrap().walls[0].lining = Some(WallLining {
                stock,
                height_m: 2.6,
                module_width_m: 0.31,
                phase: 0.24,
                joint_width_m: 0.013,
                joint_depth_m: 0.004,
                finish_slot: 1,
                courses: Some(WallCourses {
                    height_m: if stock == LiningStock::FiredBrick {
                        0.083
                    } else {
                        0.16
                    },
                    phase: 0.21,
                    bond_offset: 0.5,
                }),
            });
            let prior = images.len();
            install(&scene, &mut images, &mut mats, None, &mut finishes);
            assert_eq!(images.len() - prior, 6);
            let tones = [0, 1, 2].map(|tone| {
                mats.get(&finishes[&(stock.surface(), slot(1, tone))])
                    .unwrap()
            });
            assert!(tones[0].base_color.to_linear().red < tones[2].base_color.to_linear().red);
            for m in tones {
                assert_eq!(m.base_color_texture, tones[1].base_color_texture);
                assert_eq!(m.normal_map_texture, tones[1].normal_map_texture);
                assert_eq!(
                    m.metallic_roughness_texture,
                    tones[1].metallic_roughness_texture
                );
                assert_eq!(m.alpha_mode, AlphaMode::Opaque);
            }
            let m = tones[0];
            for (i, tex) in [
                &m.base_color_texture,
                &m.normal_map_texture,
                &m.metallic_roughness_texture,
            ]
            .into_iter()
            .enumerate()
            {
                let im = images.get(tex.as_ref().unwrap()).unwrap();
                assert_eq!(im.texture_descriptor.mip_level_count, 7);
                assert_eq!(
                    im.texture_descriptor.format,
                    if i == 0 {
                        TextureFormat::Rgba8UnormSrgb
                    } else {
                        TextureFormat::Rgba8Unorm
                    }
                );
                let p = im.data.as_ref().unwrap().as_chunks::<4>().0;
                assert!(p[..4096].windows(2).any(|w| w[0] != w[1]));
                if i == 2 {
                    assert!(p.iter().all(|s| s[2] == 0 && s[1] > 185));
                }
            }
            let selection = MaterialSelection {
                surfaces: [stock.surface()].into(),
                finishes: [(stock.surface(), slot(1, 0))].into(),
                direct_surfaces: Default::default(),
            };
            assert!(selection.omits_parent_maps(&scene, stock.surface()));
            let mut subset = BTreeMap::new();
            install(
                &scene,
                &mut images,
                &mut mats,
                Some(&selection),
                &mut subset,
            );
            assert_eq!(subset.len(), 1);
            assert_eq!(images.len() - prior, 9);
        }
    }
}
