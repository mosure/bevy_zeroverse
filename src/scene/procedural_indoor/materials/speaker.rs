//! Acoustic-facing materials. Submillimetre relief is filtered into mipmaps;
//! the grille itself remains an opaque, fabric-backed surface for capture.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::accessories::speaker::{self as enclosure, MountKind},
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::{BTreeMap, BTreeSet};

const DARK: usize = 480;
const PALE: usize = 481;
const YOKE: usize = 482;
pub(crate) const DARK_LABEL: &str = "speaker#finish480#uv_local";
pub(crate) const PALE_LABEL: &str = "speaker#finish481#uv_local";
pub(crate) const DAMPER_LABEL: &str = DARK_LABEL;
pub(crate) const YOKE_LABEL: &str = "speaker#finish482";

pub(super) fn supports(s: Surface, slot: usize) -> bool {
    matches!(
        (s, slot),
        (
            Surface::Fabric | Surface::Metal | Surface::Plastic,
            DARK | PALE
        ) | (Surface::Metal, YOKE)
            | (Surface::Rubber, DARK)
    )
}

struct Facing {
    cloth: bool,
    pitch: Vec2,
    width: f32,
    pigment: Vec3,
    backing: Vec3,
    height: f32,
    roughness: f32,
}
impl Facing {
    fn sample(cloth: bool, pale: bool, r: &mut impl Rng) -> Self {
        let pitch = if cloth {
            r.random_range(0.0008..0.00185)
        } else {
            r.random_range(0.0016..0.0035)
        };
        let value = if pale {
            r.random_range(0.63..0.91)
        } else {
            r.random_range(0.105..0.36)
        };
        let tint = r.random_range(-0.046..0.060);
        Self {
            cloth,
            pitch: Vec2::new(
                pitch,
                pitch
                    * if cloth {
                        r.random_range(0.70..1.2)
                    } else {
                        0.8660254
                    },
            ),
            width: if cloth {
                r.random_range(0.28..0.43)
            } else {
                r.random_range(0.26..0.40)
            },
            pigment: Vec3::new(value * (1. + tint), value, value * (1. - tint)),
            backing: Vec3::splat(r.random_range(0.035..0.095)),
            height: pitch * r.random_range(0.09..0.21),
            roughness: if cloth {
                r.random_range(0.80..0.90)
            } else {
                r.random_range(0.46..0.67)
            },
        }
    }

    fn point(&self, uv: Vec2) -> (Vec3, f32, f32) {
        let cell = uv * 4.;
        let local = cell - cell.floor() - 0.5;
        if self.cloth {
            // Alternating warp/weft crossings change height across each yarn.
            let q = local / self.width;
            let h = (Vec2::ONE - q * q).max(Vec2::ZERO);
            let yarn = Vec2::new(h.x.sqrt(), h.y.sqrt());
            let sign = if cell.floor().as_ivec2().element_sum() & 1 == 0 {
                1.
            } else {
                -1.
            };
            let crossing = sign * (1. - 4. * local.x * local.x) * (1. - 4. * local.y * local.y);
            let crown = (yarn.x * (0.82 + 0.18 * crossing)).max(yarn.y * (0.82 - 0.18 * crossing));
            (
                self.backing.lerp(self.pigment, yarn.max_element()),
                self.height * crown,
                self.roughness + 0.06 * (1. - crown),
            )
        } else {
            // Punched metal in staggered rows. A graded lip meets dark acoustic backing.
            let shift = (cell.y.floor() as i32 & 1) as f32 * 0.5;
            let dx = (cell.x - shift).rem_euclid(1.) - 0.5;
            let radial = Vec2::new(dx, local.y * self.pitch.y / self.pitch.x).length();
            let coat = ((radial - self.width + 0.06) / 0.1).clamp(0., 1.);
            (
                self.backing.lerp(self.pigment, coat),
                self.height * coat,
                0.93 + (self.roughness - 0.93) * coat,
            )
        }
    }
}

fn material(
    seed: u64,
    s: Surface,
    slot: usize,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut r = stream(
        seed,
        u64::from_le_bytes(*b"AcousPBR") ^ ((s as u64) << 12) ^ slot as u64,
    );
    if slot != YOKE && matches!(s, Surface::Fabric | Surface::Metal) {
        let face = Facing::sample(s == Surface::Fabric, slot == PALE, &mut r);
        return sheet::sheet_material::<32>(images, 4. * face.pitch, 0., 1., |uv| face.point(uv));
    }
    let (v, roughness) = match (s, slot) {
        (Surface::Rubber, _) => (r.random_range(0.07..0.16), r.random_range(0.67..0.85)),
        (Surface::Metal, _) => (r.random_range(0.12..0.36), r.random_range(0.42..0.63)),
        (_, PALE) => (r.random_range(0.45..0.70), r.random_range(0.48..0.75)),
        _ => (r.random_range(0.16..0.34), r.random_range(0.57..0.85)),
    };
    let warmth = r.random_range(-0.035..0.085);
    StandardMaterial {
        base_color: Color::srgb(v * (1. + warmth), v, v * (1. - warmth)),
        perceptual_roughness: roughness,
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
    let keys = if let Some(selection) = selection {
        selection
            .finishes
            .iter()
            .filter(|&&(s, i)| supports(s, i))
            .copied()
            .collect()
    } else {
        let mut keys = BTreeSet::new();
        for o in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Speaker)
        {
            let p = enclosure::parameters(o);
            let slot = if p.light_facing { PALE } else { DARK };
            keys.insert((
                if p.covered {
                    p.grille
                } else {
                    Surface::Plastic
                },
                slot,
            ));
            if !p.covered {
                keys.insert((Surface::Rubber, DARK));
            }
            if p.mount.kind == MountKind::WallYoke {
                keys.insert((Surface::Metal, YOKE));
            }
        }
        keys
    };
    for (s, i) in keys {
        variants.insert(
            (s, i),
            materials.add(material(scene.material_seed(), s, i, images)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::render::render_resource::TextureFormat;

    #[test]
    fn acoustic_stock_replays_in_metric_units_and_filters_to_opaque_dielectric() {
        for s in [Surface::Fabric, Surface::Metal] {
            for slot in [DARK, PALE] {
                let bake = |seed| {
                    let mut images = Assets::default();
                    let m = material(seed, s, slot, &mut images);
                    assert_eq!(m.metallic, 0.);
                    let pitch_x = 0.25 / m.uv_transform.matrix2.x_axis.x;
                    assert!((0.00079..0.00351).contains(&pitch_x));
                    let mut maps = Vec::new();
                    for (h, format) in [
                        (m.base_color_texture, TextureFormat::Rgba8UnormSrgb),
                        (m.normal_map_texture, TextureFormat::Rgba8Unorm),
                        (m.metallic_roughness_texture, TextureFormat::Rgba8Unorm),
                    ] {
                        let im = images.get(h.unwrap().id()).unwrap();
                        assert_eq!(im.texture_descriptor.format, format);
                        assert_eq!(im.texture_descriptor.mip_level_count, 6);
                        maps.push(im.data.as_ref().unwrap().clone());
                    }
                    assert!(maps[0].as_chunks::<4>().0.iter().all(|c| c[3] == 255));
                    assert!(maps[2]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|c| c[2] == 0 && c[1] > 100));
                    let normals = maps[1].as_chunks::<4>().0;
                    assert!(normals[..1024].iter().any(|c| c[0].abs_diff(127) > 12));
                    let n = normals.last().unwrap();
                    assert!(n[0].abs_diff(127) < 10 && n[1].abs_diff(127) < 10 && n[2] > 250);
                    (pitch_x, maps)
                };
                let a = bake(4);
                assert_eq!(a, bake(4));
                assert_ne!(a, bake(5));
            }
        }
    }
    #[test]
    fn full_and_consumed_material_builders_bind_only_realized_acoustic_finishes() {
        use crate::scene::procedural_indoor::{
            layout::IndoorLayout,
            objects::{audit, build_object},
        };
        let mut scene =
            IndoorManifest::generate_with_humans(13, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        let mut selection = MaterialSelection {
            surfaces: Default::default(),
            finishes: Default::default(),
            direct_surfaces: Default::default(),
        };
        let install_keys = |scene: &IndoorManifest, selection: Option<&MaterialSelection>| {
            let mut images = Assets::default();
            let mut mats = Assets::default();
            let mut variants = BTreeMap::new();
            install(scene, &mut images, &mut mats, selection, &mut variants);
            assert_eq!(mats.len(), variants.len());
            (variants.into_keys().collect::<BTreeSet<_>>(), images.len())
        };
        assert_eq!(install_keys(&scene, None), (BTreeSet::new(), 0));
        for seed in 0..56 {
            let mut o = audit::fixture(ObjectKind::Speaker, seed);
            o.finish_seed = Some(seed);
            o.position.y = 2.;
            o.solid = false;
            for (s, label) in build_object(&o).parts.into_keys() {
                if let Some(slot) = finish_slot(s, &label).filter(|i| supports(s, *i)) {
                    assert!(registered_finish(s, slot));
                    selection.finishes.insert((s, slot));
                }
            }
            scene.objects.push(o);
        }
        assert_eq!(selection.finishes.len(), 8);
        assert_eq!(install_keys(&scene, None), (selection.finishes.clone(), 12));
        assert_eq!(
            install_keys(&scene, Some(&selection)),
            install_keys(&scene, None)
        );
        selection.finishes.clear();
        assert_eq!(install_keys(&scene, Some(&selection)), (BTreeSet::new(), 0));
    }
}
