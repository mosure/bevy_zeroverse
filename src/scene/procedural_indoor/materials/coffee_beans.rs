//! A hopper's bean bed uses one metric, periodic surface instead of hundreds of meshes.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, ObjectKind},
    objects::public::espresso,
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const SLOT: usize = 1060;
pub(crate) const LABEL: &str = "appliance#finish1060#uv_local";

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    surface == Surface::Terracotta && slot == SLOT
}

#[derive(Clone, Copy)]
struct Bean {
    offset: Vec2,
    axis: Vec2,
    inverse_radii: Vec2,
    color: f32,
    bend: f32,
}

fn stock(seed: u64, images: &mut impl AssetStore<Image>) -> StandardMaterial {
    const N: usize = 5;
    let mut rng = stream(seed, u64::from_le_bytes(*b"BeanRoas"));
    let period = rng.random_range(0.047..0.064);
    let roast = rng.random_range(0.04..0.96);
    let pigment = Vec3::new(0.45, 0.263, 0.152).lerp(Vec3::new(0.205, 0.112, 0.079), roast);
    let beans: [Bean; N * N] = std::array::from_fn(|_| {
        let angle: f32 = rng.random_range(0.0..std::f32::consts::PI);
        Bean {
            offset: Vec2::new(rng.random_range(0.26..0.74), rng.random_range(0.26..0.74)),
            axis: Vec2::from_angle(angle),
            inverse_radii: Vec2::new(
                1. / rng.random_range(0.53..0.68),
                1. / rng.random_range(0.34..0.46),
            ),
            color: rng.random_range(0.82..1.16),
            bend: rng.random_range(-0.27..0.27),
        }
    });
    sheet::sheet_material::<64>(images, Vec2::splat(period), 0., 1., |uv| {
        let p = uv * N as f32;
        let cell = p.floor().as_ivec2();
        let mut height = 0.;
        let mut dye = pigment * 0.37;
        let mut rough = 0.68;
        for y in cell.y - 1..=cell.y + 1 {
            for x in cell.x - 1..=cell.x + 1 {
                let b =
                    beans[(y.rem_euclid(N as i32) as usize) * N + x.rem_euclid(N as i32) as usize];
                let d = p - Vec2::new(x as f32, y as f32) - b.offset;
                let q = Vec2::new(d.dot(b.axis), b.axis.perp_dot(d)) * b.inverse_radii;
                let dome = 1. - q.length_squared();
                if dome <= height {
                    continue;
                }
                let seam =
                    (1. - (q.y - b.bend * (1. - q.x * q.x)).abs() * 5.5).clamp(0., 1.) * dome;
                let exposed = dome * (1. - 0.42 * seam);
                if exposed > height {
                    height = exposed;
                    dye = pigment * b.color * (0.70 + 0.30 * dome - 0.32 * seam);
                    rough = 0.58 - 0.21 * roast - 0.085 * dome + 0.12 * seam;
                }
            }
        }
        (dye, height * 0.0017, rough)
    })
}
pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    let key = (Surface::Terracotta, SLOT);
    if scene.generator_version >= espresso::VERSION
        && selection.map_or_else(
            || {
                scene.objects.iter().any(|o| {
                    o.kind == ObjectKind::CoffeeGrinder
                        && espresso::Grinder::for_object(o).fill_fraction > 0.
                })
            },
            |s| s.finishes.contains(&key),
        )
    {
        variants.insert(key, materials.add(stock(scene.material_seed(), images)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn espresso_bean_stock_has_seeded_relief_and_filtered_registered_maps() {
        let bake = |seed| {
            let mut imgs = Assets::<Image>::default();
            let m = stock(seed, &mut imgs);
            let period = m.uv_transform.matrix2.x_axis.length().recip();
            assert!((0.046..0.065).contains(&period));
            [
                m.base_color_texture,
                m.normal_map_texture,
                m.metallic_roughness_texture,
            ]
            .map(|h| {
                let i = imgs.remove(h.unwrap().id()).unwrap();
                assert_eq!(i.texture_descriptor.mip_level_count, 7);
                i.data.unwrap()
            })
        };
        let a = bake(38);
        assert_eq!(a, bake(38));
        assert!(a.iter().zip(bake(39)).all(|(a, b)| a != &b));
        assert!(
            a[1].as_chunks::<4>()
                .0
                .iter()
                .filter(|v| v[0].abs_diff(128) > 7)
                .count()
                > 500
        );
        assert!(a[2]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| v[2] == 0 && v[1] > 60 && v[1] < 180));
    }
}
