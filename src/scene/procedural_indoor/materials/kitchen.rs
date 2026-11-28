//! Small shared food finishes. Only requested maps are synthesized; existing GPU
//! material bakes and the symbolic transport resolver use these same handles.
use super::super::preparation::AssetStore;
use super::*;
use rand::Rng;

fn pigment(seed: u64, slot: usize, base: [f32; 3]) -> [f32; 3] {
    let mut r = super::super::layout::stream(seed, 0x464f4f44434f4c52 ^ slot as u64);
    // Ripeness and baking vary within a plausible substrate gamut. The bounded
    // shared food palette costs the same number of maps as before.
    let value = r.random_range(0.72..1.18);
    let warmth = r.random_range(-0.06..0.08);
    [
        base[0] * value,
        base[1] * value * (1. + warmth),
        base[2] * value * (1. - warmth),
    ]
    .map(|c| c.clamp(0.015, 0.96))
}
pub(super) fn build(
    scene: &IndoorManifest,
    base: &[Handle<StandardMaterial>],
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    used: Option<&std::collections::BTreeSet<(Surface, usize)>>,
) -> std::collections::BTreeMap<(Surface, usize), Handle<StandardMaterial>> {
    let mut result = std::collections::BTreeMap::new();
    let skin = (Surface::Ink, produce_skin::SLOT);
    if scene.generator_version >= super::super::objects::kitchen::produce::INTRODUCED
        && used.map_or_else(
            || {
                scene.objects.iter().any(|o| {
                    o.kind == super::super::layout::ObjectKind::FruitBowl
                        || (o.kind == super::super::layout::ObjectKind::GroceryDisplay
                            && o.size.y < 1.15)
                })
            },
            |u| u.contains(&skin),
        )
    {
        result.insert(
            skin,
            materials.add(produce_skin::material(scene.material_seed(), images)),
        );
    }
    use super::super::{
        layout::ObjectKind,
        objects::{
            kitchen::{serving_for_object, Serving},
            public::casework::bakery::VERSION,
        },
    };
    let embedded = scene.generator_version >= VERSION;
    let food_case = |o: &super::super::layout::IndoorObject| {
        o.kind == ObjectKind::DisplayCase && o.variant < 16
    };
    // Selected geometry also describes the children of composite furnishings.
    // Older manifests retain their original kitchen-root palette gate.
    if (!embedded || used.is_none())
        && !scene.objects.iter().any(|o| {
            o.kind.is_kitchen() || o.kind == ObjectKind::HotFoodWell || (embedded && food_case(o))
        })
    {
        return result;
    }
    if embedded {
        let needs_bread = || {
            scene.objects.iter().any(|o| {
                food_case(o)
                    || o.kind == ObjectKind::Bread
                    || matches!(
                        serving_for_object(o),
                        Some(Serving::Toast | Serving::Biscuit)
                    )
            })
        };
        for finish in baking::Finish::ALL {
            let key = (finish.surface(), finish.slot());
            if used.map_or_else(needs_bread, |set| set.contains(&key)) {
                result.insert(
                    key,
                    materials.add(finish.material(scene.material_seed(), images)),
                );
            }
        }
    }
    let current_soup = scene.generator_version >= broth::INTRODUCED;
    let requests = |surface, slot| {
        used.map_or_else(
            || {
                scene.objects.iter().any(|o| {
                    o.kind == super::super::layout::ObjectKind::HotFoodWell
                        && slot == broth::SLOT
                        && super::super::objects::public::hot_food::HotFoodWell::for_object(o)
                            .has_soup()
                        || super::super::objects::kitchen::serving_for_object(o)
                            == Some(super::super::objects::kitchen::Serving::Soup)
                })
            },
            |u| u.contains(&(surface, slot)),
        )
    };
    if current_soup && requests(Surface::Paper, broth::HERB_SLOT) {
        result.insert(
            (Surface::Paper, broth::HERB_SLOT),
            materials.add(broth::herb(scene.material_seed())),
        );
    }
    for (slot, surface, color) in [
        (100, Surface::Ceramic, [0.68, 0.12, 0.055]),
        (101, Surface::Ceramic, [0.42, 0.55, 0.13]),
        (102, Surface::Ceramic, [0.88, 0.64, 0.13]),
        (103, Surface::Terracotta, [0.53, 0.27, 0.085]),
        (104, Surface::Paper, [0.85, 0.69, 0.43]),
        (105, Surface::Ceramic, [0.70, 0.24, 0.095]),
        (106, Surface::Terracotta, [0.27, 0.14, 0.08]),
    ] {
        if used.is_some_and(|u| !u.contains(&(surface, slot))) {
            continue;
        }
        if current_soup && slot == broth::SLOT {
            if requests(surface, slot) {
                result.insert(
                    (surface, slot),
                    materials.add(broth::material(scene.material_seed(), images)),
                );
            }
            continue;
        }
        let mut mat = materials.get(&base[surface as usize]).unwrap().clone();
        let color = if slot == broth::SLOT {
            let mut menu = super::super::layout::stream(scene.material_seed(), 0x534f555050414c);
            [color, [0.83, 0.47, 0.12], [0.43, 0.48, 0.19]][menu.random_range(0..3)]
        } else {
            color
        };
        let color = pigment(scene.material_seed(), slot, color);
        let n = 64;
        let mut pixels = Vec::with_capacity(n * n * 4);
        for y in 0..n {
            for x in 0..n {
                let u = x as f32 / n as f32;
                let v = y as f32 / n as f32;
                let mottling =
                    periodic_noise(u, v, 7, 9, scene.material_seed().wrapping_add(slot as u64));
                let pores = periodic_noise(u, v, 23, 27, slot as u64);
                let variation = 0.85 + 0.19 * mottling + 0.07 * pores;
                for c in color {
                    pixels.push((c * variation * 255.).clamp(0., 255.) as u8);
                }
                pixels.push(255);
            }
        }
        mat.base_color = Color::WHITE;
        mat.base_color_texture = Some(images.add(mip_image(pixels, n as u32, MapType::Color)));
        mat.metallic = 0.;
        mat.perceptual_roughness = if slot < 103 { 0.35 } else { 0.86 };
        mat.clearcoat = if slot < 103 { 0.12 } else { 0. };
        if slot == 105 {
            // A calm liquid has a level dielectric interface. Glazed ceramic
            // normal and ORM maps would emboss the broth with porcelain relief.
            mat.normal_map_texture = None;
            mat.metallic_roughness_texture = None;
            mat.occlusion_texture = None;
            mat.perceptual_roughness = 0.26;
            mat.reflectance = 0.35;
            mat.ior = 1.33;
        }
        mat.uv_transform = bevy::math::Affine2::from_scale(Vec2::splat(7.));
        result.insert((surface, slot), materials.add(mat));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn food_pigments_vary_without_changing_substrate_or_palette_capacity() {
        let mut colors = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let c = pigment(seed, 103, [0.53, 0.27, 0.085]);
            assert_eq!(c, pigment(seed, 103, [0.53, 0.27, 0.085]));
            assert!(c[0] > c[1] && c[1] > c[2]);
            colors.insert(c.map(|v| (v * 255.).round() as u8));
        }
        assert!(colors.len() > 100);
    }
    #[test]
    fn selected_composite_food_finishes_are_bound_without_a_kitchen_root() {
        use super::super::super::{
            layout::ObjectKind,
            objects::{audit, public::casework::bakery::VERSION},
        };
        let mut scene =
            IndoorManifest::generate_with_humans(0, Default::default(), 0.5, 0, 0.).unwrap();
        let mut cabinet = audit::fixture(ObjectKind::DisplayCase, 31);
        cabinet.variant = 0;
        scene.objects = vec![cabinet];
        let (mut images, mut mats) = (Assets::default(), Assets::<StandardMaterial>::default());
        let base = vec![mats.add(StandardMaterial::default()); Surface::ALL.len()];
        let requested = std::collections::BTreeSet::from([
            (Surface::Terracotta, 106),
            (Surface::Terracotta, 1170),
            (Surface::Paper, 1171),
        ]);
        scene.generator_version = VERSION;
        let bound = build(&scene, &base, &mut images, &mut mats, Some(&requested));
        assert_eq!(
            bound
                .keys()
                .copied()
                .collect::<std::collections::BTreeSet<_>>(),
            requested
        );
        for handle in bound.values() {
            assert!(mats.get(handle).unwrap().base_color_texture.is_some());
        }
        let before = images.len();
        let empty = build(
            &scene,
            &base,
            &mut images,
            &mut mats,
            Some(&Default::default()),
        );
        assert!(empty.is_empty());
        assert_eq!(before, images.len());
        let full = build(&scene, &base, &mut images, &mut mats, None);
        assert!(requested.iter().all(|key| full.contains_key(key)));
        scene.generator_version -= 1;
        assert!(build(&scene, &base, &mut images, &mut mats, Some(&requested)).is_empty());
    }
}
