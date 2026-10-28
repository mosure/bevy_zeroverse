use super::*;

#[test]
fn finished_surfaces_bind_the_resolved_normal_to_the_outer_coat() {
    use crate::scene::procedural_indoor::{layout::IndoorLayout, IndoorQuality};
    let scene =
        IndoorManifest::generate_with_humans(81, IndoorLayout::Conference, 0.5, 0, 0.).unwrap();
    let surfaces = [
        Surface::Wood,
        Surface::Ceramic,
        Surface::Leather,
        Surface::Fabric,
    ];
    let selection = MaterialSelection {
        surfaces: surfaces.into_iter().collect(),
        direct_surfaces: surfaces.into_iter().collect(),
        finishes: surfaces
            .into_iter()
            .flat_map(|s| (0..3).map(move |i| (s, i)))
            .collect(),
    };
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let set = IndoorMaterials::build_with_selection(
        &scene,
        IndoorQuality::Auto,
        &mut images,
        &mut materials,
        Some(&selection),
        false,
    );
    let mut coats = 0;
    for surface in surfaces {
        for handle in std::iter::once(set.get(surface))
            .chain((0..3).map(|i| set.for_part(surface, &format!("test#finish{i}"))))
        {
            let material = materials.get(&handle).unwrap();
            assert!(material.normal_map_texture.is_some());
            assert!((0.85..=1.).contains(&material.perceptual_roughness));
            if material.clearcoat > 0. {
                coats += 1;
                assert_eq!(
                    material.clearcoat_normal_texture,
                    material.normal_map_texture
                );
                assert_eq!(
                    material.clearcoat_normal_channel,
                    material.normal_map_channel
                );
            } else {
                assert!(material.clearcoat_normal_texture.is_none());
            }
            let normal = images
                .get(material.normal_map_texture.as_ref().unwrap())
                .unwrap();
            assert_eq!(normal.texture_descriptor.format, TextureFormat::Rgba8Unorm);
        }
    }
    assert!(coats >= 8);
    // Planar and uncoated materials must not inherit stale outer-layer normals.
    let mut planar = StandardMaterial {
        clearcoat: 0.4,
        ..default()
    };
    conformal_coat(&mut planar);
    assert!(planar.clearcoat_normal_texture.is_none());
    planar.normal_map_texture = Some(Handle::default());
    conformal_coat(&mut planar);
    assert!(planar.clearcoat_normal_texture.is_some());
    planar.clearcoat = 0.;
    conformal_coat(&mut planar);
    assert!(planar.clearcoat_normal_texture.is_none());
}

#[test]
fn knit_template_preserves_absolute_baked_roughness() {
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let mut requests = gpu::Requests::default();
    let handle = human::knit(81, &mut images, &mut materials, false, &mut requests);
    let material = materials.get(&handle).unwrap();
    assert_eq!(material.perceptual_roughness, 1.);
    assert_eq!(material.metallic, 0.);
    assert_eq!(material.clearcoat, 0.);
    let map = images
        .get(material.metallic_roughness_texture.as_ref().unwrap())
        .unwrap();
    assert!(map
        .data
        .as_ref()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| p[1] > 180 && p[2] == 0));
}

fn slope_rms(normal: &Image) -> f32 {
    let size = (normal.width() * normal.height()) as usize;
    (normal.data.as_ref().unwrap()[..size * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let n = Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - Vec3::ONE;
            assert!(n.z > 0.5 && (n.length() - 1.).abs() < 0.015);
            n.truncate().length_squared() / n.z.powi(2)
        })
        .sum::<f32>()
        / size as f32)
        .sqrt()
}

#[test]
fn rough_construction_poles_retain_resolved_relief_and_smooth_finishes() {
    for surface in [
        Surface::Wood,
        Surface::Paint,
        Surface::Concrete,
        Surface::Leather,
    ] {
        let mut recipe = program::sample(7)[surface as usize].clone();
        // Compare bare plaster, not an independently authored wallpaper covering.
        if let Some(c) = &mut recipe.coating {
            c.wall = None;
        }
        let mut slopes = Vec::new();
        for weights in [[1., 0., 0., 0.], [0., 1., 0., 0.]] {
            let mut recipe = recipe.clone();
            construction::apply(
                &mut recipe,
                construction::ConstructionSample {
                    weights,
                    scale: 1.,
                    variation: 0.65,
                },
                0,
            )
            .unwrap();
            slopes.push(slope_rms(&recipe.maps(0)[1]));
        }
        assert!(
            slopes[1] > 0.009 && slopes[1] > slopes[0] * 1.15,
            "{surface:?} loses rough-construction relief: {slopes:?}"
        );
        assert!(
            slopes[1] < 0.30,
            "{surface:?} acquired exaggerated relief: {slopes:?}"
        );
    }
}
