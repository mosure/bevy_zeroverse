//! Finite sky geometry closes exterior rays without changing the primary-room bounds.
//! Its metric depth is the distance to this proxy, not an astronomical sky distance.
use super::*;
use crate::scene::procedural_indoor::{preparation::AssetStore, sky};

/// Directional emissive UVs on the regular annotated StandardMaterial mesh.
pub(crate) const SKY_TAG: bevy::mesh::MeshTag = bevy::mesh::MeshTag::new(0x5a56534b);

pub(crate) fn radius(scene: &IndoorManifest) -> f32 {
    // All primary-room cameras and the adjoining room are inside. Even at the
    // largest generated footprint, the opposite sky surface is within the
    // existing 50 m capture range. No kilometre-scale far plane is necessary.
    scene.room_size.xz().length() * 0.5 + 18.
}

pub(crate) fn append(a: &mut Assembly, scene: &IndoorManifest) {
    let g = a.part(Surface::Light, "sky#exterior");
    g.ellipsoid(Vec3::splat(radius(scene)), Transform::IDENTITY);
    for normal in &mut g.normals {
        *normal = (-Vec3::from_array(*normal)).to_array();
    }
    for triangle in g.indices.as_chunks_mut::<3>().0 {
        triangle.swap(1, 2);
    }
}

pub(crate) fn material(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
) -> (StandardMaterial, sky::SkyField) {
    let (radiance, texture, field) = match sky::bake(scene) {
        Some(map) => (map.emission, Some(images.add(map.texture)), map.field),
        // The oldest manifests predate both the daylight meter and the
        // weather field. Their visible shell used a fixed emission.
        None => (
            scene
                .domain()
                .map_or(Vec3::splat(45.), |d| d.photometry.sky_radiance),
            None,
            sky::SkyField::default(),
        ),
    };
    (
        StandardMaterial {
            base_color: Color::BLACK,
            emissive: LinearRgba::new(radiance.x, radiance.y, radiance.z, 1.),
            emissive_texture: texture,
            emissive_channel: bevy::mesh::UvChannel::Uv1,
            emissive_exposure_weight: 1.,
            reflectance: 0.,
            perceptual_roughness: 1.,
            ..default()
        },
        field,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sky_is_inward_closed_bounded_and_outside_primary_room() {
        for seed in 0..128 {
            let scene = IndoorManifest::generate_with_humans(
                seed,
                super::super::super::layout::IndoorLayout::Mixed,
                0.65,
                3,
                0.,
            )
            .unwrap();
            let mut a = Assembly::default();
            append(&mut a, &scene);
            let g = &a.parts[&(Surface::Light, "sky#exterior".into())];
            let r = radius(&scene);
            assert!(r > scene.room_size.xz().length() * 0.5 + 10.);
            for n in &g.normals {
                assert!((Vec3::from_array(*n).length() - 1.).abs() < 1e-5);
            }
            for t in g.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
                assert!((b - a).cross(c - a).dot(a + b + c) < 1e-4);
            }
            for camera in &scene.cameras {
                for t in [0., 0.5, 1.] {
                    let position = camera.transform_at(t).translation;
                    assert!(position.length() < r - 10.);
                    assert!(
                        position.length() + r < 50.,
                        "seed {seed}: sky beyond capture range"
                    );
                }
            }
        }
    }
}
