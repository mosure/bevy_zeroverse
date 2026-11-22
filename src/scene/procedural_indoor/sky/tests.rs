use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn directional_radiance_preserves_metered_irradiance_at_every_weather_extreme() {
    let mut scene =
        IndoorManifest::generate_with_humans(73161, IndoorLayout::Cafe, 0.65, 0, 0.).unwrap();
    for transmission in [0., 0.04, 0.45, 0.92, 1.] {
        let sky = SkyDistribution::sample(101, transmission);
        scene
            .program
            .as_mut()
            .unwrap()
            .domain
            .as_mut()
            .unwrap()
            .photometry
            .sky_distribution = Some(sky);
        let map = bake(&scene).unwrap();
        let mean = scene.sky_radiance();
        // Independent cosine-weighted hemisphere quadrature, uniform over the projected disk.
        let mut integral = Vec3::ZERO;
        for i in 0..8192 {
            let r = ((i as f32 + 0.5) / 8192.).sqrt();
            let spin = Vec2::from_angle(i as f32 * 2.3999631);
            let d = Vec3::new(r * spin.x, (1. - r * r).sqrt(), r * spin.y);
            let l = map.field.radiance(d, mean);
            assert!(l.is_finite() && l.min_element() > 0.);
            integral += l;
        }
        assert!((integral / (8192. * mean) - 1.).abs().max_element() < 0.004);
        let horizon = map.field.radiance(Vec3::X, mean);
        let zenith = map.field.radiance(Vec3::Y, mean);
        if transmission == 0. {
            assert!(zenith.y > 2.3 * horizon.y);
        }
        if transmission == 1. {
            assert!(zenith.z / zenith.x > 1.5 * horizon.z / horizon.x);
        }
        assert_eq!(map.texture.data.as_ref().unwrap().len(), SIZE * SIZE * 4);
        assert_eq!(map.texture.data, bake(&scene).unwrap().texture.data);
    }
}

#[test]
fn changing_rig_or_material_does_not_resample_weather_and_legacy_light_is_uniform() {
    let mut scene =
        IndoorManifest::generate_with_humans(74222, IndoorLayout::Gallery, 0.65, 3, 0.).unwrap();
    let original = bake(&scene).unwrap();
    scene.cameras.clear();
    scene.world_yaw = 1.23;
    scene
        .apply_appearance(super::super::appearance::AppearanceSettings {
            material_seed: Some(2),
            illumination_scale: 0.5,
            ..default()
        })
        .unwrap();
    let dim = bake(&scene).unwrap();
    assert_eq!(original.texture.data, dim.texture.data);
    assert_eq!(original.emission * 0.5, dim.emission);
    scene
        .program
        .as_mut()
        .unwrap()
        .domain
        .as_mut()
        .unwrap()
        .photometry
        .sky_distribution = None;
    assert!(bake(&scene).is_none());
    assert_eq!(SkyField::default().radiance(Vec3::X, Vec3::ONE), Vec3::ONE);
}
