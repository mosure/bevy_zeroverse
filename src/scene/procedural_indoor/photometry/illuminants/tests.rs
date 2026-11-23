use super::*;
use crate::scene::procedural_indoor::{
    appearance::AppearanceSettings, architecture, domain::SceneDomain, layout::IndoorLayout,
    layout::IndoorManifest,
};

fn chromaticity(rgb: Vec3) -> Vec2 {
    // Independent forward sRGB to XYZ matrix for the D65 primaries.
    let x = rgb.dot(Vec3::new(0.412_390_8, 0.357_584_33, 0.180_480_8));
    let y = rgb.dot(Vec3::new(0.212_639, 0.715_168_65, 0.072_192_32));
    let z = rgb.dot(Vec3::new(0.019_330_82, 0.119_194_78, 0.950_532_14));
    Vec2::new(4. * x, 6. * y) / (x + 15. * y + 3. * z)
}

fn lamp_uv(kelvin: f32, tint: f32) -> Vec2 {
    let c = IlluminantModel::Cie1931.lamp_srgb(kelvin, tint);
    let rgb = Color::srgb(c.x, c.y, c.z).to_linear();
    chromaticity(Vec3::new(rgb.red, rgb.green, rgb.blue))
}

#[test]
fn rendered_white_points_follow_reference_spectra() {
    // Integrated 1 nm CIE 1931 2° observer × Planck spectra, 360–830 nm.
    // The 2856 K fixture also agrees with the tabulated CIE illuminant A.
    for (kelvin, u, v) in [
        (2000., 0.305_045_78, 0.359_065_6),
        (2400., 0.277_649_52, 0.355_239_93),
        (2856., 0.255_971_07, 0.349_527_1),
        (3000., 0.250_568_87, 0.347_589_25),
        (4000., 0.225_109_38, 0.334_386_6),
        (5000., 0.211_423_62, 0.323_115),
        (6500., 0.200_448_5, 0.310_361),
        (8000., 0.194_624_26, 0.301_409_1),
        (10_000., 0.190_318_52, 0.293_264_2),
    ] {
        let difference = lamp_uv(kelvin, 0.).distance(Vec2::new(u, v));
        assert!(difference < 0.00018, "{kelvin} K: Δuv={difference}");
    }
    // Independently integrated *daylight* SPDs from CIE datasets D50 and D65.
    for (k, reference) in [
        (5003., Vec2::new(0.209_167_12, 0.325_386_23)),
        (6504., Vec2::new(0.197_839_98, 0.312_224_24)),
    ] {
        let sky = IlluminantModel::Cie1931.diffuse_sky_linear(k);
        assert!(chromaticity(sky).distance(reference) < 0.00010);
        assert!(lamp_uv(k, 0.).distance(reference) > 0.002);
    }
}

#[test]
fn fixture_tint_offsets_across_the_locus_and_extreme_filaments_stay_in_gamut() {
    for k in [2300., 3000., 4000., 5700., 6800., 9300.] {
        let center = lamp_uv(k, 0.);
        let tangent = (lamp_uv(k + 5., 0.) - lamp_uv(k - 5., 0.)).normalize();
        let normal = Vec2::new(tangent.y, -tangent.x);
        for tint in [-0.055, 0.026, 0.055] {
            let offset = lamp_uv(k, tint) - center;
            assert!((offset.dot(normal) - tint * 0.1).abs() < 0.00002);
            assert!(offset.dot(tangent).abs() < 0.00002);
        }
    }
    for k in (1700..10_201).step_by(20) {
        for tint in [-0.06, 0., 0.06] {
            let c = IlluminantModel::Cie1931.lamp_srgb(k as f32, tint);
            assert!(c.is_finite() && c.min_element() >= 0.);
            assert!((c.max_element() - 1.).abs() < 1e-6);
        }
    }
    // An 1800 K filament lies outside the linear-sRGB triangle on the blue side.
    assert_eq!(IlluminantModel::Cie1931.lamp_srgb(1800., 0.).z, 0.);
}

#[test]
fn illuminant_calibration_preserves_energy_and_lighting_draws() {
    for seed in 0..256 {
        let new = SceneDomain::sample(seed);
        let old = SceneDomain::sample_with_illuminants(seed, IlluminantModel::LegacySrgb);
        let (p, q) = (&new.photometry, &old.photometry);
        assert_eq!(new.fixture_kelvin, old.fixture_kelvin);
        assert_eq!(new.target_lux, old.target_lux);
        assert_eq!(p.meter, q.meter);
        assert_eq!(p.sun_lux, q.sun_lux);
        assert_eq!(p.sun_kelvin, q.sun_kelvin);
        assert_eq!(p.sky_distribution, q.sky_distribution);
        assert!((p.diffuse_sky_lux() / q.diffuse_sky_lux() - 1.).abs() < 5e-7);
        let daylight = p
            .illuminant_model
            .diffuse_sky_linear(p.diffuse_sky_kelvin.unwrap());
        assert!(chromaticity(daylight).distance(chromaticity(p.sky_radiance)) < 1e-6);
        assert_eq!(new.validate(), Ok(()));
    }
}

#[test]
fn archived_source_colors_and_relighting_use_the_recorded_model() {
    let seed = 50800;
    let scene =
        IndoorManifest::generate_with_humans(seed, IndoorLayout::Cafeteria, 0.65, 0, 0.).unwrap();
    let mut archive = serde_json::to_value(&scene).unwrap();
    archive["generator_version"] = 168.into();
    let p = archive["program"]["domain"]["photometry"]
        .as_object_mut()
        .unwrap();
    p.remove("illuminant_model");
    p.remove("diffuse_sky_kelvin");
    // Values from the actual generator-168 cafeteria manifest.
    let original_sky = Vec3::new(1.783_095_7, 1.588_034_5, 1.381_200_8);
    p.insert(
        "sky_radiance".into(),
        serde_json::to_value(original_sky).unwrap(),
    );
    let mut replay: IndoorManifest = serde_json::from_value(archive).unwrap();
    assert_eq!(replay.illuminant_model(), IlluminantModel::LegacySrgb);
    assert_eq!(replay.sky_radiance(), original_sky);
    let old_sun = kelvin_rgb(replay.domain().unwrap().photometry.sun_kelvin);
    assert_eq!(
        architecture::sun_color(&replay),
        super::super::source_color(&replay, old_sun)
    );
    let mounts = architecture::fixture_positions(&scene);
    let mut changed = 0;
    for i in 0..mounts.len() {
        let (new, lumens) = architecture::fixture_photometry_at(&scene, i, &mounts);
        let (old, old_lumens) = architecture::fixture_photometry_at(&replay, i, &mounts);
        assert_eq!(lumens, old_lumens);
        changed += usize::from(lumens > 0. && new != old);
    }
    assert!(changed > 0);
    replay
        .apply_appearance(AppearanceSettings {
            lighting_seed: Some(seed),
            ..default()
        })
        .unwrap();
    assert_eq!(replay.illuminant_model(), IlluminantModel::LegacySrgb);
    assert!(replay.sky_radiance().distance(original_sky) < 1e-6);
    assert_eq!(replay.domain().unwrap().photometry.diffuse_sky_kelvin, None);
}
