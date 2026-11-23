use super::*;
use crate::scene::procedural_indoor::{humans::appearance::FaceProgram, layout::IndoorLayout};

#[test]
fn seeded_iris_has_a_dark_pupil_rim_and_stroma_with_bounded_light_adaptation() {
    let mut colors = std::collections::HashSet::new();
    let (mut warm, mut cool) = (0, 0);
    for seed in 0..256 {
        let face = FaceProgram::sample(seed);
        let optics = face.optics.unwrap();
        let pupil = optics.adapted_pupil(200.);
        assert_eq!(pupil, optics.pupil_fraction);
        assert!(optics.adapted_pupil(5.) > pupil && optics.adapted_pupil(1200.) < pupil);
        let chart = PigmentChart::new(seed, face.iris, pupil);
        let pupil = chart.color(Vec2::ZERO);
        let rim = chart.color(Vec2::X * 0.986);
        let iris = chart.color(Vec2::X * 0.72);
        let white = chart.color(Vec2::ONE);
        assert!(pupil.max_element() < iris.max_element() && rim.length() < iris.length());
        assert!(white.min_element() > iris.max_element());
        assert!(chart.color(Vec2::Y * 0.72).distance(iris) > 1e-6);
        colors.insert(face.iris.map(f32::to_bits));
        warm += usize::from(face.iris[0] > face.iris[2] * 1.3);
        cool += usize::from(face.iris[2] > face.iris[0]);
    }
    assert_eq!(colors.len(), 256);
    assert!(warm > 125 && cool > 10);
}

#[test]
fn eye_map_is_opaque_reproducible_and_clamps_to_sclera_at_every_mip() {
    let face = FaceProgram::sample(73);
    let chart = PigmentChart::new(73, face.iris, 0.50);
    let image = chart.image();
    let data = image.data.as_ref().unwrap();
    assert_eq!(*data, chart.image().data.unwrap());
    assert_eq!(image.texture_descriptor.mip_level_count, 7);
    assert_eq!(data.len(), 21844);
    assert!(data.as_chunks::<4>().0.iter().all(|rgba| rgba[3] == 255));
    let ImageSampler::Descriptor(sampler) = &image.sampler else {
        panic!("missing clamp sampler")
    };
    assert_eq!(
        (sampler.address_mode_u, sampler.address_mode_v),
        (ImageAddressMode::ClampToEdge, ImageAddressMode::ClampToEdge)
    );
    let corner = &data[..3];
    for side in [64usize, 32, 16, 8] {
        let offset = (side * 2..=SIZE as usize)
            .filter(|i| i.is_power_of_two())
            .map(|s| s * s * 4)
            .sum::<usize>();
        for c in 0..3 {
            assert!(corner[c].abs_diff(data[offset + c]) <= 3);
        }
    }
    // At a distance the dark centre contributes to the area average.
    assert!(data[data.len() - 4] < corner[0]);
}

#[test]
fn only_current_actors_allocate_an_eye_image() {
    let mut scene =
        IndoorManifest::generate_with_humans(44, IndoorLayout::Mixed, 0.65, 0, 0.8).unwrap();
    let mut images = Assets::<Image>::default();
    let first = scene.humans.first_mut().expect("occupied fixture");
    first.appearance.as_mut().unwrap().face.optics = None;
    assert_eq!(install(&scene, &mut images).len(), scene.humans.len() - 1);
    assert_eq!(images.len(), scene.humans.len() - 1);
    scene.humans.clear();
    assert!(install(&scene, &mut images).is_empty());
}

#[test]
fn ocular_pigment_follows_room_light_and_is_independent_of_camera_exposure_bias() {
    let mut scene =
        IndoorManifest::generate_with_humans(44, IndoorLayout::Mixed, 0.65, 0, 0.8).unwrap();
    scene.humans.truncate(1);
    let render = |scene: &IndoorManifest| {
        let mut images = Assets::<Image>::default();
        let handles = install(scene, &mut images);
        images
            .get(&handles[&scene.humans[0].id])
            .unwrap()
            .data
            .clone()
            .unwrap()
    };
    let meter = |scene: &mut IndoorManifest, daylight, bias| {
        let meter = scene
            .program
            .as_mut()
            .unwrap()
            .domain
            .as_mut()
            .unwrap()
            .photometry
            .meter
            .as_mut()
            .unwrap();
        meter.electric_lux = 5.;
        meter.direct_daylight_lux = daylight;
        meter.diffuse_daylight_lux = 0.;
        meter.bias_ev = bias;
    };
    meter(&mut scene, 0., 0.);
    let dark = render(&scene);
    meter(&mut scene, 900., 0.);
    let daylight = render(&scene);
    let pupil_pixels = |map: &[u8]| {
        map[..(SIZE * SIZE * 4) as usize]
            .chunks_exact(4)
            .filter(|p| p[0] <= 12 && p[1] <= 10 && p[2] <= 9)
            .count()
    };
    assert!(pupil_pixels(&dark) > 2 * pupil_pixels(&daylight));
    meter(&mut scene, 900., -1.5);
    assert_eq!(daylight, render(&scene));
}
