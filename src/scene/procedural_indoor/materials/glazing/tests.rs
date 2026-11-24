use super::*;
#[test]
fn treatments_are_metric_bounded_and_replayable() {
    let mut treatments = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let p = GlazingProgram::sample(seed, 0, true);
        p.validate().unwrap();
        treatments.insert(p.treatment);
        assert_eq!(p, GlazingProgram::sample(seed, 0, true));
        assert_eq!(p.band_m, GlazingProgram::sample(seed, 3, true).band_m);
        assert_eq!(
            p.thickness_m,
            GlazingProgram::sample(seed, 1, true).thickness_m
        );
        for y in 0..100 {
            let f = p.frost(Vec2::new(0.13, y as f32 * 0.04));
            assert!((0.0..=1.).contains(&f));
        }
    }
    assert_eq!(treatments.len(), 7);
}
#[test]
fn roughness_pyramid_keeps_energy_and_full_chain() {
    let mut p = GlazingProgram::sample(21, 0, false);
    p.treatment = 2;
    let opening = WindowOpening {
        min: Vec2::ZERO,
        max: Vec2::new(2., 3.),
        columns: 1,
        transoms: Vec::new(),
        transom: 0.,
        operable: false,
        glazing: Some(p.clone()),
    };
    let image = roughness_image(&p, &opening, 0.025, MapFootprint::Opening);
    assert_eq!(image.texture_descriptor.mip_level_count, 8);
    let bytes = image.data.unwrap();
    assert_eq!(
        bytes.len(),
        (0..8).map(|i| 4 * (128usize >> i).pow(2)).sum::<usize>()
    );
    let mean = bytes[..128 * 128 * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| (b[1] as f32 / 255.).powi(4))
        .sum::<f32>()
        / (128. * 128.);
    let tail = (bytes[bytes.len() - 3] as f32 / 255.).powi(4);
    assert!((mean - tail).abs() < 0.001);
}

#[test]
fn optimized_privacy_maps_match_dense_quadrature_at_every_mip() {
    let o = WindowOpening {
        min: Vec2::new(-1.31, 0.54),
        max: Vec2::new(1.93, 2.57),
        columns: 1,
        transoms: vec![],
        transom: 0.,
        operable: false,
        glazing: None,
    };
    let mut p = GlazingProgram::sample(73, 1, false);
    let clear = p.glass_recipe().roughness;
    for treatment in 1..=5 {
        p.treatment = treatment;
        let pixels = roughness_image(&p, &o, clear, MapFootprint::Opening)
            .data
            .unwrap();
        let strip = (treatment != 4).then(|| {
            roughness_image(&p, &o, clear, MapFootprint::VerticalStrip)
                .data
                .unwrap()
        });
        let mut side = 128;
        let mut field: Vec<f32> = (0..side * side)
            .map(|i| {
                let cell = Vec2::new((i % side) as f32, (i / side) as f32);
                let mut sum = 0.;
                for dy in [0.25, 0.75] {
                    for dx in [0.25, 0.75] {
                        let at = o.min + (cell + Vec2::new(dx, dy)) / side as f32 * (o.max - o.min);
                        sum += clear.lerp(p.frost_roughness, p.frost(at)).powi(4);
                    }
                }
                sum * 0.25
            })
            .collect();
        let mut offset = 0;
        let mut strip_offset = 0;
        loop {
            for (index, moment) in field.iter().enumerate() {
                let green = (moment.sqrt().sqrt() * 255.).round() as u8;
                assert_eq!(
                    pixels[offset..offset + 4],
                    [255, green, 0, 255],
                    "treatment {treatment}, mip width {side}"
                );
                if let Some(strip) = &strip {
                    let row = strip_offset + 4 * (index / side);
                    assert_eq!(pixels[offset..offset + 4], strip[row..row + 4]);
                }
                offset += 4;
            }
            strip_offset += 4 * side;
            if side == 1 {
                break;
            }
            let next = side / 2;
            field = (0..next * next)
                .map(|i| {
                    let q = 2 * (i / next) * side + 2 * (i % next);
                    (field[q] + field[q + 1] + field[q + side] + field[q + side + 1]) * 0.25
                })
                .collect();
            side = next;
        }
        assert_eq!(offset, pixels.len());
        if let Some(strip) = strip {
            assert_eq!(strip.len(), 1020);
            assert_eq!(strip_offset, strip.len());
        }
    }
}

#[test]
fn shared_partition_installs_one_finish_and_obeys_both_annotation_policies() {
    use crate::render::glass::AnnotationGlass;
    use crate::scene::procedural_indoor::{
        architecture::interior_glazing::{partition_height, shared_facade, SharedOpenings},
        layout::IndoorLayout,
    };
    let mut scene =
        IndoorManifest::generate_with_humans(32, IndoorLayout::Conference, 0.5, 0, 0.).unwrap();
    for w in &mut scene.envelope.as_mut().unwrap().walls {
        w.facade = None;
    }
    scene.adjoining_elevation = None;
    for treatment in [0, 1, 2, 3, 5, 6] {
        let p = scene
            .program
            .as_mut()
            .unwrap()
            .finishes
            .as_mut()
            .unwrap()
            .interior_glazing
            .as_mut()
            .unwrap();
        p.shared_openings = Some(SharedOpenings {
            jamb_intervals: [Vec2::Y; 2],
            height_interval: Vec2::Y,
        });
        p.opening_pitch_m = 1.8;
        let stock = &mut p.vision_panels.as_mut().unwrap().glass;
        stock.treatment = treatment;
        let identity = stock.identity;
        let thickness = stock.thickness_m;
        let f = shared_facade(&scene, partition_height(&scene));
        assert!(
            f.openings.len() > 1
                && f.openings
                    .iter()
                    .all(|o| o.glazing.as_ref().unwrap().identity == identity)
        );
        for quality in [IndoorQuality::Auto, IndoorQuality::Portable] {
            let mut images = Assets::default();
            let mut materials = Assets::default();
            let map = build(&scene, quality, &mut images, &mut materials);
            assert_eq!(map.len(), 1);
            assert_eq!(images.len(), usize::from((2..=5).contains(&treatment)));
            let mat = materials.get(&map[&identity]).unwrap();
            assert_eq!(mat.thickness, thickness);
            if treatment == 1 {
                let o = &f.openings[0];
                let p = o.glazing.as_ref().unwrap();
                let dense =
                    roughness_image(p, o, p.glass_recipe().roughness, MapFootprint::Opening);
                let value = (mat.perceptual_roughness * 255.).round() as u8;
                assert!(dense.data.unwrap().chunks_exact(4).all(|t| t[1] == value));
            } else if let Some(handle) = &mat.metallic_roughness_texture {
                let image = images.get(handle).unwrap();
                assert_eq!(image.width(), 1);
                assert_eq!(image.height(), 128);
                assert_eq!(image.texture_descriptor.mip_level_count, 8);
            }
            assert!(AnnotationGlass::Surface.includes(Some(mat)));
            assert!(!AnnotationGlass::Through.includes(Some(mat)));
        }
    }
    let p = scene
        .program
        .as_mut()
        .unwrap()
        .finishes
        .as_mut()
        .unwrap()
        .interior_glazing
        .as_mut()
        .unwrap();
    p.shared_openings.as_mut().unwrap().jamb_intervals = [Vec2::ZERO; 2];
    let mut images = Assets::default();
    let mut materials = Assets::default();
    assert!(build(&scene, IndoorQuality::Auto, &mut images, &mut materials).is_empty());
    assert!(images.is_empty() && materials.is_empty());
}
