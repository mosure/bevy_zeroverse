use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};

#[test]
fn realized_cups_and_troffers_have_recessed_emitters_and_flux_normalized_beams() {
    let mut scene =
        IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let position = Vec3::new(0.73, 2.31, -1.45);
    let mut counts = [0usize; 3];
    for seed in 0..192 {
        scene.seed = seed;
        let p = parameters_at(&scene, position);
        let insert = optical_insert(&scene, &p);
        let s = size(&scene, position);
        let (inner, outer) = beam_angles(&scene, 0, position);
        assert!(inner > 0. && inner < outer && outer < std::f32::consts::FRAC_PI_2);
        // Integrate the actual squared cosine falloff over solid angle.
        let peak = super::super::spot_intensity_for_lumens(2300., inner, outer)
            / (4. * std::f32::consts::PI);
        let flux: f32 = (0..1024)
            .map(|i| {
                let cos = (i as f32 + 0.5) / 1024.;
                peak * ((cos - outer.cos()) / (inner.cos() - outer.cos()))
                    .clamp(0., 1.)
                    .powi(2)
            })
            .sum::<f32>()
            * std::f32::consts::TAU
            / 1024.;
        assert!((flux / 2300. - 1.).abs() < 0.0001);
        let (expected_y, expected_area) = match insert {
            OpticalInsert::Basket { width, recess, .. } => {
                counts[0] += 1;
                (
                    s.y * (recess - 0.5),
                    s.x * s.z * (1. - 2. * p.border_fraction).powi(2) * width,
                )
            }
            OpticalInsert::Cup { throat, recess, .. } => {
                counts[1] += 1;
                let (count, _, r) = cells(s);
                let rim = r * p.opening_fraction;
                let depth = optics::cup_depth(s.y, rim, recess);
                assert!(outer <= ((1. + throat) * rim).atan2(depth) + 1e-6);
                (
                    depth - s.y * 0.5,
                    count as f32
                        * 16.
                        * (std::f32::consts::TAU / 32.).sin()
                        * (rim * throat).powi(2),
                )
            }
            _ => {
                counts[2] += 1;
                continue;
            }
        };
        let a = geometry(&scene, 0, position);
        let g = &a.parts[&(Surface::Light, "lamp#0".into())];
        let inv = rotation(&scene, position).inverse();
        let mut area = 0.;
        for t in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = [t[0], t[1], t[2]]
                .map(|i| inv * (Vec3::from_array(g.positions[i as usize]) - position));
            assert!([a, b, c].iter().all(|p| (p.y - expected_y).abs() < 3e-6));
            area += (b - a).cross(c - a).dot(Vec3::NEG_Y) * 0.5;
        }
        assert!((area / expected_area - 1.).abs() < 0.0001);
        assert!((area / emitting_area_at(&scene, position) - 1.).abs() < 0.0001);
    }
    assert!(counts.iter().all(|&n| n >= 10), "{counts:?}");
    scene.generator_version = 127;
    assert_eq!(
        optical_insert(&scene, &parameters(&scene)),
        OpticalInsert::Plain
    );
    let mut r = stream(scene.seed, 192);
    assert_eq!(
        beam_angles(&scene, 0, position),
        (r.random_range(0.48..0.90), r.random_range(1.02..1.40))
    );
}

#[test]
fn ring_apertures_cover_broad_bands_and_thin_hoops_without_clamp_piles() {
    let mut scene =
        IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut fractions = Vec::new();
    for seed in 0..768 {
        scene.seed = seed;
        if parameters(&scene).shape != FixtureShape::Ring {
            continue;
        }
        for x in [-1.3, 0.25, 1.8] {
            let p = parameters_at(&scene, Vec3::new(x, 2.8, 0.7));
            assert!(p.opening_fraction > 0.24 && p.opening_fraction < 0.94);
            fractions.push(p.opening_fraction);
        }
    }
    fractions.sort_by(f32::total_cmp);
    let lo = fractions[fractions.len() / 20];
    let hi = fractions[fractions.len() * 19 / 20];
    assert!(
        fractions.len() > 200 && lo < 0.40 && hi > 0.82,
        "{lo}..{hi}"
    );
    println!(
        "ring aperture p05={lo} p95={hi}; {} fixtures",
        fractions.len()
    );
}

#[test]
fn expanded_ceiling_rooms_keep_clear_multiview_trajectories() {
    for seed in 0..16 {
        let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        crate::scene::procedural_indoor::validation::validate_layout(&scene).unwrap();
        assert_eq!(scene.cameras.len(), 3);
        for camera in &scene.cameras {
            assert!(scene.camera_curve_clear(camera));
            for step in 0..=32 {
                assert!(scene.camera_clear(camera.transform_at(step as f32 / 32.).translation));
            }
        }
    }
}

#[test]
fn fixture_programs_fit_reservations_and_have_valid_semantic_geometry() {
    let mut scene =
        IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut shapes = std::collections::BTreeSet::new();
    for seed in 0..96 {
        scene.seed = seed;
        shapes.insert(format!("{:?}", parameters(&scene).shape));
        for size in [
            Vec2::new(0.3, 0.10),
            Vec2::new(1.6, 0.10),
            Vec2::new(0.3, 0.65),
            Vec2::new(1.6, 0.65),
        ] {
            scene.program.as_mut().unwrap().fixture_size = size;
            let position = super::super::fixture_positions(&scene)[0];
            let a = geometry(&scene, 0, position);
            let s = super::super::fixture_half_extents(&scene, position) * 2.;
            let mut o = audit::fixture(ObjectKind::CeilingPendant, seed);
            let (_, hi) = a.bounds();
            o.size = Vec3::new(
                s.x + 0.006,
                hi.y - (position.y - s.y * 0.5) + 0.003,
                s.z + 0.006,
            );
            let origin = position - Vec3::Y * s.y * 0.5;
            let mut local = a;
            for g in local.parts.values_mut() {
                for p in &mut g.positions {
                    *p = (Vec3::from_array(*p) - origin).to_array();
                }
            }
            let checked = audit::check(&o, &local).unwrap_or_else(|e| {
                panic!("seed={seed}, {:?} {size:?}: {e}", parameters(&scene).shape)
            });
            assert!(checked.triangles < 10_000);
            let axis = direction(&scene, position);
            let local_emitter =
                rotation(&scene, position).inverse() * (emitter(&scene, position) - position);
            assert!(
                (local_emitter - Vec3::NEG_Y * (super::size(&scene, position).y * 0.5 + 0.006))
                    .length()
                    < 1e-5
            );
            let area = emitting_area_at(&scene, position);
            let realized = super::size(&scene, position);
            assert!(area.is_finite() && area > 0.0001 && area <= realized.x * realized.z);
            let projected_area: f32 = local
                .parts
                .iter()
                .filter(|((surface, _), _)| *surface == Surface::Light)
                .map(|(_, g)| {
                    g.indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|t| {
                            let [a, b, c] = [t[0], t[1], t[2]]
                                .map(|i| Vec3::from_array(g.positions[i as usize]));
                            (b - a).cross(c - a).dot(axis).max(0.) * 0.5
                        })
                        .sum::<f32>()
                })
                .sum();
            assert!(
                (projected_area / area - 1.).abs() < 0.035,
                "seed={seed} {:?}: mesh area={projected_area} photometry area={area}",
                parameters(&scene).shape
            );
            assert!(local
                .parts
                .keys()
                .all(|(_, label)| label.starts_with("lamp#0")));
        }
    }
    assert_eq!(shapes.len(), 6);
}

#[test]
fn fixture_sizes_vary_within_a_room_and_ignore_mount_height() {
    let scene = IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::ZERO;
    for i in 0..128 {
        let p = Vec3::new(i as f32 * 0.037 - 2., 0., (i % 13) as f32 * 0.19 - 1.);
        let a = size(&scene, p);
        assert_eq!(a, size(&scene, p.with_y(2.8)));
        min = min.min(a);
        max = max.max(a);
    }
    assert!((max / min).cmpgt(Vec3::new(1.8, 1.4, 1.8)).all());
    println!(
        "fixture size sampler audit: {}",
        serde_json::json!({"min_m":min.to_array(),"max_m":max.to_array()})
    );
}

#[test]
fn lamp_shapes_and_materials_diversify_without_changing_placement_envelopes() {
    use crate::scene::procedural_indoor::objects::{self, luminaires};
    let mut heads = std::collections::BTreeSet::new();
    let mut bulbs = std::collections::BTreeSet::new();
    let mut materials = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let p = luminaires::parameters(seed);
        heads.insert(p.shape);
        bulbs.insert(p.bulb);
        materials.insert(format!("{:?}", p.finish));
        for kind in [
            ObjectKind::FloorLamp,
            ObjectKind::DeskLamp,
            ObjectKind::CeilingPendant,
        ] {
            let o = audit::fixture(kind, seed);
            let a = objects::build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
            let b = objects::build_object(&o);
            for (key, g) in &a.parts {
                assert_eq!(g.positions, b.parts[key].positions);
                assert_eq!(g.indices, b.parts[key].indices);
            }
        }
    }
    assert_eq!((heads.len(), bulbs.len(), materials.len()), (5, 4, 4));
}
