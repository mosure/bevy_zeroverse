use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, objects::audit};

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
            let (lo, hi) = a.bounds();
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
            assert!(emitter(&scene, position).y < lo.y);
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
                            (-(b - a).cross(c - a).y).max(0.) * 0.5
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
