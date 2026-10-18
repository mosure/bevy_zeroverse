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
            let s = super::super::fixture_size(&scene);
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
            let area = emitting_area(&scene);
            assert!(area.is_finite() && area > 0.0001 && area <= size.element_product());
            assert!(local
                .parts
                .keys()
                .all(|(_, label)| label.starts_with("lamp#0")));
        }
    }
    assert_eq!(shapes.len(), 6);
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
