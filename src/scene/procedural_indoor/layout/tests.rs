use super::*;
use crate::scene::procedural_indoor::validation::{validate_geometry, validate_layout};

#[test]
fn reported_sunken_room_props_remain_on_their_supports() {
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.25).unwrap();
        for prop in scene.objects.iter().filter(|o| o.support.is_some()) {
            let support = &scene.objects[prop.support.unwrap()];
            assert!(
                (prop.position.y - support.position.y - support.size.y).abs() < 1e-5,
                "seed {seed}: {:?} {} at {} on {:?} {} at {} + {}",
                prop.kind,
                prop.id,
                prop.position.y,
                support.kind,
                support.id,
                support.position.y,
                support.size.y
            );
        }
        validate_layout(&scene).unwrap();
        validate_geometry(&scene).unwrap();
    }
}

#[test]
fn supported_props_keep_exact_heights_across_world_zero() {
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.program = None; // Exercise placement without probabilistic clutter thinning.
    let mut support = scene.objects[0].clone();
    support.id = 0;
    support.kind = ObjectKind::Cabinet;
    support.size = Vec3::new(1., 0.42, 1.);
    support.yaw = 0.7;
    support.support = None;
    support.neighbor = false;
    support.interaction_target = None;
    for top in [
        -0.001, -0.0001, -0.00005, 0., 0.00005, 0.0001, 0.001, 0.42, 1.5,
    ] {
        support.position = Vec3::new(0., top - support.size.y, 0.);
        scene.envelope.as_mut().unwrap().floor_patches = vec![super::super::envelope::FloorPatch {
            min: Vec2::splat(-2.),
            max: Vec2::splat(2.),
            height: support.position.y,
        }];
        scene.objects = vec![support.clone()];
        let offset = Vec3::new(0.1, 0., -0.1);
        scene.prop(
            &support,
            ObjectKind::Mug,
            offset,
            Vec3::new(0.12, 0.1, 0.09),
            0.2,
            &mut stream(0, 99),
        );
        assert_eq!(scene.objects.len(), 2, "top={top}: prop must be retained");
        let prop = &scene.objects[1];
        let expected = support
            .transform()
            .transform_point(offset + Vec3::Y * support.size.y);
        assert_eq!(
            prop.position, expected,
            "top={top}: support contact must be exact"
        );
        assert_eq!(prop.support, Some(support.id));
    }
}

#[test]
fn reported_sunken_room_seeds_cover_density_and_activity_variants() {
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        for layout in IndoorLayout::PROFILES {
            for density in [0.15, 0.65, 1.] {
                let scene =
                    IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25).unwrap();
                validate_layout(&scene).unwrap();
                validate_geometry(&scene).unwrap_or_else(|e| {
                    panic!("seed={seed},layout={layout:?},density={density}: {e}")
                });
            }
        }
    }
}
