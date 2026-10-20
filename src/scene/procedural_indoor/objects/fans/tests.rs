use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorLayout, IndoorManifest},
    motion::{IndoorMotionConfig, MotionKind},
    validation,
};

#[test]
fn ceiling_fan_geometry_is_replayable_and_contained_at_every_angle() {
    let mut counts = std::collections::BTreeSet::new();
    let mut surfaces = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::CeilingFan, seed);
        o.variant = seed as u32 % 4;
        for (diameter, height) in [(0.85, 0.28), (1.95, 0.85)] {
            o.size = Vec3::new(diameter, height, diameter);
            let p = parameters(&o);
            assert_eq!(p, parameters(&o));
            counts.insert(p.blades);
            surfaces.insert(p.blade_surface);
            let policy = IndoorMotionConfig {
                fan_fraction: 1.,
                fan_turns: [2.25, 2.25],
                ..default()
            };
            let a = build_object_with_motion(&o, Some(&policy));
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a.joints.len(), 1);
            let j = &a.joints[0];
            assert_eq!(j.kind, MotionKind::Fan);
            for i in 0..=64 {
                let tf = j.transform(j.amount(i as f32 / 64.));
                for key in &j.parts {
                    for p in &a.parts[key].positions {
                        let v = tf.transform_point(Vec3::from_array(*p));
                        assert!(v.xz().length() <= diameter * 0.5 && v.y >= 0. && v.y <= height);
                    }
                }
            }
            let static_a = build_object(&o);
            assert!(static_a.joints.is_empty());
            let reserves = local_reservations(&o);
            assert!(static_a.parts.values().flat_map(|g| &g.positions).all(|v| {
                let p = Vec3::from_array(*v);
                reserves.iter().any(|(a, b)| {
                    p.cmpge(*a - Vec3::splat(1e-6)).all() && p.cmple(*b + Vec3::splat(1e-6)).all()
                })
            }));
            let vertices: usize = static_a.parts.values().map(|g| g.positions.len()).sum();
            assert_eq!(
                vertices,
                a.parts.values().map(|g| g.positions.len()).sum::<usize>()
            );
        }
    }
    assert!(counts.len() >= 7 && surfaces.len() == 4);
}

#[test]
fn ceiling_fans_spawn_with_full_rotor_clearance_in_diverse_rooms() {
    let mut fans = 0;
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validation::validate_layout(&scene).unwrap();
        for o in &scene.objects {
            if o.kind == ObjectKind::CeilingFan {
                fans += 1;
                assert!(super::super::super::layout::equipment::fan_clear(&scene, o));
                audit::check(o, &build_object(o)).unwrap();
            }
        }
    }
    assert!(fans >= 16, "only {fans} fans across 128 rooms");
}

#[test]
fn adjacent_fan_clearance_and_identity_survive_population_changes() {
    // These rooms caught asymmetric full-box/split-volume fan reservations and
    // occupancy-dependent fixture IDs in the broader layout distribution audit.
    for seed in [16, 407, 577, 605, 1014] {
        for layout in [IndoorLayout::Mixed, IndoorLayout::Conference] {
            for density in [0., 0.65, 1.] {
                let empty =
                    IndoorManifest::generate_with_humans(seed, layout, density, 0, 0.).unwrap();
                let populated =
                    IndoorManifest::generate_with_humans(seed, layout, density, 0, 1.).unwrap();
                for scene in [&empty, &populated] {
                    validation::validate_layout(scene)
                        .unwrap_or_else(|e| panic!("{layout:?}, density {density}: {e}"));
                }
                let fans = |scene: &IndoorManifest| {
                    scene
                        .objects
                        .iter()
                        .filter(|o| o.kind == ObjectKind::CeilingFan)
                        .cloned()
                        .collect::<Vec<_>>()
                };
                assert_eq!(fans(&empty), fans(&populated));
            }
        }
    }
}

#[test]
fn ceiling_fan_motion_is_opt_in_and_zero_speed_is_static() {
    let o = audit::fixture(ObjectKind::CeilingFan, 5);
    for policy in [
        None,
        Some(IndoorMotionConfig {
            fan_fraction: 0.,
            ..default()
        }),
        Some(IndoorMotionConfig {
            fan_fraction: 1.,
            fan_turns: [0., 0.],
            ..default()
        }),
    ] {
        assert!(build_object_with_motion(&o, policy.as_ref())
            .joints
            .is_empty());
    }
}

#[test]
fn fan_rates_and_directions_are_seeded_continuous_and_manifest_sampling_is_exact() {
    use crate::scene::procedural_indoor::{
        motion::{prepare, IndoorMotionReport},
        preparation::SceneGeometry,
    };
    let mut angles = std::collections::BTreeSet::new();
    for seed in 0..32 {
        let o = audit::fixture(ObjectKind::CeilingFan, seed);
        let policy = IndoorMotionConfig {
            fan_fraction: 1.,
            ..default()
        };
        let a = build_object_with_motion(&o, Some(&policy));
        let angle = a.joints[0].angle;
        angles.insert(angle.to_bits());
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        scene.humans.clear();
        scene.indoor_motion = Some(IndoorMotionReport::new(policy));
        let mut g = SceneGeometry {
            architecture: a,
            objects: vec![],
            humans: vec![],
        };
        prepare(&mut scene, &mut g);
        let report = &scene.indoor_motion.as_ref().unwrap().mechanisms[0];
        assert!(report.rejection.is_none(), "{report:?}");
        for t in [0., 0.137, 0.8, 1., 0.137, 0.] {
            assert_eq!(
                report.rigid_transform_at(t).unwrap(),
                g.architecture.joints[0].transform(t)
            );
        }
    }
    assert_eq!(angles.len(), 32);
    assert!(
        angles.iter().any(|&a| f32::from_bits(a) > 0.)
            && angles.iter().any(|&a| f32::from_bits(a) < 0.)
    );
}

#[test]
#[ignore = "local CPU geometry contact sheet, not a PBR render"]
fn ceiling_fan_geometry_preview() {
    let mut sheet = image::RgbImage::new(4 * 256, 2 * 256);
    for i in 0..8 {
        let mut o = audit::fixture(ObjectKind::CeilingFan, 17 + i);
        o.variant = i as u32 % 4;
        o.size.y = 0.30 + i as f32 * 0.065;
        let a = build_object(&o);
        let image = crate::scene::procedural_indoor::geometry::preview::orthographic(
            &a,
            Vec3::Y * o.size.y * 0.5,
            1.7,
            256,
            Vec3::new(0.55, 0.6, 1.),
        );
        image::imageops::replace(
            &mut sheet,
            &image,
            (i % 4) as i64 * 256,
            (i / 4) as i64 * 256,
        );
    }
    std::fs::create_dir_all("out/object_motion_review").unwrap();
    sheet
        .save("out/object_motion_review/fans-geometry.png")
        .unwrap();
}
