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
                let assembly = build_object(o);
                audit::check(o, &assembly).unwrap();
                let grid =
                    crate::scene::procedural_indoor::architecture::ceiling::CeilingGrid::for_scene(
                        &scene,
                    );
                let transform = o.transform();
                let mut contact = f32::INFINITY;
                for vertex in assembly.parts.values().flat_map(|g| &g.positions) {
                    let at = transform.transform_point(Vec3::from_array(*vertex));
                    let roof = grid.map_or_else(
                        || scene.ceiling_height(at.xz()),
                        |g| g.mount_height(&scene, at.xz()),
                    );
                    assert!(
                        at.y <= roof + 0.0002,
                        "fan penetrates ceiling in seed {seed}"
                    );
                    if (vertex[1] - o.size.y).abs() < 1e-5 {
                        contact = contact.min((roof - at.y).abs());
                    }
                }
                assert!(
                    contact < 0.0003,
                    "fan floats {contact}m below ceiling in seed {seed}"
                );
                let mut loose = o.clone();
                loose.position.y -= 0.025;
                assert!(!super::super::super::layout::equipment::fan_clear(
                    &scene, &loose
                ));
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
fn reported_fan_headroom_survives_later_rug_placement() {
    use crate::scene::procedural_indoor::layout::equipment::fan_clear;
    use crate::scene::procedural_indoor::layout::floor::area::RugSupport;
    let mut unoccupied_fans = None;
    for people in [0., 0.5, 1.] {
        let mut scene =
            IndoorManifest::generate_with_humans(65_158_972, IndoorLayout::Mixed, 0.65, 0, people)
                .unwrap();
        validation::validate_layout(&scene).unwrap();
        let fans: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::CeilingFan)
            .cloned()
            .collect();
        assert!(
            !fans.is_empty(),
            "the reported room must exercise fan placement"
        );
        assert_eq!(&fans, unoccupied_fans.get_or_insert_with(|| fans.clone()));
        for fan in &fans {
            assert!(fan_clear(&scene, fan));
            audit::check(fan, &build_object(fan)).unwrap();
            // Check the maximum possible rug rise across the full sweep,
            // including rugs entirely between the reservation's corners.
            let (lo, hi) = fan.obstacle_bounds().next().unwrap();
            let floor = scene
                .rug_supports
                .iter()
                .filter(|rug| rug.min.cmplt(hi.xz()).all() && rug.max.cmpgt(lo.xz()).all())
                .map(|rug| rug.top)
                .fold(0., f32::max);
            assert!(lo.y >= floor + 2.1);
        }
        // A rug entirely inside the rotor footprint must affect admission even
        // when none of the footprint's four corners touches it.
        let fan = &fans[0];
        let (lo, hi) = fan.obstacle_bounds().next().unwrap();
        let center = (lo.xz() + hi.xz()) * 0.5;
        let min = center - Vec2::splat(0.10);
        let max = center + Vec2::splat(0.10);
        let threshold = lo.y - 2.1;
        scene.rug_supports.push(RugSupport {
            object: usize::MAX,
            outline: vec![min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)],
            min,
            max,
            bottom: threshold - 0.004,
            top: threshold - 0.001,
        });
        assert!(fan_clear(&scene, fan));
        scene.rug_supports.last_mut().unwrap().top = threshold + 0.001;
        assert!(!fan_clear(&scene, fan));
    }
    for seed in [65_158_970, 65_158_971, 65_158_973, 65_158_974] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        validation::validate_layout(&scene).unwrap();
    }
}

#[test]
fn reported_mezzanine_fan_seed_reserves_upper_storey_headroom() {
    use crate::scene::procedural_indoor::layout::equipment::fan_clear;
    for people in [0., 0.5, 1.] {
        let scene =
            IndoorManifest::generate_with_humans(65_412_811, IndoorLayout::Mixed, 0.65, 0, people)
                .unwrap();
        validation::validate_layout(&scene).unwrap();
        let deck = &scene
            .envelope
            .as_ref()
            .unwrap()
            .mezzanine
            .as_ref()
            .unwrap()
            .deck;
        let fans: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::CeilingFan)
            .collect();
        assert!(
            !fans.is_empty(),
            "retry the placement rather than removing fans"
        );
        for fan in fans {
            assert!(fan_clear(&scene, fan));
            audit::check(fan, &build_object(fan)).unwrap();
            for (lo, hi) in fan.obstacle_bounds() {
                if deck.overlaps(lo.xz(), hi.xz()) {
                    assert!(lo.y >= deck.height + 2.1);
                }
                for rug in &scene.rug_supports {
                    if lo.xz().cmplt(rug.max).all() && hi.xz().cmpgt(rug.min).all() {
                        assert!(lo.y >= rug.top + 2.1);
                    }
                }
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

#[test]
fn installed_rotors_have_balanced_closed_blades_and_bounded_spin() {
    let mut counts = std::collections::BTreeSet::new();
    let mut families = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::CeilingFan, seed);
        o.variant = installed_variant(seed, o.size.x) & !3 | (seed as u32 & 3);
        for (diameter, height) in [(0.85, 0.28), (1.95, 0.28), (1.95, 0.85)] {
            o.size = Vec3::new(diameter, height, diameter);
            let p = parameters(&o);
            let section = p.product.as_ref().unwrap();
            counts.insert(p.blades);
            families.insert(p.style);
            let a = build_object_with_motion(
                &o,
                Some(&IndoorMotionConfig {
                    fan_fraction: 1.,
                    ..default()
                }),
            );
            audit::check(&o, &a).unwrap();
            assert_eq!(a.joints.len(), 1);
            let rotor = &a.joints[0];
            let mut blades = Geometry::default();
            blade::build(&mut blades, &o, &p, section);
            let actual = &a.parts[rotor
                .parts
                .iter()
                .find(|(surface, _)| *surface == p.blade_surface)
                .unwrap()];
            assert!(actual.positions.ends_with(&blades.positions));
            let per = blades.positions.len() / usize::from(p.blades);
            assert_eq!(per * usize::from(p.blades), blades.positions.len());
            let blank = &blades.positions[..per];
            // Either face has a metric V chart from the hub to the tip. A cut
            // across the width changes U only.
            let uv = &blades.uvs;
            assert!((uv[0][1] - uv[4][1]).abs() < 1e-5);
            assert!((uv[35][1] - uv[0][1] - (p.radius_m - p.hub_radius * 0.76)).abs() < 1e-5);
            assert!(blades.positions.iter().all(|v| {
                let v = Vec3::from_array(*v);
                v.y >= 0. && v.y < p.rotor_top && v.xz().length() <= p.radius_m + 1e-6
            }));
            assert!(blank
                .iter()
                .any(|v| Vec2::new(v[0], v[2]).length() > p.radius_m - 1e-5));
            for i in 1..p.blades {
                let yaw = Quat::from_rotation_y(TAU * f32::from(i) / f32::from(p.blades));
                let start = usize::from(i) * per;
                for (a, b) in blank.iter().zip(&blades.positions[start..]) {
                    assert!((yaw * Vec3::from_array(*a)).distance(Vec3::from_array(*b)) < 1e-5);
                }
                assert_ne!(blades.uvs[0], blades.uvs[start]);
            }
            let (mut up, mut down, mut side) = (0, 0, 0);
            for tri in blades.indices.chunks_exact(3) {
                let [i, j, k] = [tri[0], tri[1], tri[2]].map(|v| v as usize);
                let [x, y, z] = [i, j, k].map(|idx| Vec3::from_array(blades.positions[idx]));
                let cross = (y - x).cross(z - x);
                assert!(cross.length_squared() > 1e-18);
                let n = cross.normalize();
                assert!([i, j, k]
                    .into_iter()
                    .all(|idx| n.dot(Vec3::from_array(blades.normals[idx])) > 0.35));
                if n.y > 0.65 {
                    up += 1
                } else if n.y < -0.65 {
                    down += 1
                } else {
                    side += 1
                }
            }
            assert!(up > 0 && down > 0 && side > 0 && section.tip_pitch_radians < p.pitch_radians);
            let reserves = local_reservations(&o);
            for t in [0., 0.17, 0.58, 1.] {
                let tf = rotor.transform(t);
                for (key, g) in &a.parts {
                    let moving = rotor.parts.contains(key);
                    for v in &g.positions {
                        let v = Vec3::from_array(*v);
                        let v = if moving { tf.transform_point(v) } else { v };
                        assert!(
                            reserves
                                .iter()
                                .any(|(lo, hi)| v.cmpge(*lo - 1e-6).all()
                                    && v.cmple(*hi + 1e-6).all())
                        );
                    }
                }
            }
        }
    }
    assert_eq!(families.len(), 4);
    assert_eq!(counts, (2..=9).collect());
}

#[test]
fn matching_products_keep_the_same_tooling_and_independent_rotor_phase() {
    let a = audit::fixture(ObjectKind::CeilingFan, 17);
    let mut b = a.clone();
    b.seed = 492;
    b.size.y = 0.28;
    let p = parameters(&a);
    let mut q = parameters(&b);
    assert_ne!(p.phase, q.phase);
    q.phase = p.phase;
    assert_eq!(p, q);
    b.finish_seed = Some(37);
    let other = parameters(&b);
    assert_eq!(other.product, p.product);
    assert_eq!(other.blades, p.blades);
    assert_eq!(other.radius_m, p.radius_m);
}
