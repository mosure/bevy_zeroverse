use super::*;
use crate::scene::procedural_indoor::{layout::IndoorLayout, validation};

#[test]
fn doors_cover_mechanisms_states_hardware_and_materials() {
    let mut closed = 0;
    let mut ajar = 0;
    let mut open = 0;
    let mut hardware = std::collections::HashSet::new();
    for mechanism in DoorMechanism::ALL {
        for seed in 0..64 {
            let p = DoorProgram::sample(seed, mechanism);
            assert!(p.validate());
            assert_eq!(p, DoorProgram::sample(seed, mechanism));
            hardware.insert(format!("{:?}/{:?}", p.handle, p.hardware_finish));
            if p.opening == 0. {
                closed += 1;
            } else if p.opening < 0.85 {
                ajar += 1;
            } else {
                open += 1;
            }
            let d = DoorInstallation {
                id: 0,
                center: Vec3::new(0.3, 0., -0.8),
                yaw: 0.37,
                width: if mechanism == DoorMechanism::Revolving {
                    2.5
                } else {
                    1.4
                },
                height: 2.25,
                wall_depth: 0.18,
                program: p,
            };
            let mut a = Assembly::default();
            d.build(&mut a, true);
            for ((_, label), g) in a.parts {
                assert!(crate::render::semantic::SemanticLabel::from_label(
                    label.split('#').next().unwrap()
                )
                .is_some());
                assert!(g.positions.iter().flatten().all(|v| v.is_finite()));
                assert!(g.normals.iter().flatten().all(|v| v.is_finite()));
                for t in g.indices.as_chunks::<3>().0 {
                    let p = t
                        .iter()
                        .map(|&i| Vec3::from_array(g.positions[i as usize]))
                        .collect::<Vec<_>>();
                    assert!(
                        (p[1] - p[0]).cross(p[2] - p[0]).length_squared() > 1e-17,
                        "{mechanism:?} {seed} degenerate triangle"
                    );
                }
            }
            let colliders = d.obstacles();
            for leaf in d.leaves() {
                for x in [-0.49, 0., 0.49] {
                    let point = leaf.transform.transform_point(Vec3::X * leaf.size.x * x);
                    assert!(colliders
                        .iter()
                        .any(|(a, b)| point.cmpge(*a).all() && point.cmple(*b).all()));
                }
            }
        }
    }
    assert!(closed > 50 && ajar > 50 && open > 50 && hardware.len() >= 10);
}

#[test]
fn doors_are_real_scene_geometry_with_reserved_approaches() {
    for layout in [
        IndoorLayout::Mixed,
        IndoorLayout::Kitchen,
        IndoorLayout::ElevatorLanding,
    ] {
        for seed in 0..48 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            let d = scene.entry_door.as_ref().unwrap();
            assert!(d.program.validate());
            assert_eq!(d.center.x, scene.door_x);
            assert!(!scene.objects.iter().filter(|o| o.solid).any(|o| {
                let (a, b) = o.bounds();
                scene.entry_approach_overlaps(a, b)
            }));
            let a = super::super::architecture(&scene);
            assert!(a.parts.keys().any(|(_, s)| s.starts_with("door#portal0")));
            assert!(a
                .parts
                .keys()
                .any(|(_, s)| s.starts_with("door_hardware#portal0")));
            validation::validate_geometry(&scene).unwrap();
        }
    }
}

#[test]
fn closed_doors_block_navigation_and_open_hinges_clear_the_center() {
    let mut d = DoorInstallation {
        id: 0,
        center: Vec3::ZERO,
        yaw: 0.,
        width: 1.2,
        height: 2.25,
        wall_depth: 0.18,
        program: DoorProgram::sample(1, DoorMechanism::Hinged),
    };
    d.program.opening = 0.;
    let a = Vec3::new(0., 1., -1.8);
    let b = Vec3::new(0., 1., 1.8);
    assert!(d
        .obstacles()
        .into_iter()
        .any(|(lo, hi)| crate::scene::procedural_indoor::layout::segment_hits_box(a, b, lo, hi)));
    d.program.opening = 1.;
    assert!(!d
        .obstacles()
        .into_iter()
        .any(|(lo, hi)| crate::scene::procedural_indoor::layout::segment_hits_box(a, b, lo, hi)));
}

#[test]
fn cabinet_fronts_preserve_bounds_and_real_counter_support() {
    use crate::scene::procedural_indoor::layout::ObjectKind;
    use crate::scene::procedural_indoor::objects::{self, audit};
    for kind in [
        ObjectKind::Cabinet,
        ObjectKind::KitchenCounter,
        ObjectKind::KitchenCupboard,
        ObjectKind::Pantry,
        ObjectKind::KitchenSink,
    ] {
        for seed in 0..128 {
            let o = audit::fixture(kind, seed);
            let a = objects::build_object(&o);
            audit::check(&o, &a).unwrap();
            if kind == ObjectKind::KitchenCounter {
                let (lo, hi) = objects::kitchen::counter_top_bounds(&o);
                let top = a
                    .parts
                    .iter()
                    .filter(|((_, l), _)| objects::part_label(l) == "counter")
                    .flat_map(|(_, g)| g.positions.iter())
                    .map(|p| Vec3::from_array(*p));
                let (a, b) = top.fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(lo, hi), p| (lo.min(p), hi.max(p)),
                );
                assert!(
                    lo.x >= a.x - 0.001
                        && lo.y >= a.z - 0.001
                        && hi.x <= b.x + 0.001
                        && hi.y <= b.z + 0.001,
                    "support extends past worktop seed {seed}"
                );
            }
        }
    }
}

#[test]
fn resized_counters_reserve_realized_front_travel() {
    use crate::scene::procedural_indoor::layout::ObjectKind;
    use crate::scene::procedural_indoor::objects::{self, audit, kitchen};
    for seed in 0..128 {
        for scale in [0.65, 1.35] {
            let mut o = audit::fixture(ObjectKind::KitchenCounter, seed);
            let width = o.size.x * scale;
            kitchen::resize_width(&mut o, width);
            audit::check(&o, &objects::build_object(&o)).unwrap();
        }
    }
    let scene =
        IndoorManifest::generate_with_humans(27, IndoorLayout::Kitchenette, 1., 3, 0.).unwrap();
    validation::validate_layout(&scene).unwrap();
    validation::validate_geometry(&scene).unwrap();
}
