use super::*;
use crate::scene::procedural_indoor::motion::{self, IndoorMotionConfig};

fn installation(seed: u64, mechanism: DoorMechanism) -> DoorInstallation {
    let mut r = stream(seed, 7721);
    DoorInstallation {
        id: 4,
        center: Vec3::new(-3.25, 0.18, 2.4),
        yaw: r.random_range(-PI..PI),
        width: r.random_range(0.92..2.15),
        height: r.random_range(2.06..2.35),
        wall_depth: r.random_range(0.08..0.31),
        program: DoorProgram::sample(seed, mechanism),
    }
}

// Independent triangle intersections in world space, including non-axis-aligned portals.
fn intersects(
    a: &Assembly,
    origin: Vec3,
    ray: Vec3,
    distance: std::ops::RangeInclusive<f32>,
) -> bool {
    a.parts.values().any(|g| {
        g.indices.as_chunks::<3>().0.iter().any(|indices| {
            let [p, q, r] = indices.map(|i| Vec3::from_array(g.positions[i as usize]));
            let u = q - p;
            let v = r - p;
            let h = ray.cross(v);
            let det = u.dot(h);
            if det.abs() < 1e-11 {
                return false;
            }
            let delta = origin - p;
            let i = delta.dot(h) / det;
            let k = delta.cross(u);
            let j = ray.dot(k) / det;
            i >= -0.0001 && j >= -0.0001 && i + j <= 1.0001 && distance.contains(&(v.dot(k) / det))
        })
    })
}

fn crosses(a: &Assembly, at: Vec3, tolerance: f32) -> bool {
    intersects(a, at, Vec3::Y, -tolerance..=tolerance)
}

#[test]
fn opaque_sliders_close_the_jambs_and_meeting_stiles_and_clear_the_parked_aperture() {
    for mechanism in [DoorMechanism::Sliding, DoorMechanism::Telescoping] {
        for seed in 0..20 {
            let mut d = installation(seed, mechanism);
            d.program.panel = DoorPanel::Flush;
            let tf = d.transform();
            let sign = d.program.swing_side;
            for opening in [0., 1.] {
                d.program.opening = opening;
                let mut a = Assembly::default();
                d.build(&mut a, false);
                for sample in 0..=64 {
                    let x = (d.width - 0.038) * (sample as f32 / 64. - 0.5);
                    let origin = tf.transform_point(Vec3::new(x, 1.35, sign * d.wall_depth * 0.5));
                    // An oblique ray through a fully open aperture can see the parked stack.
                    let slopes: &[f32] = if opening == 0. {
                        &[-0.57735, 0., 0.57735]
                    } else {
                        &[0.]
                    };
                    for &slope in slopes {
                        let ray = tf.rotation * Vec3::new(slope, 0., sign).normalize();
                        assert_eq!(intersects(&a, origin, ray, 0.0..=0.5), opening == 0.,
                            "{mechanism:?} seed={seed} opening={opening} sample={sample} slope={slope}");
                    }
                }
            }
        }
    }
}

#[test]
fn wheel_treads_meet_both_ledges_and_hangers_pass_through_the_slot_across_travel() {
    let mut covers = [0; 2];
    for mechanism in [DoorMechanism::Sliding, DoorMechanism::Telescoping] {
        for seed in 0..32 {
            let mut d = installation(seed, mechanism);
            let track = Track::for_program(&d.program).unwrap();
            covers[usize::from(track.fascia)] += 1;
            let mut fixed = Assembly::default();
            track.build_fixed(&mut fixed, &d, "door_hardware#portal4");
            let (lo, hi) = track.overhead_bounds(&d);
            assert!(lo.y > d.center.y + d.height - 0.02 && hi.y < d.center.y + d.height + 0.10);
            assert!(fixed.parts.values().flat_map(|g| &g.positions).all(|v| {
                let p = Vec3::from_array(*v);
                p.cmpge(lo).all() && p.cmple(hi).all()
            }));
            for fraction in [0., 0.19, 0.53, 0.81, 1.] {
                d.program.opening = fraction;
                for (lane, leaf) in lanes(&d).zip(d.leaves()) {
                    let mut hangers = Assembly::default();
                    track.build_hangers(&mut hangers, &d, leaf, "door_hardware#portal4");
                    let stops = track.stops(lane);
                    for side in [-1., 1.] {
                        let x = side * track.hanger_offset(leaf.size.x);
                        let world = |z: f32, y: f32| {
                            leaf.transform.transform_point(Vec3::new(
                                x + 0.00008,
                                leaf.size.y * 0.5 + y,
                                z,
                            ))
                        };
                        let running_y = track.drop + track.gauge;
                        for face in [-1., 1.] {
                            let wheel_z =
                                face * (track.half_width - track.gauge + track.slot_half) * 0.5;
                            let contact = world(wheel_z, running_y);
                            assert!(
                                crosses(&fixed, contact, 0.00010),
                                "missing ledge at {seed}/{fraction}"
                            );
                            assert!(
                                crosses(&hangers, contact, 0.00010),
                                "missing tread at {seed}/{fraction}"
                            );
                        }
                        assert!(!crosses(&fixed, world(0., running_y), 0.001));
                        let x = lane.x_at(fraction) + x;
                        assert!(
                            x - track.radius > stops.x + STOP_LENGTH * 0.5
                                && x + track.radius < stops.y - STOP_LENGTH * 0.5
                        );
                    }
                }
            }
        }
    }
    assert!(covers.into_iter().all(|n| n >= 12));
}

#[test]
fn rails_clear_the_leaf_heads_and_park_only_as_far_as_the_mechanism_requires() {
    for seed in 0..32 {
        for mechanism in [DoorMechanism::Sliding, DoorMechanism::Telescoping] {
            let mut d = installation(seed, mechanism);
            let track = Track::for_program(&d.program).unwrap();
            let mut fixed = Assembly::default();
            track.build_fixed(&mut fixed, &d, "door_hardware#portal4");
            let span = track.span(&d);
            for opening in [0., 1.] {
                d.program.opening = opening;
                let reservation = d.reservation();
                for leaf in d.leaves() {
                    for side in [-1., 1.] {
                        let p = leaf
                            .transform
                            .transform_point(Vec3::X * (side * leaf.size.x * 0.5))
                            .xz();
                        assert!(
                            p.cmpge(reservation.0.xz()).all() && p.cmple(reservation.1.xz()).all()
                        );
                    }
                    let head = leaf.transform.translation.y + leaf.size.y * 0.5;
                    // Any hardware below the leaf head is the wall-side backplate.
                    let inv = d.transform().compute_affine().inverse();
                    let behind = d.wall_depth * 0.5 + 0.030;
                    assert!(fixed.parts.values().flat_map(|g| &g.positions).all(|v| {
                        let p = Vec3::from_array(*v);
                        p.y > head + 0.004
                            || inv.transform_point3(p).z * d.program.swing_side < behind - 0.005
                    }));
                    assert!((head - Track::head(&d) - d.center.y).abs() < 1e-6);
                }
            }
            if mechanism == DoorMechanism::Telescoping {
                assert!(span.y - span.x < 1.65 * d.width);
            }
        }
    }
}

#[test]
fn articulating_keeps_trolleys_on_their_leaf_and_the_header_stationary() {
    for mechanism in [DoorMechanism::Sliding, DoorMechanism::Telescoping] {
        let d = installation(17, mechanism);
        let mut a = Assembly {
            motion_policy: Some(IndoorMotionConfig {
                door_fraction: 1.,
                ..default()
            }),
            ..default()
        };
        d.build(&mut a, true);
        assert_eq!(a.joints.len(), lanes(&d).len());
        // Infer end progress from the first actual joint. Even the short lane
        // in a telescope must arrive at the same progress at the same instant.
        let local_delta = d.transform().rotation.inverse() * a.joints[0].translation;
        let mut target = d.clone();
        target.program.opening += local_delta.x / lanes(&d).next().unwrap().travel;
        assert!((0.0..=1.).contains(&target.program.opening));
        assert!(local_delta.x.abs() > 0.05);
        let head = Track::head(&d) + d.center.y;
        for (i, (start, end)) in d.leaves().into_iter().zip(target.leaves()).enumerate() {
            let joint = &a.joints[i];
            let delta = end.transform.translation - start.transform.translation;
            assert!(joint.translation.distance(delta) < 1e-6);
            assert_eq!(joint.attachment.as_deref(), Some("portal4"));
            assert_eq!(joint.coupled, mechanism == DoorMechanism::Telescoping);
            assert!(joint
                .parts
                .iter()
                .any(|key| key.1.starts_with("door_hardware#")
                    && motion::tag(&key.1) == Some(i)
                    && key.1.split('#').any(|part| part == "portal4")
                    && a.parts[key].positions.iter().any(|v| v[1] > head + 0.02)));
        }
        assert!(a.parts.iter().any(|((_, label), g)| label
            .starts_with("door_hardware#portal4#finish")
            && motion::tag(label).is_none()
            && g.positions.iter().all(|v| v[1] > head)));
    }
}
