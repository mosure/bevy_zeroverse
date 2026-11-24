use super::*;
use crate::scene::procedural_indoor::motion::IndoorMotionConfig;

#[test]
fn closed_leaf_rebates_block_the_jamb_and_head_gaps_but_leave_the_undercut() {
    let mut hinge_counts = std::collections::BTreeSet::new();
    for seed in 0..64 {
        for mechanism in [DoorMechanism::Hinged, DoorMechanism::Paired] {
            let mut d = door(seed, mechanism);
            d.program.opening = 0.;
            d.program.panel = DoorPanel::Flush;
            let h = d.program.swing_hanging.unwrap();
            hinge_counts.insert(h.hinges_per_leaf(&d));
            let mut a = Assembly::default();
            d.build(&mut a, true);
            let clear = (d.width - FRAME_WIDTH_M) * 0.5;
            // Probe below the lowest hinge so its barrel cannot hide a
            // missing stop. Check the emitted triangles from both rooms.
            for side in [-1., 1.] {
                for p in [
                    Vec2::new(-clear + 0.001, 0.081),
                    Vec2::new(clear - 0.001, 0.102),
                    Vec2::new(d.width * 0.19, d.height - FRAME_WIDTH_M * 0.5 - 0.001),
                ] {
                    let (_, _, label) = hit(&a, p, side, |_, _| true).unwrap();
                    assert!(
                        label.starts_with("door_frame#"),
                        "{seed}/{mechanism:?}: {label}"
                    );
                }
                assert!(hit(
                    &a,
                    Vec2::new(d.width * 0.19, FLOOR_GAP * 0.5),
                    side,
                    |_, _| true
                )
                .is_none());
            }
            let z = hit(
                &a,
                Vec2::new(d.width * 0.19, 1.2),
                d.program.swing_side,
                |_, l| l.starts_with("door#"),
            )
            .unwrap()
            .0;
            let actual = (d.wall_depth + 0.020) * 0.5 - z * d.program.swing_side;
            assert!((actual - h.reveal_m).abs() < 0.00001);
        }
    }
    assert_eq!(hinge_counts.into_iter().collect::<Vec<_>>(), [2, 3, 4]);
}

// Separating axes for the horizontal projection of a real triangle and a
// vertical jamb/stop. Works for vertical faces whose projection is a line.
fn intersects(p: [Vec2; 3], lo: Vec2, hi: Vec2) -> bool {
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    [
        Vec2::X,
        Vec2::Y,
        (p[1] - p[0]).perp(),
        (p[2] - p[1]).perp(),
        (p[0] - p[2]).perp(),
    ]
    .into_iter()
    .filter(|a| a.length_squared() > 1e-14)
    .all(|axis| {
        let t = p.map(|v| (v - center).dot(axis));
        let radius = half.dot(axis.abs());
        t.into_iter().fold(f32::INFINITY, f32::min) < radius
            && t.into_iter().fold(f32::NEG_INFINITY, f32::max) > -radius
    })
}

#[test]
fn moving_panels_and_handles_clear_both_jambs_stops_and_the_other_leaf() {
    for seed in 0..20 {
        for mechanism in [DoorMechanism::Hinged, DoorMechanism::Paired] {
            let mut d = door(seed, mechanism);
            d.width = if mechanism == DoorMechanism::Paired {
                1.30
            } else {
                0.93
            };
            let h = d.program.swing_hanging.unwrap();
            let stock = Joinery::sample(&d.program);
            let clear = (d.width - FRAME_WIDTH_M) * 0.5;
            let depth = (d.wall_depth + 0.020) * 0.5;
            let back = h.swing_face(&d) - d.program.thickness - 0.0008;
            for i in 0..h.leaf_count(&d.program) {
                let f = h.fit(&d, i);
                let leaf = Leaf {
                    size: f.size,
                    transform: Transform::IDENTITY,
                };
                let fit = Fit::new(&d, leaf, i, &stock);
                let mut a = Assembly::default();
                panel::build(&mut a, &d.program, leaf, &fit, &stock, i, &labels(&stock));
                hardware::build(&mut a, &d.program, leaf, i, &fit, &stock, "hardware");
                for deg in [0., 1., 3., 5., 8., 14., 30., 60., 82., 90.] {
                    d.program.opening = deg / 90.;
                    let tf = f.leaf(&d).transform;
                    for ((_, label), g) in &a.parts {
                        let points: Vec<_> = g
                            .positions
                            .iter()
                            .map(|&v| {
                                let world = tf.transform_point(Vec3::from_array(v));
                                Vec2::new(world.x, world.z * d.program.swing_side)
                            })
                            .collect();
                        if mechanism == DoorMechanism::Paired {
                            assert!(
                                points.iter().all(|v| f.side * v.x > 0.0002),
                                "meeting edge: {seed}/{i}/{deg}/{label}"
                            );
                        }
                        for indices in g.indices.as_chunks::<3>().0 {
                            let p = indices.map(|i| points[i as usize]);
                            for side in [-1., 1.] {
                                let p = p.map(|v| Vec2::new(v.x * side, v.y));
                                assert!(
                                    !intersects(
                                        p,
                                        Vec2::new(clear, -depth),
                                        Vec2::new(clear + FRAME_WIDTH_M, depth)
                                    ),
                                    "jamb: {seed}/{mechanism:?}/{deg}/{label}"
                                );
                                assert!(
                                    !intersects(
                                        p,
                                        Vec2::new(
                                            clear - h.stop_projection_m,
                                            back - 1.9 * h.stop_projection_m
                                        ),
                                        Vec2::new(clear, back)
                                    ),
                                    "stop: {seed}/{mechanism:?}/{deg}/{label}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rotating_hinge_wings_follow_the_leaf_and_keep_the_fixed_pin_at_every_pose() {
    for mechanism in [DoorMechanism::Hinged, DoorMechanism::Paired] {
        for seed in 0..16 {
            let mut d = door(seed, mechanism);
            d.yaw = 0.83;
            d.center = Vec3::new(2.14, 0., -3.7);
            let h = d.program.swing_hanging.unwrap();
            let mut a = Assembly {
                motion_policy: Some(IndoorMotionConfig {
                    door_fraction: 1.,
                    ..default()
                }),
                ..default()
            };
            d.build(&mut a, true);
            let leaves = d.leaves();
            assert_eq!(a.joints.len(), leaves.len());
            for (i, joint) in a.joints.iter().enumerate() {
                let fit = h.fit(&d, i);
                let fixed = d.transform().transform_point(fit.pin);
                assert!(joint.pivot.distance(fixed) < 1e-6);
                assert!(joint
                    .parts
                    .iter()
                    .any(|(_, label)| label.starts_with("door_hardware#")));
                let mut rest = d.clone();
                for t in [0., 0.29, 0.73, 1.] {
                    rest.program.opening = d.program.opening
                        + t * joint.angle / (fit.side * d.program.swing_side * FRAC_PI_2);
                    let animated = joint.transform(t);
                    let physical = rest.leaves()[i];
                    assert!(animated.transform_point(fixed).distance(fixed) < 2e-6);
                    for q in [Vec3::ZERO, -fit.offset, Vec3::X * fit.size.x * 0.5] {
                        assert!(
                            animated
                                .transform_point(leaves[i].transform.transform_point(q))
                                .distance(physical.transform.transform_point(q))
                                < 4e-6
                        );
                    }
                }
                let mut fixed_wing = crate::scene::procedural_indoor::geometry::Geometry::default();
                let mut moving_wing = fixed_wing.clone();
                hinges::wing(
                    &mut fixed_wing,
                    &d,
                    h,
                    &fit,
                    Transform::IDENTITY,
                    hinges::Wing::Jamb,
                );
                hinges::wing(
                    &mut moving_wing,
                    &d,
                    h,
                    &fit,
                    Transform::IDENTITY,
                    hinges::Wing::Leaf,
                );
                let pitch = h.hinge_length_m / f32::from(h.knuckles);
                for y in h.heights(fit.size) {
                    for i in 0..h.knuckles {
                        let end = y - h.hinge_length_m * 0.5 + f32::from(i + 1) * pitch - 0.0005;
                        let rim_at = |g: &crate::scene::procedural_indoor::geometry::Geometry| {
                            g.positions.iter().any(|&p| {
                                let v = Vec3::from_array(p);
                                (v.y - end).abs() < 1e-5
                                    && (v.xz().length() - h.barrel_radius_m).abs() < 1e-6
                            })
                        };
                        assert_eq!(rim_at(&fixed_wing), i.is_multiple_of(2));
                        assert_eq!(rim_at(&moving_wing), !i.is_multiple_of(2));
                    }
                }
            }
        }
    }
}

#[test]
fn archived_swing_programs_keep_their_original_leaf_and_frame_datum() {
    for mechanism in [DoorMechanism::Hinged, DoorMechanism::Paired] {
        let mut d = door(17, mechanism);
        d.program.opening = 0.;
        let mut archived = serde_json::to_value(&d).unwrap();
        archived["program"]
            .as_object_mut()
            .unwrap()
            .remove("swing_hanging");
        let legacy: DoorInstallation = serde_json::from_value(archived.clone()).unwrap();
        assert!(legacy.program.validate());
        assert_eq!(serde_json::to_value(&legacy).unwrap(), archived);
        let old_face =
            d.program.swing_side * (d.wall_depth * 0.5 + d.program.thickness * 0.5 + 0.008);
        for leaf in legacy.leaves() {
            assert!((leaf.transform.translation.z - old_face).abs() < 1e-6);
        }
        assert!(d.leaves()[0].transform.translation.z.abs() < old_face.abs());
        let mut invalid = d.program.clone();
        invalid.swing_hanging.as_mut().unwrap().knuckles = 4;
        assert!(!invalid.validate());
        invalid = d.program;
        invalid.mechanism = DoorMechanism::Sliding;
        assert!(!invalid.validate());
    }
}
