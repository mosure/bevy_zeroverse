use super::*;

fn door(seed: u64, mechanism: DoorMechanism) -> DoorInstallation {
    DoorInstallation {
        id: 3,
        center: Vec3::ZERO,
        yaw: 0.,
        width: 1.36,
        height: 2.17,
        wall_depth: 0.21,
        program: DoorProgram::sample(seed, mechanism),
    }
}

fn labels(j: &Joinery) -> Labels {
    Labels {
        leaf: "door#portal3#finish0".into(),
        seal: "door#portal3".into(),
        glass: format!("door#portal3#finish{}", j.glass.slot()),
        hardware: "door_hardware#portal3#finish0".into(),
    }
}

// Independent orthographic first-hit oracle, including surfaces on the other
// side of through-holes. The caller chooses whether glass participates.
fn hit(
    a: &Assembly,
    p: Vec2,
    side: f32,
    include: impl Fn(Surface, &str) -> bool,
) -> Option<(f32, Surface, &str)> {
    hit_at_depth(a, p, side, include, |_| true)
}

fn hit_at_depth(
    a: &Assembly,
    p: Vec2,
    side: f32,
    include: impl Fn(Surface, &str) -> bool,
    depth: impl Fn(f32) -> bool,
) -> Option<(f32, Surface, &str)> {
    a.parts
        .iter()
        .filter(|((s, l), _)| include(*s, l))
        .flat_map(|((s, l), g)| {
            g.indices.as_chunks::<3>().0.iter().filter_map(move |idx| {
                let [a, b, c] = idx.map(|i| Vec3::from_array(g.positions[i as usize]));
                let u = b - a;
                let v = c - a;
                let q = p - a.truncate();
                let det = u.truncate().perp_dot(v.truncate());
                if det.abs() < 1e-11 {
                    return None;
                }
                let i = q.perp_dot(v.truncate()) / det;
                let j = u.truncate().perp_dot(q) / det;
                (i >= 0. && j >= 0. && i + j <= 1.00001).then_some((
                    a.z + i * u.z + j * v.z,
                    *s,
                    l.as_str(),
                ))
            })
        })
        .filter(|(z, _, _)| depth(*z))
        .max_by(|a, b| (a.0 * side).total_cmp(&(b.0 * side)))
}

#[test]
fn latch_fixings_contact_solid_stock_and_pockets_leave_a_closed_web() {
    for mechanism in DoorMechanism::ALL {
        for seed in 0..30 {
            let d = door(seed, mechanism);
            let j = Joinery::sample(&d.program);
            for (i, leaf) in d.leaves().into_iter().enumerate() {
                let leaf = Leaf {
                    transform: Transform::IDENTITY,
                    ..leaf
                };
                let fit = Fit::new(&d, leaf, i, &j);
                let mut a = Assembly::default();
                let labels = labels(&j);
                build_leaf(&mut a, &d, leaf, i, &j, &labels);
                for side in [-1., 1.] {
                    let hz = leaf.size.z * 0.5;
                    let point = fit.mount;
                    let handle = Fit::handle(&d.program, i, side);
                    let fixes = match handle {
                        DoorHandle::Recessed => {
                            let (z, _, label) =
                                hit(&a, point + Vec2::splat(0.0001), side, |_, _| true).unwrap();
                            assert!(
                                label.starts_with("door_hardware#")
                                    && (0.007..=0.0141).contains(&(hz - z * side))
                            );
                            assert!(z * side > 0.003, "pockets must leave a solid web");
                            continue;
                        }
                        DoorHandle::Pull => {
                            vec![point - Vec2::Y * j.pull, point + Vec2::Y * j.pull]
                        }
                        DoorHandle::PanicBar => {
                            fit.bar_anchors.map(|x| Vec2::new(x, point.y)).to_vec()
                        }
                        _ => vec![point],
                    };
                    for xy in fixes {
                        let (z, _, _) = hit(&a, xy + Vec2::new(0.0001, 0.0002), side, |s, l| {
                            s != Surface::GlassInterior && l.starts_with("door#")
                        })
                        .unwrap();
                        assert!(
                            (z * side - hz).abs() < 2e-6,
                            "floating mounting shoe, {seed}/{mechanism:?}"
                        );
                        // A large flat backplate can seat against the door across
                        // an entire triangle with no vertex near its centre.
                        let contact = hit_at_depth(
                            &a,
                            xy + Vec2::new(0.0001, 0.0002),
                            side,
                            |_, l| l == labels.hardware,
                            |z| (z - side * hz).abs() < 1e-5,
                        );
                        assert!(contact.is_some(),
                            "shoe misses solid face: seed={seed} {mechanism:?} leaf={i} handle={handle:?} side={side} xy={xy:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn telescoping_and_sliding_hardware_clear_wall_and_neighboring_tracks_through_travel() {
    for mechanism in [DoorMechanism::Sliding, DoorMechanism::Telescoping] {
        for seed in 0..24 {
            let mut d = door(seed, mechanism);
            let j = Joinery::sample(&d.program);
            for opening in [0., 0.19, 0.61, 0.98, 1.] {
                d.program.opening = opening;
                let sign = d.program.swing_side;
                let mut back = d.wall_depth * 0.5;
                for (i, leaf) in d.leaves().into_iter().enumerate() {
                    let mut a = Assembly::default();
                    build_leaf(&mut a, &d, leaf, i, &j, &labels(&j));
                    let (lo, hi) = a.bounds();
                    let (near, far) = if sign > 0. {
                        (lo.z, hi.z)
                    } else {
                        (-hi.z, -lo.z)
                    };
                    assert!(
                        near - back > 0.008,
                        "{mechanism:?}/{seed} span {near}..{far} following {back}"
                    );
                    back = far;
                }
            }
        }
    }
}

#[test]
fn calibrated_lights_have_sealed_rebates_and_solid_shoulder_panels() {
    let mut width = Vec::new();
    let mut height = Vec::new();
    let mut offset = Vec::new();
    for seed in 0..96 {
        let mut d = door(seed, DoorMechanism::Hinged);
        d.program.panel = if seed % 2 == 0 {
            DoorPanel::Glazed
        } else {
            DoorPanel::VisionLite
        };
        let j = Joinery::sample(&d.program);
        let leaf = Leaf {
            transform: Transform::IDENTITY,
            ..d.leaves()[0]
        };
        let fit = Fit::new(&d, leaf, 0, &j);
        let hole = fit.opening.unwrap();
        let mut a = Assembly::default();
        panel::build(&mut a, &d.program, leaf, &fit, &j, 0, &labels(&j));
        for side in [-1., 1.] {
            let (z, s, label) =
                hit(&a, hole.center() + Vec2::splat(0.0001), side, |_, _| true).unwrap();
            assert_eq!(s, Surface::GlassInterior);
            assert_eq!(
                crate::scene::procedural_indoor::materials::finish_slot(s, label),
                Some(j.glass.slot())
            );
            assert!((z - (j.inset + side * j.glass.metres() * 0.5)).abs() < 2e-6);
            assert!(hit(&a, hole.center(), side, |s, _| s != Surface::GlassInterior).is_none());
            for (dx, stock) in [
                (-0.002, d.program.finish),
                (j.bead * 0.5, d.program.finish),
                (j.bead + j.seal * 0.5, Surface::Rubber),
            ] {
                let (_, s, _) = hit(
                    &a,
                    Vec2::new(hole.min.x + dx, hole.center().y),
                    side,
                    |s, _| s != Surface::GlassInterior,
                )
                .unwrap();
                assert_eq!(s, stock);
            }
        }
        if d.program.panel == DoorPanel::VisionLite {
            width.push(hole.width() / leaf.size.x);
            height.push(hole.height() / leaf.size.y);
            offset.push(hole.center().x / leaf.size.x);
            let (_, s, _) = hit(
                &a,
                Vec2::new(hole.center().x, hole.min.y - 0.02),
                1.,
                |_, _| true,
            )
            .unwrap();
            assert_eq!(s, d.program.finish);
        }
    }
    for (v, span) in [(width, 0.50), (height, 0.35), (offset, 0.24)] {
        let range = v.iter().copied().reduce(f32::max).unwrap()
            - v.iter().copied().reduce(f32::min).unwrap();
        assert!(range > span);
    }
}

#[test]
fn recessed_fields_and_cross_rails_form_an_opaque_leaf() {
    for seed in 0..24 {
        let mut d = door(seed, DoorMechanism::Hinged);
        d.program.panel = DoorPanel::Recessed;
        let j = Joinery::sample(&d.program);
        let leaf = Leaf {
            transform: Transform::IDENTITY,
            ..d.leaves()[0]
        };
        let fit = Fit::new(&d, leaf, 0, &j);
        let mut a = Assembly::default();
        panel::build(&mut a, &d.program, leaf, &fit, &j, 0, &labels(&j));
        let opening = fit.opening.unwrap();
        for step in 0..150 {
            let p = Vec2::new(
                opening.center().x + 0.01,
                opening.min.y + opening.height() * (step as f32 + 0.5) / 150.,
            );
            let (front, _, _) = hit(&a, p, 1., |_, _| true).unwrap();
            let (back, _, _) = hit(&a, p, -1., |_, _| true).unwrap();
            assert!(front - back >= leaf.size.z - 2. * j.recess - 1e-5);
        }
    }
}

#[test]
fn revolving_drum_is_a_sealed_optically_calibrated_sheet_with_clear_entry_sectors() {
    // Rotate the assembly to look along the middle of one of the two arcs.
    let mut d = door(12, DoorMechanism::Revolving);
    d.yaw = -FRAC_PI_2;
    let mut a = Assembly::default();
    drum::build(&mut a, &d, "door_frame#portal3");
    let radial = d.width * 0.5 - 0.015;
    for i in -145..=145 {
        let angle = i as f32 / 256.;
        let x = radial * angle.sin();
        let (z, s, l) = hit(&a, Vec2::new(x, d.height * 0.55), 1., |s, _| {
            s == Surface::GlassInterior
        })
        .unwrap();
        assert_eq!(
            crate::scene::procedural_indoor::materials::finish_slot(s, l),
            Some(crate::scene::procedural_indoor::materials::glass::SheetGauge::Eight.slot())
        );
        assert!((Vec2::new(x, z).length() - (radial + 0.004)).abs() < 0.0005);
    }
    assert!(
        hit(&a, Vec2::new(radial * 0.9, d.height * 0.5), 1., |s, _| s
            == Surface::GlassInterior)
        .is_none()
    );
}

#[test]
fn revolving_push_bars_clear_the_neighboring_wing_and_stock_seats_in_the_spindle() {
    for seed in 0..64 {
        let d = door(seed, DoorMechanism::Revolving);
        let j = Joinery::sample(&d.program);
        let leaf = Leaf {
            transform: Transform::IDENTITY,
            ..d.leaves()[0]
        };
        let mut a = Assembly::default();
        let labels = labels(&j);
        build_leaf(&mut a, &d, leaf, 0, &j, &labels);
        let points = &a.parts[&(d.program.hardware_finish, labels.hardware)].positions;
        let offset = ROTOR_HEEL_RADIUS_M + leaf.size.x * 0.5;
        let inner = points
            .iter()
            .map(|p| p[0] + offset)
            .reduce(f32::min)
            .unwrap();
        let side = points.iter().map(|p| p[2].abs()).reduce(f32::max).unwrap();
        // Rotating this same swept footprint by a quarter turn interchanges the
        // two axes. Positive separation proves disjoint bars at every phase.
        assert!(inner - side > 0.0058);
        let mut frame = Assembly::default();
        drum::build(&mut frame, &d, "door_frame#portal3");
        let spindle = &frame.parts[&(d.program.hardware_finish, "door_hardware#portal3".into())];
        let radius = spindle
            .positions
            .iter()
            .map(|p| p[0])
            .reduce(f32::max)
            .unwrap();
        assert!(radius > ROTOR_HEEL_RADIUS_M + 0.003);
    }
}
