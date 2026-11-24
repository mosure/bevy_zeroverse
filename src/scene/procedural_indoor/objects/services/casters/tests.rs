use super::*;
use crate::scene::procedural_indoor::objects::audit;

#[test]
fn installed_sets_ground_every_tread_and_fit_the_floor_and_mount() {
    for kind in [ObjectKind::MobileCart, ObjectKind::DrawerUnit] {
        let mut forms = [0_usize; 3];
        let mut rigid_pairs = 0;
        for seed in 0..96 {
            for size in [
                Vec3::new(0.34, 0.48, 0.32),
                Vec3::new(0.75, 0.85, 0.55),
                Vec3::new(0.43, 0.70, 0.60),
            ] {
                let mut o = audit::fixture(kind, seed);
                o.size = size;
                let set = if kind == ObjectKind::MobileCart {
                    let p = cart::parameters(&o);
                    let set = CasterSet::cart(&o, &p);
                    assert!(set.mount_height_m < p.shelf_tops[0] - p.thickness);
                    for m in &set.mounts {
                        assert!((m.pivot_xz_m.abs() - size.xz() * p.post_span).length() < 1e-6);
                    }
                    set
                } else {
                    let p = drawers::parameters(&o);
                    if p.base != 0 {
                        continue;
                    }
                    let set = CasterSet::pedestal(&o, &p);
                    assert_eq!(set.mount_height_m, p.foot);
                    for m in &set.mounts {
                        let floor = Vec2::new(size.x * 0.98, size.z - 0.065) * 0.5;
                        let delta = (m.pivot_xz_m - Vec2::new(0., -0.025)).abs();
                        assert!((delta + Vec2::splat(0.43 * set.radius_m))
                            .cmple(floor)
                            .all());
                    }
                    set
                };
                if let Wheel::Spoked(count) = set.wheel {
                    let phases = set.mounts.map(|m| m.spoke_phase_rad.unwrap());
                    assert!(phases
                        .iter()
                        .all(|&phi| (0.0..2. * PI / f32::from(count)).contains(&phi)));
                    assert!(phases.windows(2).any(|p| (p[1] - p[0]).abs() > 1e-4));
                } else {
                    assert!(set.mounts.iter().all(|m| m.spoke_phase_rad.is_none()));
                }
                let fixed: Vec<_> = set.mounts.iter().filter(|m| m.fixed).collect();
                assert!(fixed.is_empty() || (kind == ObjectKind::MobileCart && fixed.len() == 2));
                if fixed.len() == 2 {
                    rigid_pairs += 1;
                    assert_eq!(fixed[0].heading_rad, fixed[1].heading_rad);
                    if size.x >= size.z {
                        assert_eq!(fixed[0].pivot_xz_m.x, fixed[1].pivot_xz_m.x);
                    } else {
                        assert_eq!(fixed[0].pivot_xz_m.y, fixed[1].pivot_xz_m.y);
                    }
                }
                forms[match set.wheel {
                    Wheel::Solid => 0,
                    Wheel::Spoked(_) => 1,
                    Wheel::Twin => 2,
                }] += 1;
                let mut a = Assembly::default();
                set.build(&mut a, o.kind.class_name());
                let check = audit::check(&o, &a).unwrap();
                assert!(check.max_envelope_overrun_m < 1e-6 && check.min_y_m.abs() < 1e-6);
                assert!((a.bounds().1.y - set.mount_height_m).abs() < 1e-6);
                let tread = &a.parts[&(Surface::Rubber, o.kind.class_name().into())];
                let count = if set.wheel == Wheel::Twin { 2 } else { 1 };
                let wheels = tread
                    .positions
                    .chunks_exact(tread.positions.len() / (4 * count));
                for (i, vertices) in wheels.enumerate() {
                    let m = set.mounts[i / count];
                    let swivel = Quat::from_rotation_y(m.heading_rad);
                    let roll = (swivel * Vec3::Z).xz();
                    let across = (swivel * Vec3::X).xz();
                    let offset = if count == 1 {
                        0.
                    } else {
                        (0.19 * set.radius_m + set.tread_width_m * 0.5)
                            * if i % 2 == 0 { -1. } else { 1. }
                    };
                    let expected = m.pivot_xz_m + across * offset
                        - roll * if m.fixed { 0. } else { set.swivel_trail_m };
                    let (lo, hi) = vertices.iter().map(|p| Vec3::from(*p)).fold(
                        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                        |(lo, hi), v| (lo.min(v), hi.max(v)),
                    );
                    assert!(lo.y.abs() < 1e-6 && (hi.y - 2. * set.radius_m).abs() < 1e-6);
                    assert!(((hi + lo).xz() * 0.5 - expected).length() < 1e-6);
                    // Envelope clearance must also hold if the swivel is left
                    // pointing at the nearest side of the cabinet.
                    let clear = (size.xz() * 0.5 - m.pivot_xz_m.abs()).min_element();
                    assert!(vertices
                        .iter()
                        .all(|p| (Vec3::from(*p).xz() - m.pivot_xz_m).length() <= clear + 1e-6));
                }
            }
        }
        assert!(forms.iter().all(|&n| n > 0));
        assert_eq!(rigid_pairs > 0, kind == ObjectKind::MobileCart);
    }
}

// Project actual emitted triangles along the axle, including the fork and hub.
// Only the chosen corner belongs in the test; opposite wheels may be collinear.
fn occludes(assembly: &Assembly, axle: Vec3, swivel: Quat, radius: f32, screen: Vec2) -> bool {
    assembly.parts.values().any(|g| {
        g.indices.as_chunks::<3>().0.iter().any(|ids| {
            let p = ids
                .map(|i| swivel.inverse() * (Vec3::from(g.positions[i as usize]) - axle) / radius);
            if p.iter().all(|v| v.x > 1.) || p.iter().all(|v| v.x < -1.) {
                return false;
            }
            let [a, b, c] = p.map(|v| v.yz());
            let area = (b - a).perp_dot(c - a);
            if area.abs() < 1e-6 {
                return false;
            }
            let u = (screen - a).perp_dot(c - a) / area;
            let v = (b - a).perp_dot(screen - a) / area;
            u >= 0. && v >= 0. && u + v <= 1.
        })
    })
}

#[test]
fn a_ray_through_a_spoke_opening_sees_the_room_and_a_solid_core_blocks_it() {
    let mut seen = [false; 3];
    for seed in 0..120 {
        let o = audit::fixture(ObjectKind::MobileCart, seed);
        let p = cart::parameters(&o);
        let set = CasterSet::cart(&o, &p);
        let kind = match set.wheel {
            Wheel::Solid => 0,
            Wheel::Spoked(_) => 1,
            Wheel::Twin => 2,
        };
        if seen[kind] {
            continue;
        }
        seen[kind] = true;
        let mut a = Assembly::default();
        set.build(&mut a, o.kind.class_name());
        let m = set.mounts[0];
        let rotation = Quat::from_rotation_y(m.heading_rad);
        let axle = Vec3::new(m.pivot_xz_m.x, set.radius_m, m.pivot_xz_m.y)
            - rotation * Vec3::Z * if m.fixed { 0. } else { set.swivel_trail_m };
        let mut open = 0;
        for i in 0..36 {
            let angle = PI * (0.56 + 0.88 * (i as f32 + 0.5) / 36.);
            let ray = Vec2::from_angle(angle);
            assert!(
                occludes(&a, axle, rotation, set.radius_m, ray * 0.91),
                "missing tyre"
            );
            open += usize::from(!occludes(&a, axle, rotation, set.radius_m, ray * 0.48));
        }
        if kind == 1 {
            assert!((12..=33).contains(&open), "{open}");
        } else {
            assert_eq!(open, 0);
        }
    }
    assert_eq!(seen, [true; 3]);
}

#[test]
fn a_post_matching_the_bearing_diameter_seats_without_a_sliver_or_coplanar_cap() {
    let o = audit::fixture(ObjectKind::MobileCart, 29);
    let set = CasterSet::cart(&o, &cart::parameters(&o));
    for ratio in [0.999_999, 1.0, 1.000_001] {
        let mut a = Assembly::default();
        let label = o.kind.class_name();
        set.build(&mut a, label);
        let posts = a.part(Surface::Wood, label);
        for m in set.mounts {
            let p = Vec3::new(m.pivot_xz_m.x, 0., m.pivot_xz_m.y);
            posts.rod(
                p.with_y(set.post_bottom_m()),
                p.with_y(set.mount_height_m + 0.08),
                ratio * 0.43 * set.radius_m,
            );
        }
        let bottom = posts
            .positions
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, f32::min);
        assert!(bottom < set.mount_height_m - 0.001 * set.radius_m);
        assert!(bottom > set.mount_height_m - 0.10 * set.radius_m);
        audit::check(&o, &a).unwrap();
    }
}
