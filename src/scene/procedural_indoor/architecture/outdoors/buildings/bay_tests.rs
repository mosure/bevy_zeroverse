use super::*;

/// Intersect front-facing production triangles. Dropping all panes also lets
/// the proxy test measure the masonry backing behind an opaque pane.
fn first_hit(a: &Assembly, origin: Vec3, direction: Vec3, skip_glass: bool) -> (Surface, f32) {
    a.parts
        .iter()
        .filter(|((s, _), _)| !skip_glass || *s != Surface::Glass)
        .flat_map(|((surface, _), g)| {
            g.indices
                .as_chunks::<3>()
                .0
                .iter()
                .filter_map(move |indices| {
                    let [a, b, c] = indices.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let n = (b - a).cross(c - a);
                    let denom = n.dot(direction);
                    if denom >= -1e-8 {
                        return None;
                    }
                    let t = n.dot(a - origin) / denom;
                    let p = origin + direction * t;
                    (t > 0.
                        && [(a, b), (b, c), (c, a)]
                            .into_iter()
                            .all(|(a, b)| (b - a).cross(p - a).dot(n) >= -1e-7))
                    .then_some((*surface, t))
                })
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("enclosed building must intercept the ray")
}

#[test]
fn actual_bay_has_distinct_wall_sash_glass_and_finite_backing_at_oblique_angles() {
    let site = Parcel {
        center: Vec2::new(9., -9.),
        size: Vec2::new(2.8, 3.4),
        yaw: 0.79,
        kind: 0,
        seed: 41,
    };
    let mut b = Building::sample(&site, 30., bevy_zeroverse_capture::GENERATOR_VERSION);
    b.levels = 1;
    b.bay_pitch = 4.;
    b.pitch = 3.2;
    b.window_fraction = 0.57;
    b.window_height = 0.49;
    b.sill = 0.17;
    b.wall = Surface::Concrete;
    b.frame = Surface::Metal;
    b.frame_width = 0.043;
    b.window_installation.recess_m = 0.22;
    b.window_installation.rebate_m = 0.018;
    b.window_installation.roller_probability = 0.;
    let mut a = Assembly::default();
    b.split = 1.;
    build(&mut a, &site, &b, None);
    let rotation = Quat::from_rotation_y(site.yaw);
    let eye = |u| {
        Vec3::new(site.center.x, 0., site.center.y)
            + rotation
                * Vec3::new(
                    u,
                    -0.075 + b.pitch * (b.sill + b.window_height * 0.43),
                    -site.size.y * 0.5 - 1.,
                )
    };
    let width = (site.size.x - 0.20) * b.window_fraction;
    let outer = width * 0.5 + b.frame_width;
    let straight = rotation * Vec3::Z;
    for (u, skip, expected_surface, depth) in [
        (outer + 0.075, false, b.wall, 0.),
        (width * 0.5 + b.frame_width * 0.42, false, b.frame, 0.22),
        (width * 0.24, false, Surface::Glass, 0.238),
        (width * 0.24, true, b.wall, 0.256),
    ] {
        let (surface, t) = first_hit(&a, eye(u), straight, skip);
        assert_eq!(surface, expected_surface);
        assert!((t - 1. - depth).abs() < 2e-5, "u={u}, t={t}, depth={depth}");
    }
    let mut near = Assembly::default();
    build(&mut near, &site, &b, Some(2));
    assert!(near.parts.contains_key(&(Surface::Glass, "window".into())));
    for u in [-width * 0.24, width * 0.24] {
        let (surface, first) = first_hit(&near, eye(u), straight, false);
        assert_eq!(surface, Surface::Glass);
        assert!((first - 1.238).abs() < 2e-5);
        let (_, opaque) = first_hit(&near, eye(u), straight, true);
        assert!(
            opaque > first + 0.50 && opaque < 1. + site.size.y,
            "room at {opaque}"
        );
    }
    for slope in [-0.42, -0.24, 0.18, 0.37] {
        let (_, t) = first_hit(
            &near,
            eye(width * 0.22 - slope),
            rotation * Vec3::new(slope, 0., 1.),
            true,
        );
        assert!(
            (1.0..1. + site.size.y).contains(&t),
            "grazing ray escaped at {t}"
        );
    }
    let (surface, t) = first_hit(
        &a,
        eye(-outer + 0.045 + 0.55),
        rotation * Vec3::new(-0.55, 0., 1.),
        false,
    );
    assert_eq!(surface, b.wall);
    assert!(
        (t - 1. - 0.045 / 0.55).abs() < 2e-5,
        "missing jamb at oblique incidence {t}"
    );
}
