//! Mechanically distinct supports. All meet the same under-seat mounting datum.
use super::*;

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &ChairProgram,
    pan: Vec2,
    pan_bottom: f32,
    version: u32,
    swivel: Option<&SwivelPedestal>,
) {
    let (width, depth) = (pan.x, pan.y);
    if let Some(stock) = p.runner_stock(o, version) {
        let mounting = if p.formed_seat.is_some() {
            formed::carrier_underside(p.base_style)
        } else {
            pan_bottom
        };
        stock.build(a, p, width, depth, mounting, o.seed);
        return;
    }
    let frame = p.frame_surface;
    let label = "chair";
    if p.base_style < 4 {
        if let Some(fit) = swivel {
            fit.column(a, pan_bottom);
        } else {
            a.part(Surface::Chrome, label).cylinder(
                0.024,
                0.27,
                Transform::from_xyz(0., 0.285, 0.),
            );
            a.part(Surface::Plastic, label).cylinder(
                0.043,
                0.10,
                Transform::from_xyz(0., 0.385, 0.),
            );
        }
        let radius = o.size.x.min(o.size.z) * p.base_radius_fraction;
        if p.base_style == 3 {
            if let Some((fit, disc)) = swivel.and_then(|f| f.disc.map(|d| (f, d))) {
                fit.disc_foot(disc, a.part(frame, label), radius);
                return;
            }
            a.part(frame, label).lathe(
                &[
                    (0., 0.0),
                    (radius, 0.0),
                    (radius, 0.022),
                    (radius * 0.82, 0.037),
                    (0.034, 0.105),
                    (0.034, 0.18),
                    (0., 0.18),
                ],
                32,
                Transform::IDENTITY,
            );
        } else {
            let phase = stream(o.seed, 6433).random_range(0.0..TAU);
            let mut wheels = stream(o.seed, u64::from_le_bytes(*b"CastorAx"));
            let casters = p
                .casters
                .then(|| super::casters::CasterSet::sample(&mut wheels));
            let (reach, mount_y) = casters.as_ref().map_or((radius, 0.072), |c| {
                (
                    radius.min(o.size.x.min(o.size.z) * 0.5 - c.sweep_radius() - 0.001),
                    c.mount_height(),
                )
            });
            for i in 0..p.spoke_count {
                let angle = phase + i as f32 * TAU / p.spoke_count as f32;
                let radial = Vec3::new(angle.sin(), 0., angle.cos());
                let start = swivel.map_or(Vec3::Y * 0.165, |s| s.spoke_root(radial));
                let end = radial * reach + Vec3::Y * mount_y;
                match (
                    p.base_style,
                    swivel.and_then(|s| s.cast_spoke.map(|blade| (blade, s.hub_radius()))),
                ) {
                    (0, _) => a.part(frame, label).rod(start, end, 0.020),
                    (1, _) => {
                        let path: [Vec3; 9] = std::array::from_fn(|i| {
                            let t = i as f32 / 8.;
                            start.lerp(end, t) + Vec3::Y * (PI * t).sin() * p.spoke_arch_m
                        });
                        a.part(frame, label).tube(&path, 0.018, 8);
                    }
                    (_, Some((blade, hub))) => blade.build(a.part(frame, label), start, end, hub),
                    _ => {
                        // Archived tapered cast blade, rather than another round rod.
                        let length = start.distance(end);
                        a.part(frame, label).housing(
                            &[
                                (0., Vec2::new(0.026, 0.035)),
                                (length * 0.65, Vec2::new(0.020, 0.025)),
                                (length, Vec2::new(0.014, 0.021)),
                            ],
                            3.,
                            Transform::from_translation(start).with_rotation(
                                Quat::from_rotation_arc(Vec3::Y, (end - start).normalize()),
                            ),
                        );
                    }
                }
                if let Some(set) = &casters {
                    set.build(a, end, &mut wheels);
                } else {
                    a.part(frame, label).rod(end, end.with_y(0.008), 0.012);
                    a.part(Surface::Rubber, label).cylinder(
                        0.025,
                        0.016,
                        Transform::from_translation(end.with_y(0.008)),
                    );
                }
            }
        }
        return;
    }
    for side in [-1., 1.] {
        let x = side * width * 0.43;
        let z = depth * 0.47;
        match p.base_style {
            4 => {
                for end in [-1., 1.] {
                    let foot = Vec3::new(x * (1. + p.leg_splay), 0.010, end * z);
                    let top = Vec3::new(x, 0.435, end * depth * 0.38);
                    if o.variant % FAMILIES == 3 {
                        a.part(frame, label).housing(
                            &[
                                (0., Vec2::splat(0.013)),
                                (foot.distance(top), Vec2::splat(0.023)),
                            ],
                            4.,
                            Transform::from_translation(foot).with_rotation(
                                Quat::from_rotation_arc(Vec3::Y, (top - foot).normalize()),
                            ),
                        );
                    } else {
                        a.part(frame, label).rod(foot, top, 0.014);
                    }
                }
            }
            5 => {
                a.part(frame, label).tube(
                    &[
                        Vec3::new(x, 0.43, -z),
                        Vec3::new(x, 0.025, -z),
                        Vec3::new(x, 0.025, z),
                        Vec3::new(x, 0.43, z),
                    ],
                    0.016,
                    8,
                );
            }
            6 => {
                let bend = p.frame_bend_m;
                let mut path = vec![Vec3::new(x, 0.024, z), Vec3::new(x, 0.024, -z + bend)];
                for i in 1..=6 {
                    let t = i as f32 / 6. * FRAC_PI_2;
                    path.push(Vec3::new(
                        x,
                        0.024 + bend * (1. - t.cos()),
                        -z + bend * (1. - t.sin()),
                    ));
                }
                path.push(Vec3::new(x, 0.42 - bend, -z));
                for i in 1..=6 {
                    let t = i as f32 / 6. * FRAC_PI_2;
                    path.push(Vec3::new(
                        x,
                        0.42 - bend + bend * t.sin(),
                        -z + bend * (1. - t.cos()),
                    ));
                }
                path.push(Vec3::new(x, 0.42, z));
                a.part(frame, label).tube(&path, 0.017, 10);
            }
            _ => {
                // Trestle feet carry a central upright and an under-seat rail.
                a.part(frame, label)
                    .rod(Vec3::new(x, 0.024, -z), Vec3::new(x, 0.024, z), 0.022);
                a.part(frame, label)
                    .rod(Vec3::new(x, 0.024, 0.), Vec3::new(x, 0.42, 0.), 0.024);
                a.part(frame, label)
                    .rod(Vec3::new(x, 0.42, -z), Vec3::new(x, 0.42, z), 0.018);
            }
        }
        for end in [-1., 1.] {
            let foot_x = x * if p.base_style == 4 {
                1. + p.leg_splay
            } else {
                1.
            };
            a.part(Surface::Rubber, label).cylinder(
                0.022,
                0.012,
                Transform::from_xyz(foot_x, 0.006, end * z),
            );
        }
    }
}

/// The tilt housing and paddle bolt to the upper seat pan and swivel with it.
pub(super) fn tilt_control(a: &mut Assembly, width: f32, depth: f32, pan_bottom: f32) {
    let center = Vec3::new(0., pan_bottom - 0.014, -depth * 0.032);
    a.box_part(
        Surface::Metal,
        "chair",
        center,
        Vec3::new(width * 0.35, 0.031, depth * 0.34),
        0.006,
    );
    let root = center + Vec3::X * (width * 0.13);
    let grip = center + Vec3::new(width * 0.48, -0.013, -depth * 0.09);
    a.part(Surface::Metal, "chair").rod(root, grip, 0.006);
    a.box_part(
        Surface::Plastic,
        "chair",
        grip,
        Vec3::new(0.032, 0.014, 0.045),
        0.006,
    );
}
