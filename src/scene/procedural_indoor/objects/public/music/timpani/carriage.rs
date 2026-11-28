use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, t: Timpani) {
    let r = t.head_radius_m;
    let h = t.head_height_m;
    let hub = t.point(0., 0., 0.145);
    // Counterhoop lugs pull on a common tension spider below the kettle.
    let rods = a.part(Surface::Chrome, LABEL);
    for i in 0..t.tension_rods {
        let angle = TAU * (i as f32 + 0.5) / t.tension_rods as f32;
        let lug = t.point(angle, r + 0.025, h - 0.018);
        let knee = t.point(angle, r * 0.85, h - 0.43 * t.bowl_depth_m);
        let bottom = t.point(angle, r * 0.49, h - t.bowl_depth_m - 0.009);
        rods.tube(&[lug, knee, bottom, hub], 0.005, 6);
        let tf = Transform::from_translation(lug).with_rotation(Quat::from_rotation_y(angle));
        // Forged six-sided lug seats the rod against the counterhoop.
        rods.lathe(
            &[
                (0., -0.012),
                (0.013, -0.012),
                (0.016, -0.008),
                (0.016, 0.007),
                (0.012, 0.012),
                (0., 0.012),
            ],
            6,
            tf,
        );
    }
    // Three load-bearing casters. The forward foot carries the pedal on the
    // same chassis, with clearance between the swinging tread and the wheel.
    let foot_z = o.size.z * 0.5 - 0.076;
    for (i, angle) in [0., 2.10, -2.10].into_iter().enumerate() {
        let radial = if i == 0 {
            foot_z - t.bowl_center_z_m
        } else {
            r * 0.92
        };
        let wheel_radius = if i == 0 {
            t.wheel_radius_m * 0.82
        } else {
            t.wheel_radius_m
        };
        let wheel = t.point(angle, radial, wheel_radius);
        let rot = Quat::from_rotation_y(angle);
        let axle = Transform::from_translation(wheel)
            .with_rotation(rot * Quat::from_rotation_z(FRAC_PI_2));
        let half = 0.013;
        let rim = wheel_radius * 0.57;
        a.part(Surface::Rubber, LABEL).lathe(
            &[
                (rim, -half),
                (wheel_radius - 0.004, -half),
                (wheel_radius, -half + 0.004),
                (wheel_radius, half - 0.004),
                (wheel_radius - 0.004, half),
                (rim, half),
            ],
            16,
            axle,
        );
        a.part(Surface::Metal, LABEL).lathe(
            &[(0., -half), (rim, -half), (rim, half), (0., half)],
            16,
            axle,
        );
        let fork_top = wheel + Vec3::Y * (wheel_radius + 0.013);
        let frame = a.part(Surface::Metal, LABEL);
        caster_fork(frame, wheel, rot, half, wheel_radius + 0.013);
        let low = t.point(angle, r * 0.69, 0.17);
        frame.tube(&[hub, low, fork_top], 0.018, 8);
        let knee = t.point(angle, r + 0.005, h - 0.32 * t.bowl_depth_m);
        frame.tube(&[low, knee, t.point(angle, r + 0.004, h - 0.034)], 0.014, 8);
    }
    let centre_z = foot_z - 0.049;
    let pedal = Transform::from_xyz(0., 0.165, centre_z)
        .with_rotation(Quat::from_rotation_x(t.pedal_pitch_rad));
    let frame = a.part(Surface::Metal, LABEL);
    let bearing = pedal.transform_point(Vec3::new(0., -0.014, -0.034));
    frame.tube(&[hub, bearing], 0.014, 8);
    frame.cuboid(Vec3::new(0.081, 0.019, 0.201), 0.006, pedal);
    a.part(Surface::Rubber, LABEL).cuboid(
        Vec3::new(0.071, 0.008, 0.151),
        0.003,
        Transform {
            translation: pedal.transform_point(Vec3::new(0., 0.012, 0.013)),
            ..pedal
        },
    );
}

/// A single folded sheet straddles the tyre, open below the axle. The two
/// cheeks meet the bridge at mitres in the side faces of the extrusion.
fn caster_fork(g: &mut Geometry, axle: Vec3, rot: Quat, tyre_half: f32, height: f32) {
    let inside = tyre_half + 0.002;
    let outside = inside + 0.004;
    let underside = height - 0.004;
    let section = [
        Vec2::new(-outside, -0.006),
        Vec2::new(-outside, height),
        Vec2::new(outside, height),
        Vec2::new(outside, -0.006),
        Vec2::new(inside, -0.006),
        Vec2::new(inside, underside),
        Vec2::new(-inside, underside),
        Vec2::new(-inside, -0.006),
    ];
    let at = |p: Vec2, end: f32| axle + rot * p.extend(end * 0.011);
    for i in 0..section.len() {
        let a = section[i];
        let b = section[(i + 1) % section.len()];
        g.metric_quad(
            [at(a, -1.), at(b, -1.), at(b, 1.), at(a, 1.)],
            rot * (b - a).perp().extend(0.),
        );
    }
    for end in [-1., 1.] {
        for patch in [[0, 1, 6, 7], [1, 2, 5, 6], [2, 3, 4, 5]] {
            g.metric_quad(patch.map(|i| at(section[i], end)), rot * Vec3::Z * end);
        }
    }
}

/// Side-mounted tuning scale with an index driven by the pedal position.
/// The face normal points out and up to the player, on either side of the bowl.
pub(super) fn gauge(a: &mut Assembly, t: Timpani, angle: f32) {
    let r = t.head_radius_m;
    let center = t.point(angle, r + 0.025, t.head_height_m - 0.105);
    let rot = Quat::from_rotation_y(angle) * Quat::from_rotation_x(-0.24);
    let pose = |x: f32, y: f32, z: f32| {
        Transform::from_translation(center + rot * Vec3::new(x, y, z)).with_rotation(rot)
    };
    let frame = a.part(Surface::Metal, LABEL);
    frame.tube(
        &[
            t.point(angle, r + 0.01, t.head_height_m - 0.027),
            pose(0., 0.027, -0.008).translation,
        ],
        0.006,
        6,
    );
    frame.cuboid(Vec3::new(0.106, 0.060, 0.010), 0.0025, pose(0., 0., 0.));
    let scale = a.part(Surface::Chrome, LABEL);
    for i in 0..7 {
        scale.cuboid(
            Vec3::new(0.0028, if i % 3 == 0 { 0.012 } else { 0.007 }, 0.002),
            0.,
            pose((i as f32 - 3.) * 0.013, 0.016, 0.0056),
        );
    }
    let fraction = (t.pedal_pitch_rad - 0.08) / (0.29 - 0.08);
    scale.tube(
        &[
            pose(0., -0.021, 0.008).translation,
            pose((fraction - 0.5) * 0.079, 0.008, 0.009).translation,
        ],
        0.0021,
        6,
    );
}
