//! Geometry from manifests through generator 242.
use super::*;

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters_at(o, FORMED_ROTOR_VERSION - 1);
    let l = o.kind.class_name();
    a.part(p.finish, l).housing(
        &[(0., p.base.xz() * 0.5), (p.base.y, p.base.xz() * 0.39)],
        if p.yoke { 3.5 } else { 2.2 },
        Transform::IDENTITY,
    );
    let tf = p.head_transform();
    let top = if p.yoke {
        p.center.y - p.radius * 0.48
    } else {
        p.center.y
    };
    a.part(p.finish, l)
        .rod(Vec3::Y * p.base.y, Vec3::Y * top, o.size.x * 0.027);
    if p.yoke {
        // Side arms terminate on the actual tilted housing, not a world-axis copy.
        for side in [-1., 1.] {
            let attach = tf.transform_point(Vec3::new(side * p.radius * 0.92, 0., 0.));
            let elbow = Vec3::new(attach.x, top, attach.z);
            a.part(p.finish, l)
                .rod(Vec3::Y * top, elbow, o.size.x * 0.020);
            a.part(p.finish, l).rod(elbow, attach, o.size.x * 0.020);
        }
    }
    let mut head = Assembly::default();
    for sign in [-1., 1.] {
        for ring in 1..=p.rings {
            let r = p.radius * ring as f32 / p.rings as f32;
            let points: Vec<_> = (0..=32)
                .map(|i| {
                    let t = i as f32 * TAU / 32.;
                    Vec3::new(r * t.cos(), r * t.sin(), sign * p.depth * 0.5)
                })
                .collect();
            head.part(p.finish, l).tube(&points, p.wire, 6);
        }
        for i in 0..p.spokes {
            let t = i as f32 * TAU / p.spokes as f32;
            head.part(p.finish, l).rod(
                Vec3::Z * (sign * p.depth * 0.5),
                Vec3::new(t.cos() * p.radius, t.sin() * p.radius, sign * p.depth * 0.5),
                p.wire * 0.58,
            );
        }
    }
    for i in 0..8 {
        let t = i as f32 * TAU / 8.;
        let at = Vec3::new(t.cos() * p.radius, t.sin() * p.radius, 0.);
        head.part(p.finish, l).rod(
            at - Vec3::Z * p.depth * 0.5,
            at + Vec3::Z * p.depth * 0.5,
            p.wire,
        );
    }
    head.part(p.finish, l).cylinder(
        p.radius * 0.21,
        p.depth * 0.97,
        Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
    for i in 0..p.blades {
        let t = p.blade_phase + i as f32 * TAU / p.blades as f32;
        // Blade roots embed in the hub; tip clearance is independent of head pose.
        head.part(p.blade_finish, l).ellipsoid(
            Vec3::new(p.radius * 0.38, p.radius * p.blade_chord, p.depth * 0.055),
            Transform::from_xyz(t.cos() * p.radius * 0.48, t.sin() * p.radius * 0.48, 0.)
                .with_rotation(Quat::from_rotation_z(t + p.blade_sweep)),
        );
    }
    public::append_at(a, head, tf);
}
