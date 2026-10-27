//! Desk fans vary head pose, pedestal/yoke construction and physical guard/blades.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct FanProgram {
    pub radius: f32,
    pub depth: f32,
    pub center: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub blades: usize,
    pub blade_phase: f32,
    pub blade_chord: f32,
    pub blade_sweep: f32,
    pub spokes: usize,
    pub rings: usize,
    pub wire: f32,
    pub base: Vec3,
    pub yoke: bool,
    pub finish: Surface,
    pub blade_finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> FanProgram {
    let mut r = stream(o.seed, 0x4445534b46414e50);
    let s = o.size;
    let pitch = r.random_range(-0.32_f32..0.48);
    let yaw = r.random_range(-0.34_f32..0.34);
    let radius = (s.x * r.random_range(0.32_f32..0.43)).min(s.y * r.random_range(0.23..0.34));
    let depth = s.z * r.random_range(0.24..0.43);
    let wire = s.x.min(s.z) * r.random_range(0.005..0.009);
    let q = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
    // A conservative rotated head box fits before geometry is built.
    let ext = q.mul_vec3(Vec3::X).abs() * radius
        + q.mul_vec3(Vec3::Y).abs() * radius
        + q.mul_vec3(Vec3::Z).abs() * depth * 0.5
        + Vec3::splat(wire);
    let scale = ((s.x * 0.48) / ext.x).min((s.z * 0.48) / ext.z).min(1.);
    let radius = radius * scale;
    let depth = depth * scale;
    let ey = ext.y * scale + wire;
    let cy = s.y * r.random_range(0.61..0.72);
    let center = Vec3::new(0., cy.min(s.y - ey - 0.002), 0.);
    FanProgram {
        radius,
        depth,
        center,
        pitch,
        yaw,
        blades: r.random_range(3..=7),
        blade_phase: r.random_range(-PI..PI),
        blade_chord: r.random_range(0.13..0.24),
        blade_sweep: r.random_range(0.12..0.42),
        spokes: r.random_range(12..=24),
        rings: r.random_range(1..=3),
        wire,
        base: Vec3::new(
            s.x * r.random_range(0.54..0.83),
            s.y * r.random_range(0.035..0.075),
            s.z * r.random_range(0.60..0.94),
        ),
        yoke: r.random_bool(0.42),
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][r.random_range(0..3)],
        blade_finish: [Surface::Plastic, Surface::Metal][r.random_range(0..2)],
    }
}
impl FanProgram {
    pub(crate) fn head_transform(&self) -> Transform {
        Transform::from_translation(self.center)
            .with_rotation(Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch))
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
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
