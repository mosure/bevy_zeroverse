//! Projector housings, optics and cable ports share one physical construction.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProjectorProgram {
    pub body: Vec3,
    pub foot: f32,
    pub roundness: f32,
    pub taper: f32,
    pub lens_x: f32,
    pub lens_y: f32,
    pub lens_radius: f32,
    pub housing: Surface,
    pub feet_span: Vec2,
    pub controls: usize,
    pub vents: usize,
}
pub fn parameters(o: &IndoorObject) -> ProjectorProgram {
    let mut r = stream(o.seed, 0x50524f4a45435452);
    let body = o.size
        * Vec3::new(
            r.random_range(0.72..0.98),
            r.random_range(0.57..0.90),
            r.random_range(0.65..0.86),
        );
    let foot = (o.size.y * r.random_range(0.045_f32..0.13)).min(o.size.y - body.y - 0.002);
    let lens_x = body.x * r.random_range(-0.28..0.27);
    ProjectorProgram {
        body,
        foot,
        roundness: r.random_range(2.6..9.0),
        taper: r.random_range(0.85..0.99),
        lens_x,
        lens_y: foot + body.y * r.random_range(0.38..0.62),
        lens_radius: body.y.min(body.x) * r.random_range(0.20..0.32),
        housing: if r.random_bool(0.38) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
        feet_span: Vec2::new(r.random_range(0.24..0.30), r.random_range(0.22..0.27)),
        controls: r.random_range(2..=6),
        vents: r.random_range(5..=14),
    }
}
impl ProjectorProgram {
    fn front_z(&self, x: f32) -> f32 {
        self.body.z
            * 0.5
            * (1. - (x.abs() / (self.body.x * 0.5)).powf(self.roundness))
                .max(0.)
                .powf(1. / self.roundness)
    }
    pub fn lens(&self) -> Vec3 {
        Vec3::new(self.lens_x, self.lens_y, self.front_z(self.lens_x))
    }
    /// Front of the visible optical glass. Beam admission starts just beyond it.
    pub fn optical_origin(&self) -> Vec3 {
        self.lens() + Vec3::Z * (self.body.z * 0.045 + 0.0014)
    }
    fn port_surface(&self) -> Vec3 {
        let x = self.body.x * 0.19;
        Vec3::new(x, self.foot + self.body.y * 0.28, -self.front_z(x))
    }
    pub fn port_normal(&self) -> Vec3 {
        let p = self.port_surface();
        Vec3::new(
            (p.x / (self.body.x * 0.5)).powf(self.roundness - 1.) / self.body.x,
            0.,
            -(-p.z / (self.body.z * 0.5)).powf(self.roundness - 1.) / self.body.z,
        )
        .normalize()
    }
    pub fn power_port(&self) -> Vec3 {
        self.port_surface() + self.port_normal() * 0.001
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let s = p.body;
    let half = s.xz() * 0.5;
    a.part(p.housing, l).housing(
        &[
            (p.foot, half * p.taper),
            (p.foot + s.y * 0.18, half),
            (p.foot + s.y * 0.80, half),
            (p.foot + s.y, half * p.taper),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    for x in [-1., 1.] {
        for z in [-1., 1.] {
            a.part(Surface::Rubber, l).cylinder(
                s.x.min(s.z) * 0.032,
                p.foot,
                Transform::from_xyz(
                    x * s.x * p.feet_span.x,
                    p.foot * 0.5,
                    z * s.z * p.feet_span.y,
                ),
            );
        }
    }
    // Partly recessed barrel follows the curved front; its glass and optical
    // origin share the same coordinates used by layout beam clearance.
    let lens = p.lens();
    let radius = p.lens_radius;
    let length = s.z * 0.09;
    let tf = Transform::from_translation(lens).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    a.part(Surface::Rubber, l).cylinder(radius, length, tf);
    a.part(Surface::Metal, l)
        .cylinder(radius * 0.82, length + 0.0006, tf);
    a.part(Surface::Ink, l).cylinder(
        radius * 0.72,
        0.001,
        tf.with_translation(lens + Vec3::Z * (length * 0.5 + 0.0005)),
    );
    a.part(Surface::ContainerGlass, l).ellipsoid(
        Vec3::new(radius * 0.68, radius * 0.68, 0.001),
        Transform::from_translation(p.optical_origin() - Vec3::Z * 0.001),
    );
    for i in 0..p.vents {
        let z = s.z * ((i as f32 + 0.5) / p.vents as f32 - 0.5) * 0.54;
        let x = half.x * (1. - (z.abs() / half.y).powf(p.roundness)).powf(1. / p.roundness);
        a.box_part(
            Surface::Ink,
            l,
            Vec3::new(x, p.foot + s.y * 0.50, z),
            Vec3::new(0.0008, s.y * 0.33, (s.z / p.vents as f32 * 0.20).min(0.004)),
            0.0002,
        );
    }
    for i in 0..p.controls {
        a.box_part(
            Surface::Rubber,
            l,
            Vec3::new(
                s.x * ((i as f32 + 0.5) / p.controls as f32 - 0.5) * 0.45,
                p.foot + s.y + 0.0007,
                -s.z * 0.15,
            ),
            Vec3::new(s.x * 0.035, 0.001, s.z * 0.045),
            0.0003,
        );
    }
    a.part(Surface::Ink, l).cuboid(
        Vec3::new(0.014, 0.008, 0.0008),
        0.0002,
        Transform::from_translation(p.port_surface() + p.port_normal() * 0.0002)
            .with_rotation(Quat::from_rotation_arc(Vec3::Z, p.port_normal())),
    );
}
