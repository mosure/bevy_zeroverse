//! Waste containers with independent metric walls, rim, lid and support details.
use super::*;

#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct BinProgram {
    pub squareness: f32,
    pub taper: f32,
    pub rim_width: f32,
    pub lid: u8,
    pub finish: Surface,
    pub body: Vec3,
    pub floor: f32,
    pub wall: f32,
    pub rim_height: f32,
    pub lid_height: f32,
    pub seam: f32,
    pub pedal_width: f32,
    pub pedal_height: f32,
    pub pedal_offset: f32,
    pub rib_count: usize,
    pub rib_radius: f32,
    pub rib_phase: f32,
    pub divider_x: f32,
    pub hardware: Surface,
}
pub fn parameters(o: &IndoorObject) -> BinProgram {
    let mut r = stream(o.seed, 0x42494e50524f4752);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x42494e46494e4953);
    let s = o.size;
    let lid = r.random_range(0..5);
    let lid_height = if matches!(lid, 1 | 2) {
        s.y * r.random_range(0.025..0.10)
    } else {
        0.
    };
    let seam = s.y * r.random_range(0.004..0.012);
    let body = Vec3::new(
        s.x * r.random_range(0.88..0.95),
        s.y - lid_height
            - if lid_height > 0. { seam } else { 0. }
            - if lid == 2 { s.y * 0.003 } else { 0. },
        s.z * r.random_range(0.85..0.94),
    );
    BinProgram {
        squareness: r.random_range(2.0..8.5),
        taper: r.random_range(0.64..0.98),
        rim_width: r.random_range(0.025..0.075),
        lid,
        finish: [Surface::Metal, Surface::Plastic, Surface::Chrome][f.random_range(0..3)],
        body,
        floor: s.y * r.random_range(0.012..0.027),
        wall: s.x.min(s.z) * r.random_range(0.007..0.017),
        rim_height: s.y * r.random_range(0.025..0.07),
        lid_height,
        seam,
        pedal_width: r.random_range(0.22..0.52),
        pedal_height: s.y * r.random_range(0.04..0.095),
        pedal_offset: r.random_range(-0.12..0.12),
        rib_count: r.random_range(10..=28),
        rib_radius: s.x.min(s.z) * r.random_range(0.002..0.004),
        rib_phase: r.random_range(0.0..TAU),
        divider_x: r.random_range(-0.22..0.22),
        hardware: if f.random_bool(0.55) {
            Surface::Plastic
        } else {
            Surface::Metal
        },
    }
}
impl BinProgram {
    pub fn half_at(&self, y: f32) -> Vec2 {
        self.body.xz()
            * 0.5
            * (self.taper + (1. - self.taper) * (y / (self.body.y - self.rim_height)).min(1.))
    }
    fn perimeter(&self, t: f32, y: f32) -> Vec3 {
        let half = self.half_at(y);
        let power = 2. / self.squareness;
        // Interpolate the actual 32-sided housing, not the ideal curve outside
        // it. Even thin ribs therefore stay attached between sampled vertices.
        let u = t.rem_euclid(TAU) / TAU * 32.;
        let point = |i: f32| {
            let angle = i * TAU / 32.;
            let power = |v: f32| {
                if v.abs() < 1e-6 {
                    0.
                } else {
                    v.signum() * v.abs().powf(power)
                }
            };
            Vec3::new(half.x * power(angle.cos()), y, half.y * power(angle.sin()))
        };
        point(u.floor()).lerp(point(u.floor() + 1.), u.fract())
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let half = p.body.xz() * 0.5;
    let bottom = half * p.taper;
    let top = p.body.y;
    let rim_inner = half - Vec2::splat(p.wall.max(half.min_element() * p.rim_width));
    a.part(p.finish, l).housing(
        &[
            (0., bottom),
            (top - p.rim_height, half),
            (top - p.rim_height * 0.15, half * 1.015),
            (top, half),
            (top, rim_inner),
            (top - p.rim_height, rim_inner),
            (p.floor, p.half_at(p.floor) - Vec2::splat(p.wall)),
        ],
        p.squareness,
        Transform::IDENTITY,
    );
    if p.lid_height > 0. {
        let y = top + p.seam;
        a.part(p.hardware, l).housing(
            &[
                (y, half * 1.015),
                (y + p.lid_height * 0.65, half * 1.018),
                (y + p.lid_height, half * 0.94),
            ],
            p.squareness,
            Transform::IDENTITY,
        );
        // A connected rear hinge spans the seam; surfaces remain separated.
        a.part(p.hardware, l).cylinder(
            p.seam * 0.7 + p.lid_height * 0.16,
            p.body.x * 0.27,
            Transform::from_xyz(0., top + p.seam * 0.5, -half.y * 0.96)
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
        if p.lid == 1 {
            let z = p.half_at(p.pedal_height).y;
            let depth = (o.size.z * 0.495 - z).min(o.size.z * 0.13);
            a.box_part(
                Surface::Rubber,
                l,
                Vec3::new(p.body.x * p.pedal_offset, p.pedal_height, z + depth * 0.32),
                Vec3::new(p.body.x * p.pedal_width, p.floor * 0.68, depth),
                p.floor * 0.12,
            );
        } else {
            a.box_part(
                Surface::Ink,
                l,
                Vec3::new(
                    p.body.x * p.pedal_offset,
                    y + p.lid_height + o.size.y * 0.001,
                    0.,
                ),
                Vec3::new(
                    p.body.x * p.pedal_width * 0.65,
                    o.size.y * 0.001,
                    p.body.z * 0.10,
                ),
                0.0001,
            );
        }
    } else if p.lid == 3 {
        // A sloped divider follows the cavity's taper. Its ends penetrate only
        // the wall thickness, so both separated chambers remain real openings.
        let x = half.x * p.divider_x;
        let thickness = p.wall * 0.85;
        let end = |y: f32| {
            let h = p.half_at(y) - Vec2::splat(p.wall * 0.5);
            h.y * (1. - (x / h.x).abs().powf(p.squareness)).powf(1. / p.squareness)
        };
        a.part(p.hardware, l).housing(
            &[
                (p.floor, Vec2::new(thickness * 0.5, end(p.floor))),
                (
                    top - p.rim_height * 0.5,
                    Vec2::new(thickness * 0.5, end(top - p.rim_height * 0.5)),
                ),
            ],
            8.,
            Transform::from_xyz(x, 0., 0.),
        );
    } else if p.lid == 4 {
        for i in 0..p.rib_count {
            let t = p.rib_phase + TAU * i as f32 / p.rib_count as f32;
            let p0 = p.perimeter(t, p.floor * 2.);
            let p1 = p.perimeter(t, top - p.rim_height * 1.15);
            a.part(p.finish, l).rod(p0, p1, p.rib_radius);
        }
    }
}

pub(crate) fn dimensions(r: &mut impl Rng) -> Vec3 {
    let width = r.random_range(0.24..0.46);
    Vec3::new(
        width,
        width * r.random_range(1.05..1.95),
        width * r.random_range(0.74..1.22),
    )
}
