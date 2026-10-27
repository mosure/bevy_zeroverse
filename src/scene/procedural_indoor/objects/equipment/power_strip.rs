//! Portable distribution strips. Cable endpoints derive from the same sampled
//! deck and socket positions as the mesh, not a nominal bounding-box surface.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StripProgram {
    pub height_fraction: f32,
    pub bevel_fraction: f32,
    pub socket_span: f32,
    pub socket_offset: f32,
    pub socket_count: usize,
    pub plate_fraction: f32,
    pub switch: bool,
    pub housing: Surface,
}
pub fn parameters(o: &IndoorObject) -> StripProgram {
    let mut r = stream(o.seed, 0x504f574552535452);
    let old = super::parameters(o);
    StripProgram {
        height_fraction: r.random_range(0.62..0.94),
        bevel_fraction: r.random_range(0.07..0.28),
        socket_span: r.random_range(0.64..0.83),
        socket_offset: r.random_range(-0.055..0.025),
        socket_count: old
            .socket_count
            .min((o.size.x / 0.039).floor().max(3.) as usize),
        plate_fraction: r.random_range(0.62..0.88),
        switch: r.random_bool(0.66),
        housing: old.housing,
    }
}
impl StripProgram {
    fn socket_x(&self, s: Vec3, index: usize) -> f32 {
        s.x * (self.socket_offset
            + self.socket_span * ((index as f32 + 0.5) / self.socket_count as f32 - 0.5))
    }
    pub fn output_port(&self, s: Vec3, low_voltage: bool) -> Vec3 {
        if low_voltage {
            Vec3::new(
                s.x * 0.35,
                s.y * self.height_fraction * 0.5,
                s.z * 0.5 + 0.001,
            )
        } else {
            Vec3::new(self.socket_x(s, 0), s.y * self.height_fraction + 0.002, 0.)
        }
    }
    pub fn inlet(&self, s: Vec3) -> Vec3 {
        Vec3::new(-s.x * 0.5 - 0.001, s.y * self.height_fraction * 0.5, 0.)
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    let h = s.y * p.height_fraction;
    a.box_part(
        p.housing,
        l,
        Vec3::Y * h * 0.5,
        s.with_y(h),
        h * p.bevel_fraction,
    );
    for i in 0..p.socket_count {
        let x = p.socket_x(s, i);
        let width = s.x * p.socket_span / p.socket_count as f32 * 0.84;
        a.box_part(
            Surface::Plastic,
            l,
            Vec3::new(x, h + 0.0005, 0.),
            Vec3::new(width, 0.001, s.z * p.plate_fraction),
            0.001,
        );
        for z in [-1., 1.] {
            let position = Vec3::new(x, h + 0.0012, z * s.z * 0.17);
            if o.variant.is_multiple_of(4) {
                a.part(Surface::Ink, l).cylinder(
                    0.0026,
                    0.0004,
                    Transform::from_translation(position),
                );
            } else {
                a.box_part(
                    Surface::Ink,
                    l,
                    position,
                    Vec3::new(
                        if o.variant % 4 == 2 { 0.010 } else { 0.007 },
                        0.0004,
                        0.0028,
                    ),
                    0.0001,
                );
            }
        }
        if o.variant % 4 >= 2 {
            a.box_part(
                Surface::Ink,
                l,
                Vec3::new(x + 0.008, h + 0.0012, 0.),
                Vec3::new(0.003, 0.0004, 0.004),
                0.0001,
            );
        }
    }
    if p.switch {
        let left =
            p.socket_x(s, p.socket_count - 1) + s.x * p.socket_span / p.socket_count as f32 * 0.42;
        let right = s.x * 0.5 - h * p.bevel_fraction;
        if right - left > 0.009 {
            a.box_part(
                Surface::Rubber,
                l,
                Vec3::new((left + right) * 0.5, h + 0.001, 0.),
                Vec3::new((right - left) * 0.7, 0.002, s.z * 0.36),
                0.001,
            );
        }
    }
    let inlet = p.inlet(s) + Vec3::X * 0.0009;
    a.box_part(
        Surface::Ink,
        l,
        inlet,
        Vec3::new(0.0008, h * 0.25, s.z * 0.24),
        0.0003,
    );
    let pd = p.output_port(s, true) - Vec3::Z * 0.0009;
    a.box_part(Surface::Ink, l, pd, Vec3::new(0.008, 0.003, 0.0008), 0.0005);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routed_power_endpoints_follow_actual_socket_decks() {
        for seed in 0..128 {
            let o = crate::scene::procedural_indoor::objects::audit::fixture(
                ObjectKind::PowerStrip,
                seed,
            );
            let p = parameters(&o);
            let mut a = Assembly::default();
            build(&mut a, &o);
            let port = p.output_port(o.size, false);
            let q = port.xz();
            let mut top = f32::NEG_INFINITY;
            for g in a.parts.values() {
                for tri in g.indices.as_chunks::<3>().0 {
                    let v = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let area = (v[1].xz() - v[0].xz()).perp_dot(v[2].xz() - v[0].xz());
                    if area.abs() < 1e-10 {
                        continue;
                    }
                    let u = (v[1].xz() - q).perp_dot(v[2].xz() - q) / area;
                    let w = (v[2].xz() - q).perp_dot(v[0].xz() - q) / area;
                    let t = 1. - u - w;
                    if u >= -1e-5 && w >= -1e-5 && t >= -1e-5 {
                        top = top.max(u * v[0].y + w * v[1].y + t * v[2].y);
                    }
                }
            }
            assert!(
                (0.0003..0.0021).contains(&(port.y - top)),
                "{seed}: port={port:?}, mesh={top}"
            );
            assert_eq!(
                super::super::power_port(&o, false).unwrap(),
                o.transform().transform_point(port)
            );
        }
    }
}
