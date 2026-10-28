//! Portable distribution strips. Socket cells, chassis and routed cable ports
//! share one sampled program; electrical layout is coherent within a product.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StripProgram {
    pub height_fraction: f32,
    pub bevel_fraction: f32,
    pub socket_span: f32,
    pub socket_offset: f32,
    pub socket_count: usize,
    pub rows: usize,
    pub socket_yaw: f32,
    pub plate_fraction: f32,
    pub switch: bool,
    pub housing: Surface,
    pub plate: Surface,
    pub end_caps: bool,
}

/// Sample the product before placement, including compact twin-row units.
pub(crate) fn dimensions(r: &mut impl Rng) -> Vec3 {
    if r.random_bool(0.30) {
        Vec3::new(
            r.random_range(0.17..0.30),
            r.random_range(0.029..0.052),
            r.random_range(0.080..0.13),
        )
    } else {
        Vec3::new(
            r.random_range(0.21..0.46),
            r.random_range(0.025..0.046),
            r.random_range(0.048..0.076),
        )
    }
}

pub fn parameters(o: &IndoorObject) -> StripProgram {
    let mut r = stream(o.seed, 0x504f574552535452);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x535452495046494e);
    let rows = if o.size.z >= 0.073 && r.random_bool(0.72) {
        2
    } else {
        1
    };
    let columns = r.random_range(2..=((o.size.x / 0.050).floor() as usize).clamp(2, 8));
    StripProgram {
        height_fraction: r.random_range(0.62..0.88),
        bevel_fraction: r.random_range(0.07..0.22),
        socket_span: r.random_range(0.62..0.80),
        socket_offset: r.random_range(-0.035..0.025),
        socket_count: columns * rows,
        rows,
        // Product-level rotation, not implausibly unrelated pin angles.
        socket_yaw: [0., FRAC_PI_2, PI * 0.25, -PI * 0.25][r.random_range(0..4)],
        plate_fraction: r.random_range(0.76..0.90),
        switch: r.random_bool(0.66),
        housing: if f.random_bool(0.30) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
        plate: if f.random_bool(0.30) {
            Surface::Rubber
        } else {
            Surface::Plastic
        },
        end_caps: r.random_bool(0.48),
    }
}
impl StripProgram {
    pub fn socket_center(&self, s: Vec3, index: usize) -> Vec2 {
        let columns = self.socket_count / self.rows;
        Vec2::new(
            s.x * (self.socket_offset
                + self.socket_span * (((index % columns) as f32 + 0.5) / columns as f32 - 0.5)),
            s.z * (((index / columns) as f32 + 0.5) / self.rows as f32 - 0.5),
        )
    }
    fn plate_width(&self, s: Vec3) -> f32 {
        (s.x * self.socket_span / (self.socket_count / self.rows) as f32)
            .min(s.z / self.rows as f32)
            * self.plate_fraction
    }
    pub fn output_port(&self, s: Vec3, low_voltage: bool) -> Vec3 {
        if low_voltage {
            Vec3::new(
                s.x * 0.35,
                s.y * self.height_fraction * 0.5,
                s.z * 0.5 + 0.001,
            )
        } else {
            let center = self.socket_center(s, 0);
            Vec3::new(center.x, s.y * self.height_fraction + 0.002, center.y)
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
    if p.end_caps {
        // End caps share boundaries with the channel; no stacked coplanar lids.
        let cap = s.x * 0.055;
        a.box_part(
            p.housing,
            l,
            Vec3::Y * h * 0.5,
            Vec3::new(s.x - 2. * cap, h, s.z),
            h * p.bevel_fraction,
        );
        for sign in [-1., 1.] {
            a.box_part(
                p.plate,
                l,
                Vec3::new(sign * (s.x - cap) * 0.5, h * 0.5, 0.),
                Vec3::new(cap, h, s.z),
                h * p.bevel_fraction * 0.45,
            );
        }
    } else {
        a.box_part(
            p.housing,
            l,
            Vec3::Y * h * 0.5,
            s.with_y(h),
            h * p.bevel_fraction,
        );
    }
    let width = p.plate_width(s);
    for i in 0..p.socket_count {
        let center = p.socket_center(s, i);
        a.part(p.plate, l).cylinder(
            width * 0.5,
            0.001,
            Transform::from_xyz(center.x, h + 0.0005, center.y),
        );
        let turn = Quat::from_rotation_y(p.socket_yaw);
        for z in [-1., 1.] {
            let position = Vec3::new(center.x, h + 0.0012, center.y)
                + turn * Vec3::new(0., 0., z * width * 0.22);
            if o.variant.is_multiple_of(4) {
                a.part(Surface::Ink, l).cylinder(
                    width * 0.065,
                    0.0004,
                    Transform::from_translation(position),
                );
            } else {
                let size = Vec3::new(
                    width * if o.variant % 4 == 2 { 0.26 } else { 0.19 },
                    0.0004,
                    width * 0.072,
                );
                a.part(Surface::Ink, l).cuboid(
                    size,
                    0.0001,
                    Transform::from_translation(position).with_rotation(turn),
                );
            }
        }
        if o.variant % 4 >= 2 {
            let position =
                Vec3::new(center.x, h + 0.0012, center.y) + turn * Vec3::new(width * 0.22, 0., 0.);
            a.part(Surface::Ink, l).cuboid(
                Vec3::new(width * 0.08, 0.0004, width * 0.10),
                0.0001,
                Transform::from_translation(position).with_rotation(turn),
            );
        }
    }
    if p.switch {
        let last = p.socket_center(s, p.socket_count / p.rows - 1);
        let left = last.x + width * 0.5 + 0.002;
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
    fn sampled_strip_cells_are_disjoint_bounded_and_finish_independent() {
        use crate::scene::procedural_indoor::objects::{audit, build_object};
        let mut configurations = std::collections::BTreeSet::new();
        let mut shapes = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let mut o = audit::fixture(ObjectKind::PowerStrip, seed);
            o.size = dimensions(&mut stream(seed, 10701));
            let p = parameters(&o);
            configurations.insert((p.rows, p.end_caps, (p.socket_yaw * 100.) as i32));
            shapes.insert((
                (o.size.x * 1000.) as i32,
                (o.size.y * 1000.) as i32,
                (o.size.z * 1000.) as i32,
            ));
            let a = build_object(&o);
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
            let (lo, hi) = a.bounds();
            assert!(lo.y.abs() < 1e-6);
            assert!(hi.y <= o.size.y + 1e-6);
            for i in 0..p.socket_count {
                let center = p.socket_center(o.size, i);
                assert!((center.abs() + Vec2::splat(p.plate_width(o.size) * 0.5))
                    .cmplt(o.size.xz() * 0.5)
                    .all());
                for j in 0..i {
                    assert!((center - p.socket_center(o.size, j)).length() > p.plate_width(o.size));
                }
            }
            let replay = build_object(&o);
            for (key, g) in &a.parts {
                assert_eq!(g.positions, replay.parts[key].positions);
                assert_eq!(g.indices, replay.parts[key].indices);
            }
            o.finish_seed = Some(888);
            let alt = parameters(&o);
            assert_eq!(p.socket_count, alt.socket_count);
            assert_eq!(p.output_port(o.size, false), alt.output_port(o.size, false));
            assert_eq!(p.inlet(o.size), alt.inlet(o.size));
            assert_eq!(p.socket_yaw, alt.socket_yaw);
        }
        assert_eq!(configurations.len(), 16);
        assert!(shapes.len() > 240);
    }

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
