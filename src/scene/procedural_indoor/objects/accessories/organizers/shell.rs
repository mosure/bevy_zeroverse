//! Formed side cheeks with a lowered pickup end and standoffs between actual tiers.
use super::*;

#[derive(Debug, serde::Serialize)]
pub struct TrayProfile {
    pub front_ratio: f32,
    pub slope_start: f32,
    pub slope_end: f32,
    pub finger_opening: f32,
    pub lip_height_m: f32,
}

impl TrayProfile {
    pub(super) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"TrayRims"));
        Self {
            front_ratio: if r.random_bool(0.16) {
                1.
            } else {
                r.random_range(0.14..0.73)
            },
            slope_start: r.random_range(0.12..0.52),
            slope_end: r.random_range(0.64..0.91),
            finger_opening: r.random_range(0.28..0.78),
            lip_height_m: if r.random_bool(0.65) {
                r.random_range(0.003..0.014)
            } else {
                0.
            },
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &OrganizerProgram, b: &Bay) {
        let label = o.kind.class_name();
        let bed = b.base + b.t;
        let z = |v: f32| (v - 0.5) * o.size.z;
        let outline = [
            (z(0.), b.wall_h),
            (z(self.slope_start), b.wall_h),
            (z(self.slope_end), b.wall_h * self.front_ratio),
            (z(1.), b.wall_h * self.front_ratio),
        ];
        let g = a.part(p.finish, label);
        for sign in [-1., 1.] {
            let x = b.x + sign * (b.width * 0.5 - b.t * 0.6);
            let side = |edge: f32, y: f32, z: f32| Vec3::new(x + edge * b.t * 0.5, bed + y, z);
            for &[(za, ha), (zb, hb)] in outline.array_windows::<2>() {
                for edge in [-1., 1.] {
                    g.metric_quad(
                        [
                            side(edge, 0., za),
                            side(edge, ha, za),
                            side(edge, hb, zb),
                            side(edge, 0., zb),
                        ],
                        edge * Vec3::X,
                    );
                }
                g.metric_quad(
                    [
                        side(-1., ha, za),
                        side(1., ha, za),
                        side(1., hb, zb),
                        side(-1., hb, zb),
                    ],
                    Vec3::Y,
                );
            }
            for (zi, hi) in [outline[0], outline[3]] {
                g.metric_quad(
                    [
                        side(-1., 0., zi),
                        side(1., 0., zi),
                        side(1., hi, zi),
                        side(-1., hi, zi),
                    ],
                    zi.signum() * Vec3::Z,
                );
            }
            if !b.last {
                // Standoffs land on the flat ends of the cheek. They span
                // only the clear air between it and the underside of the next bed.
                for (zi, h) in [(-1., b.wall_h), (1., b.wall_h * self.front_ratio)] {
                    let bottom = bed + h;
                    let gap = b.base + b.height - bottom;
                    let center = Vec3::new(x, bottom + gap * 0.5, zi * (o.size.z * 0.5 - 2. * b.t));
                    g.cuboid(
                        Vec3::new(0.72 * b.t, gap, 1.3 * b.t),
                        0.,
                        Transform::from_translation(center),
                    );
                }
            }
            if self.lip_height_m > 0. {
                let clear = b.width - 2.2 * b.t;
                let span = clear * (1. - self.finger_opening) * 0.5;
                let h = self.lip_height_m.min(b.wall_h * self.front_ratio);
                let center = Vec3::new(
                    b.x + sign * (clear - span) * 0.5,
                    bed + h * 0.5,
                    z(1.) - b.t * 0.5,
                );
                g.cuboid(
                    Vec3::new(span, h, b.t),
                    0.12 * b.t,
                    Transform::from_translation(center),
                );
            }
        }
    }
}
