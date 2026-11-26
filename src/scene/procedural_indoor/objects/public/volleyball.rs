//! Removable sleeve-mounted posts, protective pads, tension cables and a net
//! made of open cells. All strands are batched by material and cast real shadows.
use super::*;
use crate::scene::procedural_indoor::{materials::sport, program::court};

const CLASS: &str = "sports_equipment";
const DEPTH: f32 = 1.0;
const HEAD: f32 = 0.070;
const FOOT: f32 = 0.050;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Net {
    pub post_center: f32,
    pub half_web: f32,
    pub center_height: f32,
    pub sag: f32,
    pub pad_height: f32,
    pub pad_radius: f32,
    pub strand_radius: f32,
    pub columns: usize,
    pub rows: usize,
    pub pad_slot: usize,
}
pub(super) fn dimensions(seed: u64) -> Vec3 {
    court::VolleyballCourt::fit(seed, court::room_size(seed))
        .unwrap()
        .net_size()
}
impl Net {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"VBWeave "));
        let pad = o.size.z * 0.5;
        let post = o.size.x * 0.5 - pad;
        let half_web = post - r.random_range(0.415..0.50);
        Self {
            post_center: post,
            half_web,
            center_height: o.size.y - 0.17,
            sag: r.random_range(0.006..0.020),
            pad_height: r.random_range(1.42..1.85),
            pad_radius: pad,
            strand_radius: r.random_range(0.00135..0.0022),
            columns: (2. * half_web / 0.10).round() as usize,
            rows: 9,
            pad_slot: sport::pad_slot(&mut r),
        }
    }
    /// Broad visibility surfaces for camera proposals. The open web between
    /// tapes transmits most rays. Capture itself traces the constructed cords.
    pub(crate) fn view_boxes(self, mut emit: impl FnMut(Vec3, Vec3)) {
        for x in [-self.post_center, self.post_center] {
            let max = self.center_height + 0.17;
            emit(
                Vec3::new(x, self.pad_height * 0.5, 0.),
                Vec3::new(self.pad_radius, self.pad_height * 0.5, self.pad_radius),
            );
            emit(
                Vec3::new(x, (max + self.pad_height) * 0.5, 0.),
                Vec3::new(0.037, (max - self.pad_height) * 0.5, 0.037),
            );
        }
        for (drop, height) in [(0., HEAD), (DEPTH - FOOT, FOOT)] {
            for i in 0..4 {
                let x = self.half_web * ((i as f32 + 0.5) * 0.5 - 1.);
                emit(
                    Vec3::new(x, self.top(x) - drop - height * 0.5, 0.),
                    Vec3::new(self.half_web * 0.25, height * 0.5, 0.0028),
                );
            }
        }
    }
    fn top(self, x: f32) -> f32 {
        self.center_height + self.sag * (x / self.half_web).powi(2)
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let n = Net::for_object(o);
    let pad = format!("{CLASS}#finish{}", n.pad_slot);
    for side in [-1., 1.] {
        let x = side * n.post_center;
        let pos = |y| Transform::from_xyz(x, y, 0.);
        // A metal floor socket, a padded lower post and an exposed telescopic head.
        a.part(Surface::Metal, CLASS)
            .cylinder(n.pad_radius, 0.014, pos(0.007));
        a.part(Surface::Leather, &pad).cylinder(
            n.pad_radius,
            n.pad_height - 0.014,
            pos((n.pad_height + 0.014) * 0.5),
        );
        a.part(Surface::Metal, CLASS).cylinder(
            0.037,
            o.size.y - n.pad_height,
            pos((o.size.y + n.pad_height) * 0.5),
        );
        for (drop, radius) in [(0.022, 0.004), (DEPTH - 0.022, 0.003)] {
            let y = n.top(n.half_web) - drop;
            a.part(Surface::Metal, CLASS).tube(
                &[
                    Vec3::new(x, y + 0.012, 0.),
                    Vec3::new(side * n.half_web, y, 0.),
                ],
                radius,
                5,
            );
        }
    }
    let cord = a.part(Surface::Rubber, sport::CORD);
    for i in 0..=n.columns {
        let x = (2. * i as f32 / n.columns as f32 - 1.) * n.half_web;
        let y = n.top(x);
        cord.tube(
            &[
                Vec3::new(x, y - DEPTH + FOOT, 0.),
                Vec3::new(x, y - HEAD, 0.),
            ],
            n.strand_radius,
            4,
        );
    }
    const SPANS: usize = 12;
    let row = |drop: f32| {
        std::array::from_fn::<_, { SPANS + 1 }, _>(|i| {
            let x = (2. * i as f32 / SPANS as f32 - 1.) * n.half_web;
            Vec3::new(x, n.top(x) - drop, 0.)
        })
    };
    for j in 1..n.rows {
        let drop = HEAD + (DEPTH - HEAD - FOOT) * j as f32 / n.rows as f32;
        // The crossing cords meet with a small depth offset, like a woven net.
        let path = row(drop).map(|p| p + Vec3::Z * n.strand_radius);
        cord.tube(&path, n.strand_radius, 4);
    }
    let canvas = a.part(Surface::Fabric, sport::TAPE);
    for (drop, h) in [(0., HEAD), (DEPTH - FOOT, FOOT)] {
        let path = row(drop);
        let front = Vec3::Z * 0.0028;
        for ends in path.windows(2) {
            let (a, b) = (ends[0], ends[1]);
            let down = Vec3::Y * h;
            for (shift, out) in [(front, Vec3::Z), (-front, -Vec3::Z)] {
                canvas.metric_quad(
                    [a + shift, a - down + shift, b - down + shift, b + shift],
                    out,
                );
            }
            canvas.metric_quad([a + front, b + front, b - front, a - front], Vec3::Y);
            canvas.metric_quad(
                [
                    a - down + front,
                    b - down + front,
                    b - down - front,
                    a - down - front,
                ],
                -Vec3::Y,
            );
        }
        for (edge, out) in [(path[0], -Vec3::X), (path[SPANS], Vec3::X)] {
            let down = Vec3::Y * h;
            canvas.metric_quad(
                [
                    edge + front,
                    edge - down + front,
                    edge - down - front,
                    edge - front,
                ],
                out,
            );
        }
    }
}
