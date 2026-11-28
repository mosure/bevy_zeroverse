//! Laminated top with open, chamfered bench-dog bores. The top, underside and
//! bore wall share the same rings. No invisible slab closes a through-hole.
use super::*;

fn xz(x: f32, z: f32, y: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn flats(g: &mut Geometry, corners: [Vec2; 4], y: f32, t: f32, chart: Vec2) {
    for (level, n) in [(y, Vec3::Y), (y - t, -Vec3::Y)] {
        g.stock_quad(
            corners.map(|v| xz(v.x, v.y, level)),
            n,
            Vec3::X,
            Some(chart),
        );
    }
}

fn rect(lo: Vec2, hi: Vec2) -> [Vec2; 4] {
    [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
}

/// One angular stencil serves every bore on a bench. The outer points are fractions
/// of a cell so adjacent cells meet exactly at their shared rectangle edges.
struct BoreRing([(Vec2, Vec2); 16]);

impl BoreRing {
    fn new(pitch: f32, strip_depth: f32) -> Self {
        let half = Vec2::new(pitch, strip_depth) * 0.5;
        let corners = rect(-half, half);
        Self(std::array::from_fn(|i| {
            let edge = i / 4;
            let a = corners[edge];
            let b = corners[(edge + 1) % 4];
            let t = (i % 4) as f32 * 0.25;
            let angle = a.y.atan2(a.x) + t * a.perp_dot(b).atan2(a.dot(b));
            let (sn, cs) = angle.sin_cos();
            let dir = Vec2::new(cs, sn);
            let outer = if i.is_multiple_of(4) {
                a
            } else if edge.is_multiple_of(2) {
                Vec2::new(cs * a.y / sn, a.y)
            } else {
                Vec2::new(a.x, sn * a.x / cs)
            };
            ((outer / half + Vec2::ONE) * 0.5, dir)
        }))
    }

    fn drill(&self, g: &mut Geometry, cell: Rect, radius: f32, y: f32, t: f32, chart: Vec2) {
        let points = self.0.map(|(ext, dir)| {
            (
                cell.min + cell.size() * ext,
                cell.center() + dir * (radius + BORE_CHAMFER_M),
                cell.center() + dir * radius,
            )
        });
        for (i, from) in points.iter().enumerate() {
            let to = points[(i + 1) % points.len()];
            for (level, n, a, b) in [(y, Vec3::Y, from.1, to.1), (y - t, -Vec3::Y, from.2, to.2)] {
                g.stock_quad(
                    [from.0, to.0, b, a].map(|v| xz(v.x, v.y, level)),
                    n,
                    Vec3::X,
                    Some(chart),
                );
            }
            let inward = cell.center() - (from.2 + to.2) * 0.5;
            let n = Vec3::new(inward.x, 0., inward.y).normalize();
            let lift = |v: Vec2, h: f32| xz(v.x, v.y, h);
            g.metric_quad(
                [
                    lift(from.1, y),
                    lift(to.1, y),
                    lift(to.2, y - BORE_CHAMFER_M),
                    lift(from.2, y - BORE_CHAMFER_M),
                ],
                n + Vec3::Y,
            );
            g.metric_quad(
                [
                    lift(from.2, y - BORE_CHAMFER_M),
                    lift(to.2, y - BORE_CHAMFER_M),
                    lift(to.2, y - t),
                    lift(from.2, y - t),
                ],
                n,
            );
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &Bench, cuts: &mut TimberCuts) {
    let l = o.kind.class_name();
    let deck = p.deck(o);
    let y = o.size.y;
    if p.surface != WorkingSurface::DogHoles {
        let mat = if p.surface == WorkingSurface::CuttingMat {
            0.004
        } else {
            0.
        };
        let d = deck.size();
        let c = deck.center();
        a.board(
            cuts,
            (Surface::Wood, l),
            xz(c.x, c.y, y - (p.stock_m + mat) * 0.5),
            Vec3::new(d.x, p.stock_m - mat, d.y),
            0.0022,
            GrainAxis::X,
        );
        if mat > 0. {
            a.box_part(
                Surface::Rubber,
                crate::scene::procedural_indoor::materials::cutting_mat::LABEL,
                xz(c.x, c.y, y - mat * 0.5),
                Vec3::new(d.x - 0.002, mat, d.y - 0.002),
                0.00065,
            );
        }
        return;
    }
    let g = a.part(Surface::Wood, l);
    let n = p.bores_per_row(o);
    let ring = BoreRing::new(p.bore_pitch_m, deck.height() / p.laminations as f32);
    for i in 0..p.laminations {
        let z = |i: usize| deck.min.y + deck.height() * (i as f32 / p.laminations as f32);
        let (left, right) = (deck.min.x, deck.max.x);
        let (z0, z1) = (z(i), z(i + 1));
        let chart = cuts.origin();
        let flat = |g: &mut Geometry, x0: f32, x1: f32| {
            flats(
                g,
                rect(Vec2::new(x0, z0), Vec2::new(x1, z1)),
                y,
                p.stock_m,
                chart,
            );
        };
        if p.bored_strip(i) {
            let start = -(n as f32) * p.bore_pitch_m * 0.5;
            flat(g, left, start);
            for j in 0..n {
                ring.drill(
                    g,
                    Rect::from_corners(
                        Vec2::new(start + j as f32 * p.bore_pitch_m, z0),
                        Vec2::new(start + (j + 1) as f32 * p.bore_pitch_m, z1),
                    ),
                    p.bore_radius_m,
                    y,
                    p.stock_m,
                    chart,
                );
            }
            flat(g, start + n as f32 * p.bore_pitch_m, right);
        } else {
            flat(g, left, right);
        }
        for (x, n) in [(left, -Vec3::X), (right, Vec3::X)] {
            g.metric_quad(
                [
                    xz(x, z0, y),
                    xz(x, z1, y),
                    xz(x, z1, y - p.stock_m),
                    xz(x, z0, y - p.stock_m),
                ],
                n,
            );
        }
        for (outer, z, n) in [
            (i == 0, z0, -Vec3::Z),
            (i + 1 == p.laminations, z1, Vec3::Z),
        ] {
            if outer {
                g.stock_quad(
                    [
                        xz(left, z, y),
                        xz(right, z, y),
                        xz(right, z, y - p.stock_m),
                        xz(left, z, y - p.stock_m),
                    ],
                    n,
                    Vec3::X,
                    Some(chart),
                );
            }
        }
    }
}
