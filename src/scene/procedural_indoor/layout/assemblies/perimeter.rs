//! Arc-length seating on the actual convex tabletop, including curved/tapered ends.
use super::*;

pub(in super::super) struct SeatPath {
    points: Vec<Vec2>,
    lengths: Vec<f32>,
    pub length: f32,
}
impl SeatPath {
    pub fn new(table: &IndoorObject, setback: f32) -> Self {
        let top = crate::scene::procedural_indoor::objects::tables::outline(table);
        let points: Vec<_> = (0..top.len())
            .map(|i| {
                let before = (top[i] - top[(i + top.len() - 1) % top.len()]).normalize();
                let after = (top[(i + 1) % top.len()] - top[i]).normalize();
                let normal = -after.perp();
                let bisector = -(before + after).normalize().perp();
                top[i] + bisector * setback / bisector.dot(normal).max(0.20)
            })
            .collect();
        let lengths: Vec<_> = (0..points.len())
            .map(|i| points[i].distance(points[(i + 1) % points.len()]))
            .collect();
        let length = lengths.iter().sum();
        Self {
            points,
            lengths,
            length,
        }
    }

    /// Position and outward-facing local +Z (the chair faces along local -Z).
    pub fn sample(&self, fraction: f32) -> (Vec2, f32) {
        let mut along = fraction.rem_euclid(1.) * self.length;
        for (i, &length) in self.lengths.iter().enumerate() {
            if along <= length || i + 1 == self.points.len() {
                let edge = self.points[(i + 1) % self.points.len()] - self.points[i];
                let point = self.points[i] + edge * (along / length).clamp(0., 1.);
                let normal = -edge.normalize().perp();
                return (point, normal.x.atan2(normal.y));
            }
            along -= length;
        }
        unreachable!("a table has a nonempty perimeter")
    }

    /// Intersect a line across a convex seating perimeter. This also finds the
    /// actual chair position and edge normal on boat-shaped and oval boards.
    pub(super) fn on_side(&self, axis: usize, sign: f32, along: f32) -> Option<(Vec2, f32)> {
        let cross = 1 - axis;
        let mut outermost = f32::NEG_INFINITY;
        let mut seat = None;
        for (a, b) in self.points.iter().zip(self.points.iter().cycle().skip(1)) {
            let edge = *b - *a;
            if edge[cross].abs() < 1e-6 {
                continue;
            }
            let t = (along - a[cross]) / edge[cross];
            if (0.0..=1.0).contains(&t) {
                let p = *a + edge * t;
                if sign * p[axis] > outermost {
                    outermost = sign * p[axis];
                    let outward = -edge.perp();
                    seat = Some((p, outward.x.atan2(outward.y)));
                }
            }
        }
        seat
    }
}
