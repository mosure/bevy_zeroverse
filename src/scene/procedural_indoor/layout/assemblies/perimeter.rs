//! Arc-length seating on the actual convex tabletop, including curved/tapered ends.
use super::*;

pub(super) struct SeatPath {
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
}
