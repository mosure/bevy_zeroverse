//! Narrow phase for thin hinged surfaces next to a stationary carcass.
//! The convex hull of a triangle's two endpoint poses, expanded by the arc
//! sagitta, contains its entire continuous rigid sweep.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct TriangleHull {
    pub points: [Vec3; 6],
    pub error: f32,
}
impl TriangleHull {
    pub(super) fn intersects(self, triangle: [Vec3; 3]) -> bool {
        let separated = |axis: Vec3| {
            let a = triangle.map(|p| p.dot(axis));
            let b = self.points.map(|p| p.dot(axis));
            let min = |p: &[f32]| p.iter().copied().fold(f32::INFINITY, f32::min);
            let max = |p: &[f32]| p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let pad = (self.error + 1e-5) * axis.length();
            max(&a) < min(&b) - pad || min(&a) > max(&b) + pad
        };
        let edges = [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ];
        if separated(edges[0].cross(edges[1])) {
            return false;
        }
        // All vertex pairs/triples contain the hull's edges/face normals. Extra
        // axes are safe: separation on *any* axis certifies no intersection.
        // Six vertices keep this bounded without allocating a hull mesh.
        for i in 0..6 {
            for j in i + 1..6 {
                let edge = self.points[j] - self.points[i];
                if edges.iter().any(|e| separated(edge.cross(*e))) {
                    return false;
                }
                for k in j + 1..6 {
                    if separated(edge.cross(self.points[k] - self.points[i])) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expanded_endpoint_hulls_contain_curved_motion_and_exclude_empty_hinge_wedges() {
        let points = [
            Vec3::new(0., 0., 0.014),
            Vec3::new(1.5, 0., 0.014),
            Vec3::new(0., 1.8, 0.014),
        ];
        let angle = 0.04_f32;
        let q = Quat::from_rotation_y(-angle);
        let hull = TriangleHull {
            points: [
                points[0],
                points[1],
                points[2],
                q * points[0],
                q * points[1],
                q * points[2],
            ],
            error: 2.4 * (1. - (angle * 0.5).cos()) + 0.0001,
        };
        for step in 0..=100 {
            let q = Quat::from_rotation_y(-angle * step as f32 / 100.);
            assert!(hull.intersects(points.map(|p| q * p)));
        }
        assert!(
            !hull.intersects([
                Vec3::new(0.001, 0., 0.),
                Vec3::new(0.001, 1.8, 0.),
                Vec3::new(0.01, 0., 0.),
            ]),
            "a carcass behind a forward-opening leaf is not inside its sweep"
        );
    }
}
