//! Ownership of the end surface where finite-thickness wall runs meet.
use super::*;

pub(super) struct Joint {
    pub(super) contact: f32,
    pub(super) roof_slope: f32,
    /// Intervals owned by a facing return at the mouth of a narrow recess.
    covered: Vec<Vec2>,
}

impl Joint {
    pub(super) fn new(
        outline: &[Vec2],
        edge: usize,
        at_end: bool,
        height_at: &impl Fn(Vec2) -> f32,
        depth_at: &impl Fn(usize) -> f32,
    ) -> Self {
        let n = outline.len();
        let next = (edge + 1) % n;
        let prev = (edge + n - 1) % n;
        let along = (outline[next] - outline[edge]).normalize();
        let out = -along.perp();
        let (corner, neighbor, far, sign) = if at_end {
            (next, next, (next + 1) % n, 1.)
        } else {
            (edge, prev, prev, -1.)
        };
        let p = outline[corner];
        let run = outline[far] - p;
        let dir = run.normalize();
        let travel = out.dot(dir);
        let into = -out.dot(sign * dir.perp());
        let depth = depth_at(edge);
        // The cap starts inside the adjacent prism at an inside corner. For a
        // square return the ray lies exactly on the neighbour's finished face.
        if travel >= -1e-6 && into >= -1e-6 {
            let mut contact = depth;
            if travel > 1e-6 {
                contact = contact.min(run.length() / travel);
            }
            if into > 1e-6 {
                contact = contact.min(depth_at(neighbor) / into);
            }
            return Self {
                contact,
                roof_slope: (height_at(outline[far]) - height_at(p)) * travel / run.length(),
                covered: Vec::new(),
            };
        }

        let mut covered = Vec::new();
        // At the mouth of a narrow notch opposing convex returns can terminate
        // in the very same plane. The taller end owns the overlap throughout;
        // equal heights tie by edge. Both keep their independently exposed tips.
        for (j, (a, b)) in polygon::edges(outline).enumerate() {
            if j == edge || j == neighbor {
                continue;
            }
            let (q, other_corner) = if at_end { (a, j) } else { (b, (j + 1) % n) };
            let between = q - p;
            if between.dot(along).abs() > 1e-5 {
                continue;
            }
            let distance = between.dot(out);
            if distance <= 0. || distance >= depth + depth_at(j) {
                continue;
            }
            let other = (b - a).normalize();
            if along.dot(other) >= 0. || along.perp_dot(other).abs() > 1e-6 {
                continue;
            }
            let incoming = q - outline[(other_corner + n - 1) % n];
            let outgoing = outline[(other_corner + 1) % n] - q;
            if incoming.perp_dot(outgoing) <= 0. {
                continue;
            }
            // Valid openings have a solid margin at the end of the run.
            let (h, other_h) = (height_at(p), height_at(q));
            if other_h > h || (other_h == h && j < edge) {
                covered.push(Vec2::new(
                    (distance - depth_at(j)).max(0.),
                    distance.min(depth),
                ));
            }
        }
        covered.sort_unstable_by(|a, b| a.x.total_cmp(&b.x));
        Self {
            contact: 0.,
            roof_slope: 0.,
            covered,
        }
    }

    pub(super) fn exposed_depths(&self, depth: f32, mut emit: impl FnMut(f32, f32)) {
        let mut from = self.contact;
        for interval in &self.covered {
            if interval.x > from {
                emit(from, interval.x);
            }
            from = from.max(interval.y);
        }
        if from < depth {
            emit(from, depth);
        }
    }
}
