//! Room-side finish, exterior skin and exposed returns of a punched wall.
//! Each opening keeps its metric jamb depth. A butt end buried in the next
//! wall has no visible face, including where it meets that wall's lining.
use super::*;
mod joint;
use architecture::facade::{solid_rectangles, WindowOpening};
use joint::Joint;

pub(super) struct WallSection {
    frame: Transform,
    span: f32,
    heights: Vec2,
    depth: f32,
    joints: [Joint; 2],
}

impl WallSection {
    pub(super) fn new(
        outline: &[Vec2],
        edge: usize,
        height_at: impl Fn(Vec2) -> f32,
        depth_at: impl Fn(usize) -> f32,
    ) -> Self {
        let n = outline.len();
        let next = (edge + 1) % n;
        let start = outline[edge];
        let end = outline[next];
        Self {
            frame: polygon::wall_frame(start, end),
            span: start.distance(end),
            heights: Vec2::new(height_at(start), height_at(end)),
            depth: depth_at(edge),
            joints: [false, true]
                .map(|at_end| Joint::new(outline, edge, at_end, &height_at, &depth_at)),
        }
    }

    pub(super) fn build(
        &self,
        holes: &[WindowOpening],
        finish_line: f32,
        mut emit: impl FnMut(&[Vec3], Vec3, bool),
    ) {
        let half = self.span * 0.5;
        let slope = (self.heights.y - self.heights.x) / self.span;
        let mid = self.heights.element_sum() * 0.5;
        let front = self.frame.rotation * Vec3::Z;
        let at = |p: Vec2, d: f32| self.frame.transform_point(p.extend(-d));
        let eq = |a: f32, b: f32| (a - b).abs() < 1e-5;

        for (lo, hi) in solid_rectangles(
            Vec2::new(-half, 0.),
            Vec2::new(half, self.heights.max_element()),
            holes,
        ) {
            let rect = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
            let poly = polygon::clip(&rect, Vec2::new(slope, -1.), -mid);
            if poly.len() < 3 {
                continue;
            }
            let face = |p: &[Vec2], d| p.iter().map(|&v| at(v, d)).collect::<Vec<_>>();
            emit(&face(&poly, self.depth), -front, false);
            if lo.y >= finish_line {
                emit(&face(&poly, 0.), front, true);
            } else if hi.y > finish_line {
                emit(
                    &face(&polygon::clip(&poly, Vec2::Y, finish_line), 0.),
                    front,
                    true,
                );
            }
            for (a, b) in polygon::edges(&poly) {
                let direction = b - a;
                let normal = self.frame.rotation * (-direction.perp()).extend(0.);
                let mut reveal = |a: Vec2, b: Vec2, from: f32, to: f32| {
                    if from < to && a != b {
                        emit(
                            &[at(a, from), at(b, from), at(b, to), at(a, to)],
                            normal,
                            false,
                        );
                    }
                };
                if a.x == b.x && (eq(a.x, -half) || eq(a.x, half)) {
                    let end = usize::from(a.x > 0.);
                    let joint = &self.joints[end];
                    joint.exposed_depths(self.depth, |from, to| reveal(a, b, from, to));
                    let roof = self.heights[end];
                    if joint.contact > 0.
                        && joint.roof_slope < 0.
                        && a.y.max(b.y) > roof + joint.roof_slope * joint.contact
                    {
                        // Above a falling roof the last strip of a butt end is
                        // exposed. Clip in (depth, elevation) coordinates.
                        let rect = [
                            Vec2::new(0., a.y),
                            Vec2::new(joint.contact, a.y),
                            Vec2::new(joint.contact, b.y),
                            Vec2::new(0., b.y),
                        ];
                        let wedge = polygon::clip(&rect, Vec2::new(-joint.roof_slope, 1.), roof);
                        let points: Vec<_> = wedge
                            .into_iter()
                            .map(|p| at(Vec2::new(a.x, p.y), p.x))
                            .collect();
                        emit(&points, normal, false);
                    }
                } else if (a.y == 0. && b.y == 0.)
                    || (eq(a.y, mid + slope * a.x) && eq(b.y, mid + slope * b.x))
                {
                    reveal(a, b, 0., self.depth);
                } else {
                    // A merged face below several windows has a continuous top
                    // edge. Only the spans actually underneath an opening are sills.
                    let (axis, fixed) = if a.x == b.x { (1, 0) } else { (0, 1) };
                    for h in holes {
                        let boundary = if (-direction.perp())[fixed] > 0. {
                            h.min[fixed]
                        } else {
                            h.max[fixed]
                        };
                        if !eq(a[fixed], boundary) || !eq(b[fixed], boundary) {
                            continue;
                        }
                        let mut u = a;
                        let mut v = b;
                        u[axis] = a[axis].min(b[axis]).max(h.min[axis]);
                        v[axis] = a[axis].max(b[axis]).min(h.max[axis]);
                        if v[axis] > u[axis] {
                            reveal(u, v, 0., self.depth);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
