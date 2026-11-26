//! Exact Euclidean lookup for grid slots outside the lighting domain. Keep the
//! original probe index on distance ties so the traced random sequence replays.
use bevy::math::Vec3;

const LEAF: usize = 12;

pub(super) struct ProbeSources<'a> {
    points: &'a [Vec3],
    indices: Vec<usize>,
    partitioned: bool,
}

impl<'a> ProbeSources<'a> {
    pub(super) fn new(points: &'a [Vec3], valid: &[bool]) -> Self {
        let indices: Vec<_> = (0..points.len()).filter(|&i| valid[i]).collect();
        let queries = points.len() - indices.len();
        Self::from_indices(points, indices, queries)
    }

    pub(super) fn from_indices(
        points: &'a [Vec3],
        mut indices: Vec<usize>,
        queries: usize,
    ) -> Self {
        // A handful of holes costs less to scan than to build a tree.
        let partitioned = indices.len() > LEAF && queries > indices.len().ilog2() as usize;
        if partitioned {
            partition(points, &mut indices, 0);
        }
        Self {
            points,
            indices,
            partitioned,
        }
    }

    pub(super) fn nearest(&self, point: Vec3) -> Option<usize> {
        let mut best = (f32::INFINITY, usize::MAX);
        if self.partitioned {
            self.search(&self.indices, point, 0, &mut best);
        } else {
            self.scan(&self.indices, point, &mut best);
        }
        (best.1 != usize::MAX).then_some(best.1)
    }

    fn scan(&self, indices: &[usize], p: Vec3, best: &mut (f32, usize)) {
        for &i in indices {
            let candidate = (self.points[i].distance_squared(p), i);
            if candidate < *best {
                *best = candidate;
            }
        }
    }

    fn search(&self, indices: &[usize], p: Vec3, axis: usize, best: &mut (f32, usize)) {
        if indices.len() <= LEAF {
            self.scan(indices, p, best);
            return;
        }
        let (left, right) = indices.split_at(indices.len() / 2);
        let (median, right) = right.split_first().unwrap();
        self.scan(std::slice::from_ref(median), p, best);
        let delta = p[axis] - self.points[*median][axis];
        let (near, far) = if delta < 0. {
            (left, right)
        } else {
            (right, left)
        };
        let next = (axis + 1) % 3;
        self.search(near, p, next, best);
        // Both halves may include the dividing coordinate. Equality must visit
        // the far half, which can have a smaller original index at the same distance.
        if delta * delta <= best.0 {
            self.search(far, p, next, best);
        }
    }
}

fn partition(points: &[Vec3], indices: &mut [usize], axis: usize) {
    if indices.len() <= LEAF {
        return;
    }
    let mid = indices.len() / 2;
    let (left, _, right) = indices.select_nth_unstable_by(mid, |&a, &b| {
        points[a][axis].total_cmp(&points[b][axis]).then(a.cmp(&b))
    });
    let next = (axis + 1) % 3;
    partition(points, left, next);
    partition(points, right, next);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(points: &[Vec3], valid: &[bool], p: Vec3) -> Option<usize> {
        (0..points.len()).filter(|&i| valid[i]).min_by(|&a, &b| {
            points[a]
                .distance_squared(p)
                .total_cmp(&points[b].distance_squared(p))
        })
    }

    #[test]
    fn relocation_matches_exhaustive_squared_metres_and_original_index() {
        let points: Vec<_> = (0..770)
            .map(|i| {
                Vec3::new(
                    ((i % 11) as f32 - 5.) * 0.375,
                    ((i / 11 % 10) as f32 - 5.) * 0.73,
                    ((i / 110) as f32 - 3.) * 0.375,
                )
            })
            .collect();
        for pattern in 0..7 {
            let mask: Vec<_> = (0..points.len())
                .map(|i| match pattern {
                    0 => false,
                    1 => i == 37,
                    2 => i != 2,
                    3 => (i / 11 + i % 11) % 2 == 1,
                    4 => i % 11 == 0,
                    5 => i % 43 == 0,
                    _ => (i as u32).wrapping_mul(0x9e3779b1).count_ones() < 15,
                })
                .collect();
            let tree = ProbeSources::new(&points, &mask);
            for &point in &points {
                assert_eq!(tree.nearest(point), brute(&points, &mask, point));
                let tie = point + Vec3::new(0.1875, 0., -0.1875);
                assert_eq!(tree.nearest(tie), brute(&points, &mask, tie));
            }
        }
    }
}
