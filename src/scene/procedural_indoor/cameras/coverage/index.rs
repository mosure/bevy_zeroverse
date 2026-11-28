//! Conservative broad phase only: camera decisions retain the scalar OBB test,
//! original collider order for ties, and the complete candidate/ray population.
use super::Collider;
use bevy::{math::Vec3A, prelude::*};

struct Bounds {
    lo: Vec3A,
    hi: Vec3A,
}
impl Bounds {
    fn collider(c: &Collider) -> Self {
        let rotation = Mat3::from_quat(c.inverse.inverse());
        let extent = rotation.x_axis.abs() * c.half.x
            + rotation.y_axis.abs() * c.half.y
            + rotation.z_axis.abs() * c.half.z;
        // Only the broad phase expands. Cover rotation/slab roundoff, including
        // thin boxes and rays on a face; never change narrow-phase clearances.
        let padding = (c.center.abs() + extent + Vec3::ONE) * 0.0001;
        Self {
            lo: (c.center - extent - padding).into(),
            hi: (c.center + extent + padding).into(),
        }
    }
    #[inline]
    fn intersects(&self, ray: &SlabRay, limit: f32) -> bool {
        if ray.finite_inverse {
            let a = (self.lo - ray.origin) * ray.inverse;
            let b = (self.hi - ray.origin) * ray.inverse;
            let near = a.min(b).max_element().max(0.0);
            let far = a.max(b).min_element().min(limit);
            // Multiplication by a rounded reciprocal can disagree with division
            // at touching slabs. Widen only broad-phase comparisons. Triangle
            // and OBB hits retain the original division and exclusive limit.
            return near <= far + far.abs() * (4. * f32::EPSILON) + f32::MIN_POSITIVE;
        }
        // Parallel rays can start on a slab face; avoid 0 * infinity. This
        // also handles tiny, nonzero components whose reciprocal overflows.
        let origin = ray.origin;
        let ray = ray.direction;
        let mut near = 0.0_f32;
        let mut far = limit;
        for axis in 0..3 {
            if ray[axis] == 0.0 {
                if origin[axis] < self.lo[axis] || origin[axis] > self.hi[axis] {
                    return false;
                }
            } else {
                let a = (self.lo[axis] - origin[axis]) / ray[axis];
                let b = (self.hi[axis] - origin[axis]) / ray[axis];
                near = near.max(a.min(b));
                far = far.min(a.max(b));
                if near > far {
                    return false;
                }
            }
        }
        far >= 0.0
    }
}

struct SlabRay {
    origin: Vec3A,
    direction: Vec3,
    inverse: Vec3A,
    finite_inverse: bool,
}

impl SlabRay {
    fn new(origin: Vec3, direction: Vec3) -> Self {
        let inverse = Vec3A::from(direction).recip();
        Self {
            origin: origin.into(),
            direction,
            inverse,
            finite_inverse: inverse.is_finite(),
        }
    }
}

struct Node {
    bounds: Bounds,
    start: usize,
    count: usize,
    right: usize,
    axis: usize,
}

pub(super) struct Index {
    objects: Tree,
    ceiling: Tree,
}

impl Index {
    pub(super) fn new(colliders: &[Collider]) -> Self {
        // Detailed ceiling triangles are emitted last. Keep their original
        // indices so ties with earlier fixtures still favor the fixture.
        let split = colliders
            .iter()
            .rposition(|c| c.triangle.is_none() || c.label != "ceiling")
            .map_or(0, |i| i + 1);
        let split = if colliders.len() - split >= 32 {
            split
        } else {
            colliders.len()
        };
        Self {
            objects: Tree::new(&colliders[..split], 0),
            ceiling: Tree::new(&colliders[split..], split),
        }
    }

    pub(super) fn closest(
        &self,
        colliders: &[Collider],
        origin: Vec3,
        direction: Vec3,
        limit: f32,
        skip_people: bool,
    ) -> Option<(usize, f32)> {
        let ray = SlabRay::new(origin, direction);
        let object = self
            .objects
            .hit::<false>(colliders, &ray, limit, skip_people);
        self.ceiling
            .hit::<false>(colliders, &ray, object.map_or(limit, |v| v.1), false)
            .or(object)
    }

    pub(super) fn occluded(
        &self,
        colliders: &[Collider],
        origin: Vec3,
        direction: Vec3,
        limit: f32,
    ) -> bool {
        let ray = SlabRay::new(origin, direction);
        self.objects
            .hit::<true>(colliders, &ray, limit, true)
            .is_some()
            || self
                .ceiling
                .hit::<true>(colliders, &ray, limit, false)
                .is_some()
    }

    /// Content admission reads only the first label. Once the nearest shell
    /// or object is known, any strictly nearer roof triangle has the same
    /// ceiling label. Its distance and its order among other triangles are immaterial.
    pub(super) fn first_label(
        &self,
        colliders: &[Collider],
        origin: Vec3,
        direction: Vec3,
        shell: (f32, &'static str),
    ) -> &'static str {
        let ray = SlabRay::new(origin, direction);
        let (limit, label) = self
            .objects
            .hit::<false>(colliders, &ray, shell.0, false)
            .map_or(shell, |(id, t)| (t, colliders[id].label));
        if label != "ceiling"
            && self
                .ceiling
                .hit::<true>(colliders, &ray, limit, false)
                .is_some()
        {
            "ceiling"
        } else {
            label
        }
    }
}

struct Tree {
    nodes: Vec<Node>,
    order: Vec<usize>,
}
impl Tree {
    fn new(colliders: &[Collider], first: usize) -> Self {
        let bounds: Vec<_> = colliders.iter().map(Bounds::collider).collect();
        let mut order: Vec<_> = (0..colliders.len()).collect();
        let mut nodes = Vec::with_capacity(colliders.len() / 2);
        if !order.is_empty() {
            Self::build(&mut order, 0, &bounds, &mut nodes);
        }
        if first != 0 {
            for id in &mut order {
                *id += first;
            }
        }
        Self { nodes, order }
    }
    fn build(order: &mut [usize], start: usize, bounds: &[Bounds], nodes: &mut Vec<Node>) {
        let mut lo = Vec3A::splat(f32::INFINITY);
        let mut hi = Vec3A::splat(f32::NEG_INFINITY);
        for &i in order.iter() {
            lo = lo.min(bounds[i].lo);
            hi = hi.max(bounds[i].hi);
        }
        let node = nodes.len();
        nodes.push(Node {
            bounds: Bounds { lo, hi },
            start,
            count: order.len(),
            right: 0,
            axis: 0,
        });
        if order.len() <= 4 {
            return;
        }
        let extent = hi - lo;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };
        nodes[node].axis = axis;
        let middle = order.len() / 2;
        order.select_nth_unstable_by(middle, |&a, &b| {
            (bounds[a].lo[axis] + bounds[a].hi[axis])
                .total_cmp(&(bounds[b].lo[axis] + bounds[b].hi[axis]))
                .then(a.cmp(&b))
        });
        let (left, right) = order.split_at_mut(middle);
        Self::build(left, start, bounds, nodes);
        nodes[node].right = nodes.len();
        nodes[node].count = 0;
        Self::build(right, start + middle, bounds, nodes);
    }

    fn hit<const ANY: bool>(
        &self,
        colliders: &[Collider],
        ray: &SlabRay,
        limit: f32,
        skip_people: bool,
    ) -> Option<(usize, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<(usize, f32)> = None;
        // Median splits halve the population at each branch. One pending far
        // child per level fits even the largest addressable slice.
        let mut stack = [0usize; usize::BITS as usize];
        let mut pending = 1;
        while pending > 0 {
            pending -= 1;
            let node = stack[pending];
            let n = &self.nodes[node];
            if !n.bounds.intersects(ray, best.map_or(limit, |v| v.1)) {
                continue;
            }
            if n.count == 0 {
                let (near, far) = if ray.direction[n.axis] >= 0. {
                    (node + 1, n.right)
                } else {
                    (n.right, node + 1)
                };
                stack[pending] = far;
                stack[pending + 1] = near;
                pending += 2;
                continue;
            }
            for &id in &self.order[n.start..n.start + n.count] {
                let c = &colliders[id];
                if skip_people && c.label == "person" {
                    continue;
                }
                if let Some(distance) = c
                    .hit(ray.origin.into(), ray.direction)
                    .filter(|&d| d < limit)
                {
                    if ANY {
                        return Some((id, distance));
                    }
                    if best.is_none_or(|(old, d)| distance.total_cmp(&d).then(id.cmp(&old)).is_lt())
                    {
                        best = Some((id, distance));
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn roof_geometry_keeps_shell_ties_first_labels_and_exact_depths() {
        let mut rng = crate::scene::procedural_indoor::layout::stream(91, 25);
        let mut colliders = vec![
            Collider {
                center: Vec3::new(0., 0., 3.),
                half: Vec3::ONE,
                inverse: Quat::IDENTITY,
                triangle: None,
                label: "person",
            },
            Collider {
                center: Vec3::new(2., 0., 3.),
                half: Vec3::ONE,
                inverse: Quat::IDENTITY,
                triangle: None,
                label: "chair",
            },
        ];
        for i in 0..96 {
            let z = if i % 2 == 0 {
                2.
            } else {
                rng.random_range(1.0..5.0)
            };
            let center = Vec3::new((i % 5) as f32 - 2., (i % 3) as f32 - 1., z);
            let points = [
                center - Vec3::new(1., 1., 0.),
                center + Vec3::new(1., -1., 0.),
                center + Vec3::Y,
            ];
            colliders.push(Collider {
                triangle: Some(points),
                center,
                half: Vec3::new(1., 1., 0.),
                inverse: Quat::IDENTITY,
                label: "ceiling",
            });
        }
        let index = Index::new(&colliders);
        assert!(!index.ceiling.nodes.is_empty());
        for i in 0..512 {
            let origin = Vec3::new(rng.random_range(-3.0..3.0), rng.random_range(-2.0..2.0), 0.);
            let ray = match i % 3 {
                0 => Vec3::Z,
                1 => Vec3::X,
                _ => Vec3::new(rng.random_range(-0.5..0.5), rng.random_range(-0.5..0.5), 1.)
                    .normalize(),
            };
            let hits: Vec<_> = colliders
                .iter()
                .enumerate()
                .filter_map(|(id, c)| c.hit(origin, ray).map(|d| (id, d)))
                .collect();
            let closest = hits
                .iter()
                .copied()
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let t = closest.map_or(2., |(_, d)| d);
            for limit in [t.next_down(), t, t.next_up(), f32::INFINITY] {
                let expected = closest.filter(|&(_, d)| d < limit);
                assert_eq!(
                    index.closest(&colliders, origin, ray, limit, false),
                    expected
                );
                for shell in ["wall", "ceiling", "background"] {
                    assert_eq!(
                        index.first_label(&colliders, origin, ray, (limit, shell)),
                        expected.map_or(shell, |(id, _)| colliders[id].label)
                    );
                }
                assert_eq!(
                    index.occluded(&colliders, origin, ray, limit),
                    hits.iter()
                        .any(|&(id, d)| d < limit && colliders[id].label != "person")
                );
            }
        }
    }

    #[test]
    fn accelerated_hits_retain_scalar_distances_and_ties() {
        let mut rng = crate::scene::procedural_indoor::layout::stream(202, 999);
        let mut colliders: Vec<_> = (0..256)
            .map(|i| Collider {
                triangle: None,
                center: Vec3::new(
                    rng.random_range(-12.0..12.0),
                    rng.random_range(0.0..4.0),
                    rng.random_range(-12.0..12.0),
                ),
                half: Vec3::new(
                    rng.random_range(0.00001..2.0),
                    rng.random_range(0.00001..2.0),
                    rng.random_range(0.00001..2.0),
                ),
                inverse: Quat::from_rotation_y(
                    rng.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                ),
                label: if i % 7 == 0 { "person" } else { "chair" },
            })
            .collect();
        for _ in 0..16 {
            colliders.push(Collider {
                triangle: None,
                center: Vec3::ZERO,
                half: Vec3::ONE,
                inverse: Quat::IDENTITY,
                label: "wall",
            });
        }
        let index = Index::new(&colliders);
        for i in 0..4096 {
            let origin = if i < colliders.len() {
                colliders[i].center
            } else {
                Vec3::new(
                    rng.random_range(-15.0..15.0),
                    rng.random_range(-1.0..5.0),
                    rng.random_range(-15.0..15.0),
                )
            };
            let ray = match i % 9 {
                0 => Vec3::X,
                1 => -Vec3::Y,
                2 => Vec3::Z,
                3 => Vec3::new(1e-8, -0.0, -1.0),
                4 => Vec3::new(f32::from_bits(1), f32::MIN_POSITIVE, 1.0),
                _ => Vec3::new(
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                    rng.random_range(-1.0..1.0),
                )
                .normalize_or(Vec3::X),
            };
            for limit in [0.0, 0.06, 2.0, f32::INFINITY] {
                for skip_people in [false, true] {
                    let expected = colliders
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| !skip_people || c.label != "person")
                        .filter_map(|(i, c)| {
                            c.hit(origin, ray).filter(|&d| d < limit).map(|d| (i, d))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                    assert_eq!(
                        index.closest(&colliders, origin, ray, limit, skip_people),
                        expected,
                        "ray {i}, limit {limit}"
                    );
                }
            }
        }
    }
}
