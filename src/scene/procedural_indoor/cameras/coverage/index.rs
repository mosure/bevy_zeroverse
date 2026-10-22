//! Conservative broad phase only: camera decisions retain the scalar OBB test,
//! original collider order for ties, and the complete candidate/ray population.
use super::Collider;
use bevy::prelude::*;

struct Bounds {
    lo: Vec3,
    hi: Vec3,
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
            lo: c.center - extent - padding,
            hi: c.center + extent + padding,
        }
    }
    fn intersects(&self, origin: Vec3, ray: Vec3, limit: f32) -> bool {
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

struct Node {
    bounds: Bounds,
    start: usize,
    count: usize,
    right: usize,
}

pub(super) struct Index {
    nodes: Vec<Node>,
    order: Vec<usize>,
}
impl Index {
    pub(super) fn new(colliders: &[Collider]) -> Self {
        let bounds: Vec<_> = colliders.iter().map(Bounds::collider).collect();
        let mut order: Vec<_> = (0..colliders.len()).collect();
        let mut nodes = Vec::with_capacity(colliders.len() / 2);
        if !order.is_empty() {
            Self::build(&mut order, 0, &bounds, &mut nodes);
        }
        Self { nodes, order }
    }
    fn build(order: &mut [usize], start: usize, bounds: &[Bounds], nodes: &mut Vec<Node>) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
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
        });
        if order.len() <= 8 {
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

    pub(super) fn closest(
        &self,
        colliders: &[Collider],
        origin: Vec3,
        ray: Vec3,
        limit: f32,
        skip_people: bool,
    ) -> Option<(usize, f32)> {
        struct Query<'a> {
            index: &'a Index,
            colliders: &'a [Collider],
            origin: Vec3,
            ray: Vec3,
            limit: f32,
            skip_people: bool,
            best: Option<(usize, f32)>,
        }
        impl Query<'_> {
            fn visit(&mut self, node: usize) {
                let n = &self.index.nodes[node];
                if !n.bounds.intersects(
                    self.origin,
                    self.ray,
                    self.best.map_or(self.limit, |v| v.1),
                ) {
                    return;
                }
                if n.count == 0 {
                    let right = n.right;
                    self.visit(node + 1);
                    self.visit(right);
                    return;
                }
                for &id in &self.index.order[n.start..n.start + n.count] {
                    let c = &self.colliders[id];
                    if self.skip_people && c.label == "person" {
                        continue;
                    }
                    if let Some(distance) = c.hit(self.origin, self.ray).filter(|&d| d < self.limit)
                    {
                        if self.best.is_none_or(|(old, d)| {
                            distance.total_cmp(&d).then(id.cmp(&old)).is_lt()
                        }) {
                            self.best = Some((id, distance));
                        }
                    }
                }
            }
        }
        let mut query = Query {
            index: self,
            colliders,
            origin,
            ray,
            limit,
            skip_people,
            best: None,
        };
        if !self.nodes.is_empty() {
            query.visit(0);
        }
        query.best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

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
