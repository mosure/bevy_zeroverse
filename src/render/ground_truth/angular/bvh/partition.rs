//! Fixed-bin surface-area partitioning. Only triangle order changes; payloads,
//! intersection arithmetic, culling and the original-ordinal hit tie stay exact.
use super::{Bounds, Primitive};
use bevy::prelude::*;
const BINS: usize = 16;

#[derive(Clone, Copy)]
struct Bin {
    lo: Vec3,
    hi: Vec3,
    count: usize,
}
impl Default for Bin {
    fn default() -> Self {
        Self {
            lo: Vec3::splat(f32::INFINITY),
            hi: Vec3::splat(f32::NEG_INFINITY),
            count: 0,
        }
    }
}
impl Bin {
    fn merge(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
            count: self.count + other.count,
        }
    }
    fn cost(self) -> f64 {
        if self.count == 0 {
            return 0.;
        }
        let d = (self.hi - self.lo).as_dvec3();
        (d.x * d.y + d.y * d.z + d.z * d.x) * self.count as f64
    }
}

pub(super) fn partition(order: &mut [Primitive], bounds: &[Bounds]) -> Option<usize> {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in order.iter() {
        let c = Vec3::from_array(p.center);
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let scales = std::array::from_fn::<_, 3, _>(|axis| BINS as f64 / (hi[axis] - lo[axis]) as f64);
    let index = |p: &Primitive, axis: usize| {
        (((p.center[axis] - lo[axis]) as f64 * scales[axis]) as usize).min(BINS - 1)
    };
    let mut best = None;
    let mut best_cost = f64::INFINITY;
    // Visit the (potentially much larger than cache) bounds array once for all
    // axes. Min/max and integer counts are order-independent, so this preserves
    // the exact split and tie policy of three separate full geometry scans.
    let mut axes = [[Bin::default(); BINS]; 3];
    for p in order.iter() {
        let b = &bounds[p.index as usize];
        for axis in 0..3 {
            if hi[axis] > lo[axis] {
                let bin = &mut axes[axis][index(p, axis)];
                *bin = bin.merge(Bin {
                    lo: b.lo,
                    hi: b.hi,
                    count: 1,
                });
            }
        }
    }
    for axis in 0..3 {
        if hi[axis] <= lo[axis] {
            continue;
        }
        let bins = axes[axis];
        let mut right = [Bin::default(); BINS];
        right[BINS - 1] = bins[BINS - 1];
        for b in (0..BINS - 1).rev() {
            right[b] = bins[b].merge(right[b + 1]);
        }
        let mut left = Bin::default();
        for b in 0..BINS - 1 {
            left = left.merge(bins[b]);
            if left.count == 0 || right[b + 1].count == 0 {
                continue;
            }
            let cost = left.cost() + right[b + 1].cost();
            if cost < best_cost {
                best_cost = cost;
                best = Some((axis, b));
            }
        }
    }
    let (axis, bin) = best?;
    let (mut left, mut right) = (0, order.len());
    while left < right {
        if index(&order[left], axis) <= bin {
            left += 1;
        } else {
            right -= 1;
            order.swap(left, right);
        }
    }
    (left > 0 && left < order.len()).then_some(left)
}
