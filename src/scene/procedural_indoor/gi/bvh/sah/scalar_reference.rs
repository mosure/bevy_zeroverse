//! Frozen scalar SAH builder for exact SIMD/order qualification.
//! Deterministic native BVH construction; independent branches use a bounded pool.
use super::{Node, Triangle, Vec3};

const BINS: usize = 16;
const LEAF_SIZE: usize = 8;
// Both traversal implementations have 64 stack entries. At most 24 unbalanced
// splits followed by median splits of a u32 triangle count need < 56 entries.
const MAX_SAH_DEPTH: usize = 24;

#[derive(Clone, Copy)]
struct Bounds {
    lo: Vec3,
    hi: Vec3,
}
impl Bounds {
    const EMPTY: Self = Self {
        lo: Vec3::splat(f32::INFINITY),
        hi: Vec3::splat(f32::NEG_INFINITY),
    };
    fn include(self, other: Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
    fn area(self) -> f32 {
        let d = (self.hi - self.lo).max(Vec3::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }
}

struct Primitive {
    bounds: Bounds,
    centroid: Vec3,
    original: usize,
}

#[derive(Clone, Copy)]
struct Bin {
    bounds: Bounds,
    count: usize,
}
impl Bin {
    const EMPTY: Self = Self {
        bounds: Bounds::EMPTY,
        count: 0,
    };
    fn include(self, other: Self) -> Self {
        Self {
            bounds: self.bounds.include(other.bounds),
            count: self.count + other.count,
        }
    }
    fn cost(self) -> f32 {
        self.bounds.area() * self.count as f32
    }
}

pub(super) fn build(triangles: &[Triangle]) -> (Vec<Triangle>, Vec<Node>) {
    build_with_parallelism(triangles, true)
}

fn build_with_parallelism(triangles: &[Triangle], parallel: bool) -> (Vec<Triangle>, Vec<Node>) {
    if triangles.is_empty() {
        return (Vec::new(), Vec::new());
    }
    assert!(
        u32::try_from(triangles.len()).is_ok(),
        "GI triangle index overflow"
    );
    let mut primitives: Vec<_> = triangles
        .iter()
        .enumerate()
        .map(|(original, triangle)| {
            let (lo, hi) = triangle.bounds();
            Primitive {
                bounds: Bounds { lo, hi },
                centroid: triangle.a + (triangle.ab + triangle.ac) / 3.0,
                original,
            }
        })
        .collect();
    let mut nodes = Vec::with_capacity(triangles.len() / 2);
    subdivide(&mut primitives, 0, 0, parallel, &mut nodes);
    // The caller already retains the original median/probe tree. Its immutable
    // triangles are read directly into the final SAH-owned order: no preliminary
    // full clone or copy-back allocation is required.
    let ordered = primitives.iter().map(|p| triangles[p.original]).collect();
    (ordered, nodes)
}

fn bin(centroid: f32, lower: f32, scale: f32) -> usize {
    (((centroid - lower) * scale) as usize).min(BINS - 1)
}

/// Accumulate all active axes while each compact primitive is in cache. Every
/// individual bin receives the same primitives in the same order as the three
/// separate scans, so bounds, split ties and the final traversal remain exact.
fn spatial_bins(primitives: &[Primitive], centroids: Bounds) -> ([[Bin; BINS]; 3], [f32; 3]) {
    let extent = (centroids.hi - centroids.lo).to_array();
    let scales = extent.map(|e| if e <= 0.0 { 0.0 } else { BINS as f32 / e });
    let mut bins = [[Bin::EMPTY; BINS]; 3];
    for primitive in primitives {
        for axis in 0..3 {
            if extent[axis] <= 0.0 {
                continue;
            }
            let b =
                &mut bins[axis][bin(primitive.centroid[axis], centroids.lo[axis], scales[axis])];
            b.bounds = b.bounds.include(primitive.bounds);
            b.count += 1;
        }
    }
    (bins, scales)
}

fn subdivide(
    primitives: &mut [Primitive],
    start: usize,
    depth: usize,
    parallel: bool,
    nodes: &mut Vec<Node>,
) {
    let mut bounds = Bounds::EMPTY;
    let mut centroids = Bounds::EMPTY;
    for p in primitives.iter() {
        bounds = bounds.include(p.bounds);
        centroids = centroids.include(Bounds {
            lo: p.centroid,
            hi: p.centroid,
        });
    }
    let index = nodes.len();
    nodes.push(Node {
        lo: bounds.lo - Vec3::splat(1e-5),
        hi: bounds.hi + Vec3::splat(1e-5),
        start,
        count: primitives.len(),
        right: 0,
        axis: 0,
    });
    if primitives.len() <= LEAF_SIZE {
        return;
    }
    let extent = centroids.hi - centroids.lo;
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    if depth < MAX_SAH_DEPTH {
        let (axis_bins, scales) = spatial_bins(primitives, centroids);
        for axis in 0..3 {
            if extent[axis] <= 0.0 {
                continue;
            }
            let scale = scales[axis];
            let bins = &axis_bins[axis];
            let mut suffix = [Bin::EMPTY; BINS];
            suffix[BINS - 1] = bins[BINS - 1];
            for i in (0..BINS - 1).rev() {
                suffix[i] = bins[i].include(suffix[i + 1]);
            }
            let mut left = Bin::EMPTY;
            for i in 0..BINS - 1 {
                left = left.include(bins[i]);
                let right = suffix[i + 1];
                if left.count == 0 || right.count == 0 {
                    continue;
                }
                let cost = left.cost() + right.cost();
                if cost < best_cost {
                    best_cost = cost;
                    best = Some((axis, i, scale));
                }
            }
        }
    }
    let (axis, middle) = if let Some((axis, split, scale)) = best {
        let mut middle = 0;
        for i in 0..primitives.len() {
            if bin(primitives[i].centroid[axis], centroids.lo[axis], scale) <= split {
                primitives.swap(i, middle);
                middle += 1;
            }
        }
        (axis, middle)
    } else {
        // Coincident centroids and the depth limit still get bounded leaves.
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };
        let middle = primitives.len() / 2;
        primitives.select_nth_unstable_by(middle, |a, b| {
            a.centroid[axis]
                .total_cmp(&b.centroid[axis])
                .then(a.original.cmp(&b.original))
        });
        (axis, middle)
    };
    debug_assert!(middle > 0 && middle < primitives.len());
    nodes[index].count = 0;
    nodes[index].axis = axis;
    let count = primitives.len();
    let (left, right) = primitives.split_at_mut(middle);
    #[cfg(not(target_arch = "wasm32"))]
    if parallel && count >= 32_768 && depth < 2 {
        let [left_nodes, right_nodes] = parallel_branches(left, right, start, depth);
        append_branch(nodes, left_nodes);
        nodes[index].right = nodes.len();
        append_branch(nodes, right_nodes);
        return;
    }
    #[cfg(target_arch = "wasm32")]
    let _ = count;
    subdivide(left, start, depth + 1, parallel, nodes);
    nodes[index].right = nodes.len();
    subdivide(right, start + middle, depth + 1, parallel, nodes);
}

#[cfg(not(target_arch = "wasm32"))]
fn parallel_branches(
    left: &mut [Primitive],
    right: &mut [Primitive],
    start: usize,
    depth: usize,
) -> [Vec<Node>; 2] {
    let pool = super::super::native_pool();
    let right_start = start + left.len();
    let mut branches = pool.scope(|scope| {
        for (order, primitives, offset) in [(0, left, start), (1, right, right_start)] {
            scope.spawn(async move {
                let mut nodes = Vec::with_capacity(primitives.len() / 2);
                subdivide(primitives, offset, depth + 1, true, &mut nodes);
                (order, nodes)
            });
        }
    });
    // Completion order must not change flattened indices or traversal order.
    branches.sort_unstable_by_key(|(order, _)| *order);
    let mut branches = branches.into_iter().map(|(_, nodes)| nodes);
    [branches.next().unwrap(), branches.next().unwrap()]
}

#[cfg(not(target_arch = "wasm32"))]
fn append_branch(nodes: &mut Vec<Node>, mut branch: Vec<Node>) {
    let offset = nodes.len();
    for node in &mut branch {
        if node.count == 0 {
            node.right += offset;
        }
    }
    nodes.extend(branch);
}
