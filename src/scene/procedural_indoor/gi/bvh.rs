//! Binned surface-area splits for the GPU diffuse-transport BVH.
//! Only the acceleration structure changes: every triangle and its attributes
//! survive verbatim. Compact references avoid repeatedly moving full triangles
//! or recomputing their bounds during construction.
use super::{Node, Triangle, Vec3};

// Probe classification uses the original median tree. At touching/opposed faces,
// equal-distance hits can have opposite normal signs: changing traversal order
// would change solid classification and therefore relocate otherwise identical
// probes. Keep that discrete placement decision and the CPU oracle unchanged.
pub(super) fn build_probe_tree(triangles: &mut [Triangle]) -> Vec<Node> {
    let parallel = triangles.len() >= 65_536 && parallel_builds();
    build_probe_tree_with_parallelism(triangles, parallel)
}

// Materials and mesh/tangent jobs already share this pool. Tiny worker budgets
// cannot profitably split tree branches as well: a nested blocking join may
// steal a long unrelated job and delay lighting readiness. Keep independent
// asset stages parallel, and fork trees only when there are spare workers.
fn parallel_builds() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native_pool().thread_num() >= 4
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

// Median subdivision depends only on the count and leaf limit. Reserve its exact
// final layout, so disjoint subtrees can write directly without node copies.
fn probe_node_count(count: usize) -> usize {
    if count == 0 {
        return 0;
    }
    if count <= 8 {
        return 1;
    }
    let quotient = count / 8;
    let branches = 1usize << (usize::BITS - 1 - quotient.leading_zeros());
    let leaves = if count / branches == 8 {
        branches + count % branches
    } else {
        2 * branches
    };
    2 * leaves - 1
}

#[cfg(not(target_arch = "wasm32"))]
fn native_pool() -> &'static bevy::tasks::TaskPool {
    // Nested scopes cooperatively execute queued work: no blocking permits or
    // separate four-thread BVH pool multiplying each producer's CPU budget.
    super::super::preparation::workers::pool()
}

fn build_probe_tree_with_parallelism(triangles: &mut [Triangle], parallel: bool) -> Vec<Node> {
    struct Primitive {
        lo: Vec3,
        hi: Vec3,
        centroid: Vec3,
        original: usize,
    }
    // Cache exactly the original float operations, including centroid ties.
    // Partition compact records instead of repeatedly moving all UVs/normals.
    let mut primitives: Vec<_> = triangles
        .iter()
        .enumerate()
        .map(|(original, t)| {
            let (lo, hi) = t.bounds();
            Primitive {
                lo,
                hi,
                centroid: t.a + (t.ab + t.ac) / 3.0,
                original,
            }
        })
        .collect();
    fn split(
        triangles: &mut [Primitive],
        start: usize,
        index: usize,
        nodes: &mut [Node],
        parallel: bool,
    ) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for t in triangles.iter() {
            lo = lo.min(t.lo);
            hi = hi.max(t.hi);
        }
        nodes[0] = Node {
            lo: lo - Vec3::splat(1e-5),
            hi: hi + Vec3::splat(1e-5),
            start,
            count: triangles.len(),
            right: 0,
            axis: 0,
        };
        if triangles.len() > 8 {
            let size = hi - lo;
            let axis = if size.x > size.y && size.x > size.z {
                0
            } else if size.y > size.z {
                1
            } else {
                2
            };
            let middle = triangles.len() / 2;
            triangles.select_nth_unstable_by(middle, |a, b| {
                a.centroid[axis].total_cmp(&b.centroid[axis])
            });
            let (left, right) = triangles.split_at_mut(middle);
            let right_index = index + 1 + probe_node_count(left.len());
            nodes[0].right = right_index;
            nodes[0].axis = axis;
            nodes[0].count = 0;
            let (left_nodes, right_nodes) = nodes[1..].split_at_mut(right_index - index - 1);
            // Only the root forks; share the bounded preparation pool rather
            // than adding a pool per room or an unbounded task tree.
            #[cfg(not(target_arch = "wasm32"))]
            if parallel && left.len() + right.len() >= 65_536 {
                native_pool().scope(|scope| {
                    scope.spawn(async move {
                        split(right, start + middle, right_index, right_nodes, false)
                    });
                    // Complete useful local work before the scope starts
                    // stealing global jobs while waiting for its other branch.
                    split(left, start, index + 1, left_nodes, false);
                });
                return;
            }
            split(left, start, index + 1, left_nodes, false);
            split(right, start + middle, right_index, right_nodes, false);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = parallel;
    }
    let count = probe_node_count(triangles.len());
    let mut nodes = Vec::with_capacity(count);
    nodes.resize_with(count, || Node {
        lo: Vec3::ZERO,
        hi: Vec3::ZERO,
        start: 0,
        count: 0,
        right: 0,
        axis: 0,
    });
    if !triangles.is_empty() {
        split(&mut primitives, 0, 0, &mut nodes, parallel);
    }
    // Apply the permutation in place, cycle by cycle. Each complete triangle
    // moves only here; no second full triangle allocation or persistent cache.
    for start in 0..primitives.len() {
        let mut current = start;
        while primitives[current].original != start {
            let next = primitives[current].original;
            triangles.swap(current, next);
            primitives[current].original = current;
            current = next;
        }
        primitives[current].original = current;
    }
    nodes
}

#[cfg(not(target_arch = "wasm32"))]
mod sah;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn build_transport(
    triangles: &[Triangle],
) -> (Vec<super::TransportTriangle>, Vec<Node>) {
    sah::build_shadow(triangles)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn build_transport_scalar_reference(
    triangles: &[Triangle],
) -> (Vec<Triangle>, Vec<Node>) {
    sah::build_scalar_reference(triangles)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn build(triangles: &mut [Triangle]) -> Vec<Node> {
    let (ordered, nodes) = sah::build(triangles);
    triangles.copy_from_slice(&ordered);
    nodes
}

#[cfg(test)]
mod probe_tests;
