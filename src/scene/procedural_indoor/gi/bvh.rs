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
    build_probe_tree_with_parallelism(triangles, true)
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
    static POOL: std::sync::OnceLock<bevy::tasks::TaskPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        bevy::tasks::TaskPoolBuilder::new()
            .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(4)))
            .thread_name("indoor-bvh".into())
            .build()
    })
}

fn build_probe_tree_with_parallelism(triangles: &mut [Triangle], parallel: bool) -> Vec<Node> {
    fn split(
        triangles: &mut [Triangle],
        start: usize,
        index: usize,
        nodes: &mut [Node],
        parallel: bool,
    ) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for t in triangles.iter() {
            let (a, b) = t.bounds();
            lo = lo.min(a);
            hi = hi.max(b);
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
                (a.a + (a.ab + a.ac) / 3.0)[axis].total_cmp(&(b.a + (b.ab + b.ac) / 3.0)[axis])
            });
            let (left, right) = triangles.split_at_mut(middle);
            let right_index = index + 1 + probe_node_count(left.len());
            nodes[0].right = right_index;
            nodes[0].axis = axis;
            nodes[0].count = 0;
            let (left_nodes, right_nodes) = nodes[1..].split_at_mut(right_index - index - 1);
            // Only the root forks: share the existing four-thread transport BVH
            // pool instead of adding a pool per room or an unbounded task tree.
            #[cfg(not(target_arch = "wasm32"))]
            if parallel && left.len() + right.len() >= 65_536 {
                native_pool().scope(|scope| {
                    scope.spawn(async move { split(left, start, index + 1, left_nodes, false) });
                    scope.spawn(async move {
                        split(right, start + middle, right_index, right_nodes, false)
                    });
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
        split(triangles, 0, 0, &mut nodes, parallel);
    }
    nodes
}

#[cfg(not(target_arch = "wasm32"))]
mod sah;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn build_transport(triangles: &[Triangle]) -> (Vec<Triangle>, Vec<Node>) {
    sah::build(triangles)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(super) fn build(triangles: &mut [Triangle]) -> Vec<Node> {
    let (ordered, nodes) = build_transport(triangles);
    triangles.copy_from_slice(&ordered);
    nodes
}

#[cfg(test)]
mod probe_tests;
