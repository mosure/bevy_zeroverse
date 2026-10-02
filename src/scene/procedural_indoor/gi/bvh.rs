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
    fn split(triangles: &mut [Triangle], start: usize, nodes: &mut Vec<Node>) {
        let index = nodes.len();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for t in triangles.iter() {
            let (a, b) = t.bounds();
            lo = lo.min(a);
            hi = hi.max(b);
        }
        nodes.push(Node {
            lo: lo - Vec3::splat(1e-5),
            hi: hi + Vec3::splat(1e-5),
            start,
            count: triangles.len(),
            right: 0,
            axis: 0,
        });
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
            split(left, start, nodes);
            nodes[index].right = nodes.len();
            split(right, start + middle, nodes);
            nodes[index].axis = axis;
            nodes[index].count = 0;
        }
    }
    let mut nodes = Vec::new();
    if !triangles.is_empty() {
        split(triangles, 0, &mut nodes);
    }
    nodes
}

#[cfg(not(target_arch = "wasm32"))]
mod sah;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn build(triangles: &mut [Triangle]) -> Vec<Node> {
    sah::build(triangles)
}
