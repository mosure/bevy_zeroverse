//! Deterministic BVH over authoritative annotation triangles. Surface-area
//! splits near the root isolate large architectural surfaces from small props;
//! bounded-depth median fallback limits both construction and GPU stack use.
use super::super::*;
mod bulk;
mod parallel;
mod partition;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Triangle {
    pub p: [[f32; 4]; 3],
    pub n: [[f32; 4]; 3],
    pub info: [u32; 4], // semantic, cull, batch, original triangle
}
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Node {
    pub lo: [f32; 4],
    pub hi: [f32; 4],
    pub info: [u32; 4],
}

pub(super) fn build(geometry: &ExtractedGeometry) -> (Vec<Triangle>, Vec<Node>) {
    let pool = bulk::pool(geometry.vertices.len().max(geometry.indices.len() / 3));
    let matrices: Vec<_> = geometry
        .instances
        .iter()
        .map(|i| {
            (
                Mat4::from_cols_array_2d(&i.world_from_local),
                Mat4::from_cols_array_2d(&i.normal_from_local),
            )
        })
        .collect();
    // Indexed vertices are shared by many triangles. Transform/normalize each
    // once, preserving exactly the same operations as the triangle expansion.
    let vertices = bulk::collect(geometry.vertices.len(), pool, |i| {
        let vertex = &geometry.vertices[i];
        let (world, normal) = matrices[vertex.instance as usize];
        (
            world
                .transform_point3(Vec3::from_array(vertex.position))
                .extend(0.)
                .to_array(),
            normal
                .transform_vector3(Vec3::from_array(vertex.normal))
                .normalize_or_zero()
                .extend(0.)
                .to_array(),
        )
    });
    let count = geometry.batches.iter().map(|b| b.indices.len() / 3).sum();
    let mut triangles = bulk::Output::new(count);
    for (batch_id, batch) in geometry.batches.iter().enumerate() {
        let indices = geometry.indices[batch.indices.start as usize..batch.indices.end as usize]
            .as_chunks::<3>()
            .0;
        triangles.extend_with(indices.len(), pool, |j| {
            let indices = indices[j];
            let p = indices.map(|i| vertices[i as usize].0);
            let n = indices.map(|i| vertices[i as usize].1);
            Triangle {
                p,
                n,
                info: [
                    geometry.instances[geometry.vertices[indices[0] as usize].instance as usize]
                        .semantic,
                    batch.cull as u32,
                    batch_id as u32,
                    batch.indices.start / 3 + j as u32,
                ],
            }
        });
    }
    finish(triangles.finish())
}
pub(super) fn finish(triangles: Vec<Triangle>) -> (Vec<Triangle>, Vec<Node>) {
    finish_with_strategy(triangles, true)
}
fn finish_with_strategy(
    triangles: Vec<Triangle>,
    surface_area: bool,
) -> (Vec<Triangle>, Vec<Node>) {
    // Partition compact indices rather than repeatedly moving 112-byte triangle
    // records. Cache bounds/centroid sums once; the original ordinal remains the
    // tie-breaker, preserving leaf membership and the hit tie policy.
    let pool = bulk::pool(triangles.len());
    let bounds = bulk::collect(triangles.len(), pool, |i| {
        let t = &triangles[i];
        let p = t.p.map(|p| Vec3::from_array([p[0], p[1], p[2]]));
        Bounds {
            lo: p[0].min(p[1]).min(p[2]),
            hi: p[0].max(p[1]).max(p[2]),
        }
    });
    let mut order = bulk::collect(triangles.len(), pool, |i| Primitive::new(i, &triangles[i]));
    let mut nodes = Vec::with_capacity(triangles.len().div_ceil(4));
    if !triangles.is_empty() {
        if let Some(pool) = pool.filter(|_| surface_area) {
            parallel::build(&mut order, &bounds, &mut nodes, pool);
        } else {
            split(&mut order, &bounds, 0, &mut nodes, 0, surface_area);
        }
    }
    drop(bounds);
    let triangles = bulk::collect(order.len(), pool, |i| triangles[order[i].index as usize]);
    (triangles, nodes)
}
#[derive(Clone, Copy, Default)]
struct Bounds {
    lo: Vec3,
    hi: Vec3,
}
/// Keep median/SAH keys beside the partition index. Comparing two keys no
/// longer performs two random reads into a room-sized bounds allocation. The
/// payload and full bounds stay immutable; only these compact records move.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Primitive {
    index: u32,
    center: [f32; 3],
    ordinal: u32,
}
impl Primitive {
    fn new(index: usize, triangle: &Triangle) -> Self {
        Self {
            index: index as u32,
            center: std::array::from_fn(|axis| triangle.p.iter().map(|p| p[axis]).sum::<f32>()),
            ordinal: triangle.info[3],
        }
    }
}
fn bounds_for(order: &[Primitive], bounds: &[Bounds]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in order {
        let i = p.index as usize;
        lo = lo.min(bounds[i].lo);
        hi = hi.max(bounds[i].hi);
    }
    (lo, hi)
}
fn bounded_node(lo: Vec3, hi: Vec3, base: usize, count: usize) -> Node {
    let eps = Vec3::splat(0.00001) * (hi.abs().max(lo.abs()) + Vec3::ONE);
    Node {
        lo: (lo - eps).extend(0.).to_array(),
        hi: (hi + eps).extend(0.).to_array(),
        info: [0, 0, base as u32, count as u32],
    }
}
fn partition_order(
    order: &mut [Primitive],
    bounds: &[Bounds],
    extent: Vec3,
    depth: u32,
    surface_area: bool,
) -> usize {
    // Spend the surface-area search on broad scene partitions. Deep per-prop
    // searches cost several full scans per level for little traversal benefit;
    // exact median subdivisions then bound construction and GPU stack depth.
    (surface_area && depth < 6 && order.len() >= 64)
        .then(|| partition::partition(order, bounds))
        .flatten()
        .unwrap_or_else(|| {
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            let mid = order.len() / 2;
            order.select_nth_unstable_by(mid, |a, b| {
                a.center[axis]
                    .total_cmp(&b.center[axis])
                    .then(a.ordinal.cmp(&b.ordinal))
            });
            mid
        })
}
fn split(
    order: &mut [Primitive],
    bounds: &[Bounds],
    base: usize,
    nodes: &mut Vec<Node>,
    depth: u32,
    surface_area: bool,
) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(Node::default());
    let (lo, hi) = bounds_for(order, bounds);
    // Conservative box expansion protects triangles on slab boundaries.
    let mut node = bounded_node(lo, hi, base, order.len());
    if order.len() > 8 {
        let mid = partition_order(order, bounds, hi - lo, depth, surface_area);
        let (a, b) = order.split_at_mut(mid);
        node.info = [
            split(a, bounds, base, nodes, depth + 1, surface_area),
            split(b, bounds, base + mid, nodes, depth + 1, surface_area),
            0,
            0,
        ];
    }
    nodes[id as usize] = node;
    id
}

#[cfg(test)]
#[path = "bvh_tests.rs"]
mod tests;
