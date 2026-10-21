//! Deterministic median BVH over authoritative annotation triangles. The empty
//! tree is a valid clear image. Balanced subdivision bounds GPU traversal depth.
use super::super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
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
    let vertices: Vec<_> = geometry
        .vertices
        .iter()
        .map(|vertex| {
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
        })
        .collect();
    let mut triangles = Vec::with_capacity(geometry.indices.len() / 3);
    for (batch_id, batch) in geometry.batches.iter().enumerate() {
        for (j, indices) in geometry.indices
            [batch.indices.start as usize..batch.indices.end as usize]
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            let p = indices.map(|i| vertices[i as usize].0);
            let n = indices.map(|i| vertices[i as usize].1);
            triangles.push(Triangle {
                p,
                n,
                info: [
                    geometry.instances[geometry.vertices[indices[0] as usize].instance as usize]
                        .semantic,
                    batch.cull as u32,
                    batch_id as u32,
                    batch.indices.start / 3 + j as u32,
                ],
            });
        }
    }
    finish(triangles)
}
pub(super) fn finish(triangles: Vec<Triangle>) -> (Vec<Triangle>, Vec<Node>) {
    // Partition compact indices rather than repeatedly moving 112-byte triangle
    // records. Cache bounds/centroid sums once; the original ordinal remains the
    // tie-breaker, preserving leaf membership and the hit tie policy.
    let bounds: Vec<_> = triangles
        .iter()
        .map(|t| {
            let p = t.p.map(|p| Vec3::from_array([p[0], p[1], p[2]]));
            Bounds {
                lo: p[0].min(p[1]).min(p[2]),
                hi: p[0].max(p[1]).max(p[2]),
                center: std::array::from_fn(|axis| t.p.iter().map(|p| p[axis]).sum::<f32>()),
                ordinal: t.info[3],
            }
        })
        .collect();
    let mut order: Vec<_> = (0..triangles.len()).collect();
    let mut nodes = Vec::with_capacity(triangles.len().div_ceil(4));
    if !triangles.is_empty() {
        split(&mut order, &bounds, 0, &mut nodes);
    }
    drop(bounds);
    let triangles = order.into_iter().map(|i| triangles[i]).collect();
    (triangles, nodes)
}
struct Bounds {
    lo: Vec3,
    hi: Vec3,
    center: [f32; 3],
    ordinal: u32,
}
fn split(order: &mut [usize], bounds: &[Bounds], base: usize, nodes: &mut Vec<Node>) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(Node::default());
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &i in order.iter() {
        lo = lo.min(bounds[i].lo);
        hi = hi.max(bounds[i].hi);
    }
    // Conservative box expansion protects triangles on slab boundaries.
    let eps = Vec3::splat(0.00001) * (hi.abs().max(lo.abs()) + Vec3::ONE);
    let mut node = Node {
        lo: (lo - eps).extend(0.).to_array(),
        hi: (hi + eps).extend(0.).to_array(),
        info: [0, 0, base as u32, order.len() as u32],
    };
    if order.len() > 8 {
        let d = hi - lo;
        let axis = if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        };
        let mid = order.len() / 2;
        order.select_nth_unstable_by(mid, |&a, &b| {
            bounds[a].center[axis]
                .total_cmp(&bounds[b].center[axis])
                .then(bounds[a].ordinal.cmp(&bounds[b].ordinal))
        });
        let (a, b) = order.split_at_mut(mid);
        node.info = [
            split(a, bounds, base, nodes),
            split(b, bounds, base + mid, nodes),
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
