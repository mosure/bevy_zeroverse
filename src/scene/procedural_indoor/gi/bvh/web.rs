//! Transfer only float bounds/centroids to the worker. Full triangles and their
//! shading attributes stay in the renderer. Preserve native median ordering and ties.
use super::*;
#[cfg(target_arch = "wasm32")]
use crate::scene::procedural_indoor::preparation::{cooperate, pipeline::browser};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::js_sys::{Float32Array, Uint32Array};

#[cfg(target_arch = "wasm32")]
const CHUNK: usize = 4096;
const NODE_WORDS: usize = 10;

#[cfg(target_arch = "wasm32")]
pub(in crate::scene::procedural_indoor::gi) async fn build(
    triangles: &mut [Triangle],
) -> Option<Vec<Node>> {
    if triangles.len() < CHUNK || !browser::worker_enabled() {
        return None;
    }
    let input = Float32Array::new_with_length((triangles.len() * 9) as u32);
    let mut scratch = Vec::with_capacity(CHUNK * 9);
    for (i, chunk) in triangles.chunks(CHUNK).enumerate() {
        scratch.clear();
        for t in chunk {
            let (lo, hi) = t.bounds();
            scratch.extend(lo.to_array());
            scratch.extend(hi.to_array());
            scratch.extend((t.a + (t.ab + t.ac) / 3.).to_array());
        }
        let start = (i * CHUNK * 9) as u32;
        input
            .subarray(start, start + scratch.len() as u32)
            .copy_from(&scratch);
        cooperate().await;
    }
    let mut buffer = browser::run("probeTree", &input)
        .await?
        .unchecked_into::<Uint32Array>()
        .to_vec();
    let count = probe_node_count(triangles.len());
    if buffer.len() != count * NODE_WORDS + triangles.len() {
        return None;
    }
    let (tree, order) = buffer.split_at_mut(count * NODE_WORDS);
    let mut nodes = Vec::with_capacity(count);
    for chunk in tree.chunks(CHUNK * NODE_WORDS) {
        for n in chunk.as_chunks::<NODE_WORDS>().0 {
            nodes.push(Node {
                lo: Vec3::new(
                    f32::from_bits(n[0]),
                    f32::from_bits(n[1]),
                    f32::from_bits(n[2]),
                ),
                hi: Vec3::new(
                    f32::from_bits(n[3]),
                    f32::from_bits(n[4]),
                    f32::from_bits(n[5]),
                ),
                start: n[6] as usize,
                count: n[7] as usize,
                right: n[8] as usize,
                axis: n[9] as usize,
            });
        }
        cooperate().await;
    }
    // Use the same in-place cycles as the scalar path. Keep only indices in
    // transit and yield even inside a single long permutation cycle.
    let mut swaps = 0;
    for start in 0..order.len() {
        let mut i = start;
        while order[i] as usize != start {
            let next = order[i] as usize;
            triangles.swap(i, next);
            order[i] = i as u32;
            i = next;
            swaps += 1;
            if swaps == CHUNK {
                cooperate().await;
                swaps = 0;
            }
        }
        order[i] = i as u32;
    }
    Some(nodes)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(js_name = zeroverseProbeTree))]
pub fn probe_tree(bounds: &[f32]) -> Vec<u32> {
    let mut primitives: Vec<_> = bounds
        .as_chunks::<9>()
        .0
        .iter()
        .enumerate()
        .map(|(original, p)| Primitive {
            bounds: Bounds {
                lo: Vec3::from_slice(p),
                hi: Vec3::from_slice(&p[3..]),
            },
            centroid: Vec3::from_slice(&p[6..]),
            original,
        })
        .collect();
    let nodes = probe_nodes(&mut primitives, false);
    let mut words = Vec::with_capacity(nodes.len() * NODE_WORDS + primitives.len());
    for n in nodes {
        words.extend(n.lo.to_array().map(f32::to_bits));
        words.extend(n.hi.to_array().map(f32::to_bits));
        words.extend([
            n.start as u32,
            n.count as u32,
            n.right as u32,
            n.axis as u32,
        ]);
    }
    words.extend(primitives.into_iter().map(|p| p.original as u32));
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn studio_probe_worker_preserves_scalar_nodes_and_triangle_tie_order() {
        for n in [0, 1, 8, 9, 4097] {
            let mut triangles: Vec<Triangle> = (0..n)
                .map(|i| Triangle {
                    a: Vec3::new((i % 31) as f32, (i % 3) as f32, (i % 17) as f32),
                    ab: Vec3::X,
                    ac: Vec3::Z,
                    uv: Default::default(),
                    normal: Vec3::Y,
                    material: i,
                })
                .collect();
            let mut raw = Vec::new();
            for t in &triangles {
                let (lo, hi) = t.bounds();
                raw.extend(lo.to_array());
                raw.extend(hi.to_array());
                raw.extend((t.a + (t.ab + t.ac) / 3.).to_array());
            }
            let words = probe_tree(&raw);
            let (nodes, _) = build_probe_tree_with_parallelism(&mut triangles, false);
            let (packed, order) = words.split_at(nodes.len() * NODE_WORDS);
            for (r, n) in packed.as_chunks::<NODE_WORDS>().0.iter().zip(nodes) {
                assert_eq!(r[..3], n.lo.to_array().map(f32::to_bits));
                assert_eq!(r[3..6], n.hi.to_array().map(f32::to_bits));
                assert_eq!(
                    r[6..],
                    [n.start, n.count, n.right, n.axis].map(|v| v as u32)
                );
            }
            assert_eq!(
                order,
                triangles
                    .iter()
                    .map(|t| t.material as u32)
                    .collect::<Vec<_>>()
            );
        }
    }
}
