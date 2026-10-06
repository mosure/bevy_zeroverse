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
        for axis in 0..3 {
            if extent[axis] <= 0.0 {
                continue;
            }
            let scale = BINS as f32 / extent[axis];
            let mut bins = [Bin::EMPTY; BINS];
            for p in primitives.iter() {
                let b = &mut bins[bin(p.centroid[axis], centroids.lo[axis], scale)];
                b.bounds = b.bounds.include(p.bounds);
                b.count += 1;
            }
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
    let pool = super::native_pool();
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Vec2;

    // Pre-optimization ownership path, retained independently as an exact
    // oracle. Tree splitting is unchanged; this models its old clone, ordered
    // allocation and copy-back rather than using the new ownership helper.
    fn legacy_build(original: &[Triangle], parallel: bool) -> (Vec<Triangle>, Vec<Node>) {
        let mut triangles = original.to_vec();
        if triangles.is_empty() {
            return (triangles, Vec::new());
        }
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
        let ordered: Vec<_> = primitives.iter().map(|p| triangles[p.original]).collect();
        triangles.copy_from_slice(&ordered);
        (triangles, nodes)
    }

    fn fixture(count: usize, coincident: bool) -> Vec<Triangle> {
        (0..count)
            .map(|i| Triangle {
                a: if coincident {
                    Vec3::ZERO
                } else {
                    Vec3::new(
                        (i % 97) as f32 * 0.13,
                        (i / 97 % 23) as f32 * 0.4,
                        (i / (97 * 23)) as f32 * 0.21,
                    )
                },
                ab: Vec3::X * if !coincident && i % 5 == 0 { 1e-7 } else { 0.1 },
                ac: Vec3::Y * 0.1,
                uv: [Vec2::ZERO, Vec2::X, Vec2::Y],
                normal: if i % 2 == 0 { Vec3::Z } else { -Vec3::Z },
                material: i,
            })
            .collect()
    }

    fn assert_exact_triangles(a: &[Triangle], b: &[Triangle]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(
                a.a.to_array().map(f32::to_bits),
                b.a.to_array().map(f32::to_bits)
            );
            assert_eq!(
                a.ab.to_array().map(f32::to_bits),
                b.ab.to_array().map(f32::to_bits)
            );
            assert_eq!(
                a.ac.to_array().map(f32::to_bits),
                b.ac.to_array().map(f32::to_bits)
            );
            assert_eq!(
                a.normal.to_array().map(f32::to_bits),
                b.normal.to_array().map(f32::to_bits)
            );
            for (a, b) in a.uv.iter().zip(b.uv.iter()) {
                assert_eq!(
                    a.to_array().map(f32::to_bits),
                    b.to_array().map(f32::to_bits)
                );
            }
            assert_eq!(a.material, b.material);
        }
    }

    fn assert_exact_nodes(a: &[Node], b: &[Node]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(
                a.lo.to_array().map(f32::to_bits),
                b.lo.to_array().map(f32::to_bits)
            );
            assert_eq!(
                a.hi.to_array().map(f32::to_bits),
                b.hi.to_array().map(f32::to_bits)
            );
            assert_eq!(
                (a.start, a.count, a.right, a.axis),
                (b.start, b.count, b.right, b.axis)
            );
        }
    }

    #[test]
    fn owned_transport_matches_legacy_clone_copy_builder() {
        for count in [0, 1, 257, 40_000] {
            for coincident in [false, true] {
                let original = fixture(count, coincident);
                let unchanged = original.clone();
                let expected = legacy_build(&original, true);
                let actual = build(&original);
                assert_exact_triangles(&expected.0, &actual.0);
                assert_exact_nodes(&expected.1, &actual.1);
                // The original closest-hit/probe tree remains immutable.
                assert_exact_triangles(&original, &unchanged);
            }
        }
    }

    #[test]
    #[ignore = "bounded allocation/BVH microbenchmark; run separately from GPU measurements"]
    fn benchmark_owned_transport() {
        use std::{hint::black_box, time::Instant};
        for count in [100_000, 300_000, 700_000] {
            let original = fixture(count, false);
            let expected = legacy_build(&original, true);
            let actual = build(&original);
            assert_exact_triangles(&expected.0, &actual.0);
            assert_exact_nodes(&expected.1, &actual.1);
            drop((expected, actual));
            let mut elapsed = [0.0; 2];
            for repetition in 0..4 {
                for candidate in if repetition % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let started = Instant::now();
                    let result = if candidate {
                        build(black_box(&original))
                    } else {
                        legacy_build(black_box(&original), true)
                    };
                    black_box(&result);
                    elapsed[usize::from(candidate)] += started.elapsed().as_secs_f64();
                    drop(result);
                }
            }
            eprintln!(
                "{{\"kernel\":\"owned_transport\",\"triangles\":{count},\"repetitions\":4,\"legacy_seconds\":{},\"candidate_seconds\":{},\"speedup\":{},\"peak_bytes_removed\":{}}}",
                elapsed[0] / 4.0,
                elapsed[1] / 4.0,
                elapsed[0] / elapsed[1],
                count * std::mem::size_of::<Triangle>()
            );
        }
    }

    fn check_tree(nodes: &[Node], index: usize, triangles: &[Triangle], depth: usize) -> usize {
        assert!(depth < 64, "CPU and GPU traversal stack budget");
        let node = &nodes[index];
        if node.count != 0 {
            assert!(node.count <= LEAF_SIZE);
            for t in &triangles[node.start..node.start + node.count] {
                let (lo, hi) = t.bounds();
                assert!(node.lo.cmple(lo).all() && node.hi.cmpge(hi).all());
            }
            node.count
        } else {
            assert!(node.right > index + 1);
            for child in [index + 1, node.right] {
                assert!(node.lo.cmple(nodes[child].lo).all());
                assert!(node.hi.cmpge(nodes[child].hi).all());
            }
            check_tree(nodes, index + 1, triangles, depth + 1)
                + check_tree(nodes, node.right, triangles, depth + 1)
        }
    }

    #[test]
    fn spatial_splits_preserve_thin_clustered_and_coincident_triangles() {
        for coincident in [false, true] {
            let mut original: Vec<_> = (0..2048)
                .map(|i| Triangle {
                    a: if coincident {
                        Vec3::ZERO
                    } else {
                        Vec3::new((i % 23) as f32 * 0.13, (i / 23) as f32 * 0.4, 0.0)
                    },
                    ab: Vec3::X * if i % 5 == 0 { 1e-7 } else { 0.1 },
                    ac: Vec3::Y * 0.1,
                    uv: [Vec2::ZERO, Vec2::X, Vec2::Y],
                    normal: Vec3::Z,
                    material: i,
                })
                .collect();
            // Truly identical centroids exercise median fallback, including ties.
            if coincident {
                for t in &mut original {
                    t.ab = Vec3::X * 0.1;
                }
            }
            let (triangles, nodes) = build(&original);
            assert_eq!(check_tree(&nodes, 0, &triangles, 0), original.len());
            let mut ids: Vec<_> = triangles.iter().map(|t| t.material).collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..original.len()).collect::<Vec<_>>());
            for t in &triangles {
                let o = &original[t.material];
                assert_eq!(
                    (t.a, t.ab, t.ac, t.uv, t.normal),
                    (o.a, o.ab, o.ac, o.uv, o.normal)
                );
            }
            let (repeated, repeated_nodes) = build(&original);
            assert_eq!(nodes.len(), repeated_nodes.len());
            assert_eq!(
                triangles.iter().map(|t| t.material).collect::<Vec<_>>(),
                repeated.iter().map(|t| t.material).collect::<Vec<_>>()
            );
        }
        let (triangles, nodes) = build(&[]);
        assert!(triangles.is_empty() && nodes.is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn parallel_subtrees_have_identical_bounds_indices_and_triangle_order() {
        let original: Vec<_> = (0..70_000)
            .map(|i| Triangle {
                a: Vec3::new((i % 97) as f32 * 0.13, (i / 97) as f32 * 0.4, 0.0),
                ab: Vec3::X * 0.1,
                ac: Vec3::Y * 0.1,
                uv: [Vec2::ZERO, Vec2::X, Vec2::Y],
                normal: Vec3::Z,
                material: i,
            })
            .collect();
        let (serial, a) = build_with_parallelism(&original, false);
        let (parallel, b) = build_with_parallelism(&original, true);
        assert_eq!(check_tree(&b, 0, &parallel, 0), parallel.len());
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b.iter()) {
            assert_eq!(
                (a.lo, a.hi, a.start, a.count, a.right, a.axis),
                (b.lo, b.hi, b.start, b.count, b.right, b.axis)
            );
        }
        assert!(serial
            .iter()
            .zip(parallel.iter())
            .all(|(a, b)| a.material == b.material));
    }
}
