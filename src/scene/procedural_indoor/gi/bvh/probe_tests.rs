//! Original median builder is a full triangle/node-order oracle.
use super::*;

fn legacy(triangles: &mut [Triangle]) -> Vec<Node> {
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

fn fixture(count: usize, pattern: usize) -> Vec<Triangle> {
    use bevy::prelude::Vec2;
    (0..count)
        .map(|i| {
            let value = |salt: usize| {
                let mut bits = (i as u32)
                    .wrapping_mul(0x9e37_79b9)
                    .wrapping_add(salt as u32);
                bits ^= bits >> 16;
                bits = bits.wrapping_mul(0x7feb_352d);
                bits ^= bits >> 15;
                (bits >> 8) as f32 / 16_777_216.0 * 20.0 - 10.0
            };
            let a = match pattern {
                1 => Vec3::new(value(7), value(853), value(101) * 1e-6),
                2 => Vec3::new((i % 3) as f32, 0., 0.),
                _ => Vec3::new(value(7), value(853), value(101)),
            };
            Triangle {
                a,
                ab: Vec3::new(0.02 + (i % 13) as f32 * 0.001, 0., 0.),
                ac: Vec3::new(0., 0.04, 0.0001),
                uv: [Vec2::ZERO, Vec2::X, Vec2::new(value(3000), value(7000))],
                normal: Vec3::new(-0., 0., if i % 2 == 0 { 1. } else { -1. }),
                material: i,
            }
        })
        .collect()
}

fn same_triangles(a: &[Triangle], b: &[Triangle]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        for (a, b) in [(a.a, b.a), (a.ab, b.ab), (a.ac, b.ac), (a.normal, b.normal)] {
            assert_eq!(
                a.to_array().map(f32::to_bits),
                b.to_array().map(f32::to_bits)
            );
        }
        for (a, b) in a.uv.iter().zip(b.uv) {
            assert_eq!(
                a.to_array().map(f32::to_bits),
                b.to_array().map(f32::to_bits)
            );
        }
        assert_eq!(a.material, b.material);
    }
}

fn same_nodes(a: &[Node], b: &[Node]) {
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
fn exact_node_capacity_matches_independent_recursive_leaf_counts() {
    fn recursive(n: usize) -> usize {
        match n {
            0 => 0,
            1..=8 => 1,
            _ => 1 + recursive(n / 2) + recursive(n - n / 2),
        }
    }
    for n in (0..8192).chain([32_768, 65_536, 131_079, 300_000, 700_000]) {
        assert_eq!(probe_node_count(n), recursive(n), "triangle count {n}");
    }
}

#[test]
fn preallocated_parallel_probe_tree_matches_original_order_and_float_bits() {
    for pattern in 0..3 {
        for count in [0, 1, 8, 9, 15, 16, 17, 31, 32, 33, 4097, 65_536, 131_079] {
            let original = fixture(count, pattern);
            let mut reference = original.clone();
            let reference_nodes = legacy(&mut reference);
            for parallel in [false, true] {
                let mut actual = original.clone();
                let nodes = build_probe_tree_with_parallelism(&mut actual, parallel);
                same_triangles(&reference, &actual);
                same_nodes(&reference_nodes, &nodes);
            }
        }
    }
}

#[test]
#[ignore = "CPU diagnostic; not scene or GPU throughput"]
fn probe_tree_cpu_benchmark() {
    use std::time::Instant;
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(f64::total_cmp);
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    }
    for count in [100_000, 300_000, 700_000] {
        let input = fixture(count, 0);
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        for repeat in 0..6 {
            let order = if repeat % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for mode in order {
                let begin = Instant::now();
                let mut triangles = input.clone();
                let nodes = match mode {
                    0 => legacy(&mut triangles),
                    1 => build_probe_tree_with_parallelism(&mut triangles, false),
                    _ => build_probe_tree_with_parallelism(&mut triangles, true),
                };
                std::hint::black_box((&triangles, &nodes));
                drop(nodes);
                drop(triangles);
                times[mode].push(begin.elapsed().as_secs_f64());
            }
        }
        let legacy = median(&mut times[0]);
        let serial = median(&mut times[1]);
        let parallel = median(&mut times[2]);
        println!(
            "{}",
            serde_json::json!({"triangles":count,"repetitions":6,"legacy_seconds":legacy,"preallocated_serial_seconds":serial,"preallocated_parallel_seconds":parallel,"parallel_speedup":legacy/parallel,"scope":"CPU tree diagnostic including fresh input clone, node initialization/allocation and free; geometry synthesis excluded"}),
        );
    }
}
