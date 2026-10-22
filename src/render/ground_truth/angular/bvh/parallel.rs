//! Bounded subtree jobs on Bevy's existing compute pool. No new thread pool,
//! shared mutable node vector, worker-count-dependent partition or GPU handoff.
use super::*;
use bevy::tasks::TaskPool;

struct Job<'a> {
    root: usize,
    base: usize,
    depth: u32,
    order: &'a mut [Primitive],
}

pub(super) fn build(
    order: &mut [Primitive],
    bounds: &[Bounds],
    nodes: &mut Vec<Node>,
    pool: &TaskPool,
) {
    let levels = pool.thread_num().clamp(2, 8).ilog2();
    let mut jobs = Vec::new();
    frontier(order, bounds, 0, 0, levels, nodes, &mut jobs);
    // Cooperatively execute the pool while waiting: Render preparation itself
    // may occupy its workers. Disabling this can deadlock a small CPU budget.
    let mut results = pool.scope(|scope| {
        for job in jobs {
            scope.spawn(async move {
                let mut local = Vec::with_capacity(job.order.len().div_ceil(4));
                split(job.order, bounds, job.base, &mut local, job.depth, true);
                (job.root, local)
            });
        }
    });
    // Completion order cannot affect node numbering or deterministic replay.
    results.sort_unstable_by_key(|r| r.0);
    for (root, local) in results {
        let offset = nodes.len() as u32 - 1;
        for (i, mut node) in local.into_iter().enumerate() {
            if node.info[3] == 0 {
                node.info[0] += offset;
                node.info[1] += offset;
            }
            if i == 0 {
                nodes[root] = node;
            } else {
                nodes.push(node);
            }
        }
    }
}

fn frontier<'a>(
    order: &'a mut [Primitive],
    bounds: &[Bounds],
    base: usize,
    depth: u32,
    levels: u32,
    nodes: &mut Vec<Node>,
    jobs: &mut Vec<Job<'a>>,
) -> u32 {
    let id = nodes.len();
    nodes.push(Node::default());
    if depth >= levels || order.len() < 1024 {
        jobs.push(Job {
            root: id,
            base,
            depth,
            order,
        });
    } else {
        let (lo, hi) = bounds_for(order, bounds);
        let mut node = bounded_node(lo, hi, base, order.len());
        let mid = partition_order(order, bounds, hi - lo, depth, true);
        let (a, b) = order.split_at_mut(mid);
        node.info = [
            frontier(a, bounds, base, depth + 1, levels, nodes, jobs),
            frontier(b, bounds, base + mid, depth + 1, levels, nodes, jobs),
            0,
            0,
        ];
        nodes[id] = node;
    }
    id as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn subtree_parallelism_preserves_serial_partitions_and_bounds() {
        let input = super::super::tests::triangles(70_000, false);
        let bounds: Vec<_> = input
            .iter()
            .map(|t| {
                let p = t.p.map(|p| Vec3::from_array(p[..3].try_into().unwrap()));
                Bounds {
                    lo: p[0].min(p[1]).min(p[2]),
                    hi: p[0].max(p[1]).max(p[2]),
                }
            })
            .collect();
        let mut serial: Vec<_> = input
            .iter()
            .enumerate()
            .map(|(i, b)| Primitive::new(i, b))
            .collect();
        let mut expected = Vec::new();
        split(&mut serial, &bounds, 0, &mut expected, 0, true);
        for threads in [2, 4] {
            let pool = bevy::tasks::TaskPoolBuilder::new()
                .num_threads(threads)
                .build();
            let results = pool.scope(|scope| {
                for _ in 0..threads {
                    let (pool, bounds, input) = (&pool, &bounds, &input);
                    scope.spawn(async move {
                        let mut order: Vec<_> = input
                            .iter()
                            .enumerate()
                            .map(|(i, b)| Primitive::new(i, b))
                            .collect();
                        let mut nodes = Vec::new();
                        build(&mut order, bounds, &mut nodes, pool);
                        (order, nodes)
                    });
                }
            });
            for (order, nodes) in results {
                assert_eq!(order, serial);
                assert_eq!(nodes.len(), expected.len());
                let mut stack = vec![(0, 0)];
                while let Some((a, b)) = stack.pop() {
                    let (a, b) = (nodes[a], expected[b]);
                    assert_eq!(a.lo, b.lo);
                    assert_eq!(a.hi, b.hi);
                    assert_eq!(a.info[2..], b.info[2..]);
                    if a.info[3] == 0 {
                        stack.extend([
                            (a.info[0] as usize, b.info[0] as usize),
                            (a.info[1] as usize, b.info[1] as usize),
                        ]);
                    }
                }
            }
        }
    }
}
