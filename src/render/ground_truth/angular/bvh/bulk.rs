//! Ordered, bounded linear passes on the existing render compute pool. Workers
//! own disjoint output slices; no channel, second result copy or new pool.
use bevy::tasks::{ComputeTaskPool, TaskPool};
use std::mem::MaybeUninit;

pub(super) fn pool(len: usize) -> Option<&'static TaskPool> {
    ComputeTaskPool::try_get()
        .filter(|pool| !cfg!(target_arch = "wasm32") && pool.thread_num() > 1 && len >= 65_536)
        .map(|pool| &**pool)
}

pub(super) fn collect<T: Copy + Send>(
    len: usize,
    pool: Option<&TaskPool>,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    if pool.is_none() {
        return (0..len).map(f).collect();
    }
    let mut output = Output::new(len);
    output.extend_with(len, pool, f);
    output.finish()
}

/// An initialized prefix of an exact-size allocation. Only `extend_with` can
/// advance that prefix, after every scoped worker has finished successfully.
/// Copy values need no destructor if a mapper panics before completing a chunk.
pub(super) struct Output<T: Copy> {
    values: Box<[MaybeUninit<T>]>,
    initialized: usize,
}
impl<T: Copy + Send> Output<T> {
    pub(super) fn new(len: usize) -> Self {
        Self {
            values: Box::new_uninit_slice(len),
            initialized: 0,
        }
    }
    pub(super) fn extend_with(
        &mut self,
        len: usize,
        pool: Option<&TaskPool>,
        f: impl Fn(usize) -> T + Sync,
    ) {
        let end = self
            .initialized
            .checked_add(len)
            .expect("BVH output length overflow");
        fill(&mut self.values[self.initialized..end], pool, f);
        self.initialized = end;
    }
    pub(super) fn finish(self) -> Vec<T> {
        assert_eq!(self.initialized, self.values.len(), "incomplete BVH output");
        // SAFETY: extend_with writes every element of a disjoint prefix before
        // advancing initialized. Scope joins every task and propagates panics;
        // the equality above proves the entire allocation contains valid T.
        unsafe { self.values.assume_init() }.into_vec()
    }
}

fn fill<T: Send>(
    output: &mut [MaybeUninit<T>],
    pool: Option<&TaskPool>,
    f: impl Fn(usize) -> T + Sync,
) {
    let Some(pool) = pool.filter(|_| output.len() >= 16_384) else {
        for (i, item) in output.iter_mut().enumerate() {
            item.write(f(i));
        }
        return;
    };
    let chunk_size = output.len().div_ceil(pool.thread_num().clamp(1, 8));
    let f = &f;
    // Scope cooperatively executes queued tasks, including when preparation is
    // called by a worker in this same pool. This cannot exhaust a small pool.
    pool.scope(|scope| {
        for (chunk, output) in output.chunks_mut(chunk_size).enumerate() {
            scope.spawn(async move {
                for (i, item) in output.iter_mut().enumerate() {
                    item.write(f(chunk * chunk_size + i));
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_passes_match_serial_inside_saturated_pools() {
        for threads in [1, 2, 4] {
            let pool = bevy::tasks::TaskPoolBuilder::new()
                .num_threads(threads)
                .build();
            pool.scope(|scope| {
                for _ in 0..threads {
                    let pool = &pool;
                    scope.spawn(async move {
                        for len in [0, 1, 16_383, 16_384, 70_003] {
                            let value = |i| (i as f32 * 0.013).sin().to_bits();
                            assert_eq!(collect(len, Some(pool), value), collect(len, None, value));
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn incomplete_or_panicking_output_cannot_be_observed() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(2).build();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            collect(70_003, Some(&pool), |i| {
                assert_ne!(i, 40_000, "injected worker failure");
                i
            })
        }))
        .is_err());
        assert!(catch_unwind(|| Output::<u32>::new(1).finish()).is_err());
        let mut output = Output::new(70_003);
        output.extend_with(7, None, |i| i);
        output.extend_with(69_996, Some(&pool), |i| i + 7);
        assert_eq!(output.finish(), (0..70_003).collect::<Vec<_>>());
    }
}
